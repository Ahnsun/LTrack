# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, List

from util import box_ops
from util.misc import inverse_sigmoid
from models.structures import Boxes, Instances, pairwise_iou
from .clip import ContextDecoder, CLIPTextContextEncoder
from util.token import tokenize


def random_drop_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    if drop_probability > 0 and len(track_instances) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_idxes]
    return track_instances


class QueryInteractionBase(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt

class ClipAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt

class query_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = output_dim

        super(query_attention, self).__init__()

        self.q_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.input_dim, self.hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim, self.out_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, query, key):
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(key)

        B, L, N = Q.shape

        scaling = float(self.hidden_dim) ** -0.5
        Q = Q * scaling

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        atten_weights = torch.bmm(Q, K.transpose(1,2))
        attention = self.drop1(torch.softmax(atten_weights, dim=-1))
        attn_ouput = torch.bmm(attention, V).transpose(0,1).contiguous().view(B, L, N)
        output = self.drop2(self.out_proj(attn_ouput))

        return output

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class QueryInteractionModule(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]

            # add fp for each active track in a specific probability.
            fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
            selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

            if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
                num_fp = len(selected_active_track_instances)
                if num_fp >= len(inactive_instances):
                    fp_track_instances = inactive_instances
                else:
                    inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                    selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                    ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                    # select the fp with the largest IoU for each active track.
                    fp_indexes = ious.max(dim=0).indices

                    # remove duplicate fp.
                    fp_indexes = torch.unique(fp_indexes)
                    fp_track_instances = inactive_instances[fp_indexes]

                merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
                return merged_track_instances

            return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim//2:]
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos[:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[:, dim//2:] = query_feat

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances

class ClipQueryInteractionModule(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        # CLIP
        # query attention: 10, 11
        # only query: 10, 11
        # learnable context + query: 10, 21
        # query concat track text: 10, 10
        self.context_length = 15
        self.context_decoder = ContextDecoder(transformer_width=256, transformer_heads=4, transformer_layers=3, visual_dim=512, dropout=0.1, outdim=512)
        self.text_encoder = CLIPTextContextEncoder(context_length=15, transformer_width=512, transformer_heads=8, transformer_layers=12, embed_dim=1024)
        self.text_encoder.init_weights(pretrained='./pre_trained/RN50.pt')
        self.query_attn = query_attention(input_dim = 1024, hidden_dim = 256, output_dim = 256, dropout = 0.1)
        self.clipadapter = ClipAdapter(input_dim=1024, hidden_dim=512, dropout=0.1)
        self.featresize = FeatureResizer(input_feat_size=512, output_feat_size=256, dropout=0.1)

        # learnable textual contexts
        # self.contexts = nn.Parameter(torch.randn(1, self.context_length, 1, 512))
        # nn.init.trunc_normal_(self.contexts)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.dropout_feat3 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]

            # add fp for each active track in a specific probability.
            fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
            selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

            if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
                num_fp = len(selected_active_track_instances)
                if num_fp >= len(inactive_instances):
                    fp_track_instances = inactive_instances
                else:
                    inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                    selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                    ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                    # select the fp with the largest IoU for each active track.
                    fp_indexes = ious.max(dim=0).indices

                    # remove duplicate fp.
                    fp_indexes = torch.unique(fp_indexes)
                    fp_track_instances = inactive_instances[fp_indexes]

                merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
                return merged_track_instances

            return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances, visual_context) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim//2:]
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos[:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)

        self.base_class = ['person']
        self.trackbook = ['person by the streetlight', 'person on the lawn', 'person in front of the car', 'person on the sidewalk', 'person next to a building', 'person under the tree', 
        'person in the restaurant', 'person at the crosswalk', 'person by the signboard', 'person at bus stops', 'person in the shopping mall', 'person in a sunny day', 
        'person in the night', 'person in the crowd', 'person by the window', 'person by the traffic light', 'walking person', 'running person', 'standing person', 'sitting person', 
        'talking person', 'shopping person', 'person riding bicycle', 'person who is waiting', 'person driving car', 'person crossing the road', 'person who is looking around',
        'person who is drinking', 'person who is eating', 'person who is smiling', 'person who is meeting', 'person watching phones', 'man in a suit', 'woman with long hair', 
        'person carrying a bag', 'person wearing glasses', 'woman with a scarf', 'person in white', 'person in black', 'perosn in orange', 'person in blue', 'person in gray', 
        'person in red', 'occluded person', 'person in a hat', 'person in jeans', 'person in the coat', 'woman in a dress', 'person with brown hair', 'woman in high heels', 'bald man',
        'person in white shirts', 'person in plaid shirts', 'person with curly hair', 'person in green', 'woman laying on the floor', 'person raising her arms', 'person lifting her leg', 
        'person on the right', 'person on the left', 'person being the second from the left',
        'person being the third from the left', 'person being the second from the right', 'person being the third from the right', 'person at the front of the line', 'person with reddish brown hair',
        'person with grey hair', 'person with yellow-brown hair', 'person with black hair', 'person who is shaking heads', 'person who are twisting', 'person in white T-shirts',
        'person in black T-shitts', 'person who is jumping', 'person who is pouting ass', 'person who is shaking hands', 'person in the middle', 'person squatting on the floor',
        'person who is occluded', 'person stretching arms', 'person located on top', 'person located below', 'person on one knee', 'person whose back is turned', 'person with straight hair',
        'person with curly hair', 'person with red hair', 'person in a golden hat', 'person dancing ballet', 'person on tiptoe', 'person spinning in circles', 'person pacing', 'a person bowing',
        'person in high heels', 'person in white shoes', 'person with long hair', 'person with short hair']
        query_context = self.context_decoder(query_feat, visual_context).unsqueeze(1)
        self.texts = torch.cat([tokenize(c, context_length=self.context_length-1) for c in self.base_class]).to(query_context.device).long()
        self.track_texts = torch.cat([tokenize('A photo of a ' + c, context_length=self.context_length) for c in self.trackbook]).to(query_context.device).long()
        # B, L, K, C = query_context.shape

        # text_embeddings = self.text_encoder(self.texts, torch.cat([self.contexts.expand(B, -1, K, -1), query_context], dim=1), 'querytext')    # concat a learnable context
        # text_embeddings = self.text_encoder(self.texts, query_context, 'querytext')  # only query context
        # text_embeddings2 = self.text_encoder(self.track_texts, None, 'onlytext')     # onlt text context
        text_embeddings, text_embeddings2 = self.text_encoder([self.track_texts, self.texts], query_context, 'query_concat_text')   # query concat track text
        adapter_embedding = self.clipadapter(torch.cat([text_embeddings, text_embeddings2], dim=1))

        query_feat3 = self.query_attn(adapter_embedding[:, :text_embeddings.shape[1], :], adapter_embedding[:, text_embeddings.shape[1]:, :])
        # query_feat3 = self.query_attn(text_embeddings, text_embeddings2)
        # query_feat3 = self.dropout_feat3(self.activation(self.linear_feat3(text_embeddings)))
        # query_feat = self.linear_feat3(torch.cat([query_feat, query_feat3.squeeze(0)], dim=-1))
        query_feat = self.featresize(torch.cat([query_feat, query_feat3.squeeze(0)], dim=-1))

        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[:, dim//2:] = query_feat

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances, data['context'])
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances

class ClipEncQueryInteractionModule(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        # CLIP
        # query attention: 10, 11
        # only query: 10, 11
        # learnable context + query: 10, 21
        # query concat track text: 10, 10
        self.context_length = 15
        self.context_decoder = ContextDecoder(transformer_width=256, transformer_heads=4, transformer_layers=3, visual_dim=256, dropout=0.1, outdim=512)
        self.text_encoder = CLIPTextContextEncoder(context_length=15, transformer_width=512, transformer_heads=8, transformer_layers=12, embed_dim=1024)
        self.text_encoder.init_weights(pretrained='./pre_trained/RN50.pt')
        self.query_attn = query_attention(input_dim = 1024, hidden_dim = 256, output_dim = 256, dropout = 0.1)
        self.clipadapter = ClipAdapter(input_dim=1024, hidden_dim=512, dropout=0.1)
        self.featresize = FeatureResizer(input_feat_size=512, output_feat_size=256, dropout=0.1)

        # learnable textual contexts
        # self.contexts = nn.Parameter(torch.randn(1, self.context_length, 1, 512))
        # nn.init.trunc_normal_(self.contexts)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.dropout_feat3 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]

            # add fp for each active track in a specific probability.
            fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
            selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

            if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
                num_fp = len(selected_active_track_instances)
                if num_fp >= len(inactive_instances):
                    fp_track_instances = inactive_instances
                else:
                    inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                    selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                    ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                    # select the fp with the largest IoU for each active track.
                    fp_indexes = ious.max(dim=0).indices

                    # remove duplicate fp.
                    fp_indexes = torch.unique(fp_indexes)
                    fp_track_instances = inactive_instances[fp_indexes]

                merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
                return merged_track_instances

            return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances, visual_context) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim//2:]
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos[:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)

        self.base_class = ['person']
        self.trackbook = ['person by the streetlight', 'person on the lawn', 'person in front of the car', 'person on the sidewalk', 'person next to a building', 'person under the tree', 
        'person in the restaurant', 'person at the crosswalk', 'person by the signboard', 'person at bus stops', 'person in the shopping mall', 'person in a sunny day', 
        'person in the night', 'person in the crowd', 'person by the window', 'person by the traffic light', 'walking person', 'running person', 'standing person', 'sitting person', 
        'talking person', 'shopping person', 'person riding bicycle', 'person who is waiting', 'person driving car', 'person crossing the road', 'person who is looking around',
        'person who is drinking', 'person who is eating', 'person who is smiling', 'person who is meeting', 'person watching phones', 'man in a suit', 'woman with long hair', 
        'person carrying a bag', 'person wearing glasses', 'woman with a scarf', 'person in white', 'person in black', 'perosn in orange', 'person in blue', 'person in gray', 
        'person in red', 'occluded person', 'person in a hat', 'person in jeans', 'person in the coat', 'woman in a dress', 'person with brown hair', 'woman in high heels', 'bald man',
        'person in white shirts', 'person in plaid shirts', 'person with curly hair', 'person in green']

        
        self.dancebook = ['woman laying on the floor', 'person raising her arms', 'person lifting her leg', 'person on the right', 'person on the left', 'person being the second from the left',
        'person being the third from the left', 'person being the second from the right', 'person being the third from the right', 'person at the front of the line', 'person with reddish brown hair',
        'person with grey hair', 'person with yellow-brown hair', 'person with black hair', 'person who is shaking heads', 'person who are twisting', 'person in white T-shirts',
        'person in black T-shitts', 'person who is jumping', 'person who is pouting ass', 'person who is shaking hands', 'person in the middle', 'person squatting on the floor',
        'person who is occluded', 'person stretching arms', 'person located on top', 'person located below', 'person on one knee', 'person whose back is turned', 'person with straight hair',
        'person with curly hair', 'person with red hair', 'person in a golden hat', 'person dancing ballet', 'person on tiptoe', 'person spinning in circles', 'person pacing', 'a person bowing',
        'person with green hair', 'person with glasses', 'person in jeans', 'person in a dress', 'person in high heels', 'person in white shoes', 'person with long hair', 'person with short hair']
        query_context = self.context_decoder(query_feat, visual_context).unsqueeze(1)
        self.texts = torch.cat([tokenize(c, context_length=self.context_length-1) for c in self.base_class]).to(query_context.device).long()
        self.track_texts = torch.cat([tokenize('A photo of a ' + c, context_length=self.context_length) for c in self.trackbook]).to(query_context.device).long()
        # B, L, K, C = query_context.shape

        # text_embeddings = self.text_encoder(self.texts, torch.cat([self.contexts.expand(B, -1, K, -1), query_context], dim=1), 'querytext')    # concat a learnable context
        # text_embeddings = self.text_encoder(self.texts, query_context, 'querytext')  # only query context
        # text_embeddings2 = self.text_encoder(self.track_texts, None, 'onlytext')     # onlt text context
        text_embeddings, text_embeddings2 = self.text_encoder([self.track_texts, self.texts], query_context, 'query_concat_text')   # query concat track text
        adapter_embedding = self.clipadapter(torch.cat([text_embeddings, text_embeddings2], dim=1))

        query_feat3 = self.query_attn(adapter_embedding[:, :text_embeddings.shape[1], :], adapter_embedding[:, text_embeddings.shape[1]:, :])
        # query_feat3 = self.query_attn(text_embeddings, text_embeddings2)
        # query_feat3 = self.dropout_feat3(self.activation(self.linear_feat3(text_embeddings)))
        query_feat = self.featresize(torch.cat([query_feat, query_feat3.squeeze(0)], dim=-1))

        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[:, dim//2:] = query_feat

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances, data['context'])
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances

class ClipEncQueryInteractionModuleV2(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        # CLIP
        # query attention: 10, 11
        # only query: 10, 11
        # learnable context + query: 10, 21
        # query concat track text: 10, 10
        self.context_length = 15
        self.context_decoder = ContextDecoder(transformer_width=256, transformer_heads=4, transformer_layers=3, visual_dim=256, dropout=0.1, outdim=512)
        self.text_encoder = CLIPTextContextEncoder(context_length=15, transformer_width=512, transformer_heads=8, transformer_layers=12, embed_dim=1024)
        self.text_encoder.init_weights(pretrained='./pre_trained/RN50.pt')
        self.query_attn = query_attention(input_dim = 1024, hidden_dim = 256, output_dim = 256, dropout = 0.1)
        self.clipadapter = ClipAdapter(input_dim=1024, hidden_dim=512, dropout=0.1)
        self.featresize = FeatureResizer(input_feat_size=512, output_feat_size=256, dropout=0.1)

        # learnable textual contexts
        # self.contexts = nn.Parameter(torch.randn(1, self.context_length, 1, 512))
        # nn.init.trunc_normal_(self.contexts)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.dropout_feat3 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]

            # add fp for each active track in a specific probability.
            fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
            selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

            if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
                num_fp = len(selected_active_track_instances)
                if num_fp >= len(inactive_instances):
                    fp_track_instances = inactive_instances
                else:
                    inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                    selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                    ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                    # select the fp with the largest IoU for each active track.
                    fp_indexes = ious.max(dim=0).indices

                    # remove duplicate fp.
                    fp_indexes = torch.unique(fp_indexes)
                    fp_track_instances = inactive_instances[fp_indexes]

                merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
                return merged_track_instances

            return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances, visual_context) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim//2:]
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos[:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)

        self.base_class = ['person']
        self.trackbook = ['person by the streetlight', 'person on the lawn', 'person in front of the car', 'person on the sidewalk', 'person next to a building', 'person under the tree', 
        'person in the restaurant', 'person at the crosswalk', 'person by the signboard', 'person at bus stops', 'person in the shopping mall', 'person in a sunny day', 
        'person in the night', 'person in the crowd', 'person by the window', 'person by the traffic light', 'walking person', 'running person', 'standing person', 'sitting person', 
        'talking person', 'shopping person', 'person riding bicycle', 'person who is waiting', 'person driving car', 'person crossing the road', 'person who is looking around',
        'person who is drinking', 'person who is eating', 'person who is smiling', 'person who is meeting', 'person watching phones', 'man in a suit', 'woman with long hair', 
        'person carrying a bag', 'person wearing glasses', 'woman with a scarf', 'person in white', 'person in black', 'perosn in orange', 'person in blue', 'person in gray', 
        'person in red', 'occluded person', 'person in a hat', 'person in jeans', 'person in the coat', 'woman in a dress', 'person with brown hair', 'woman in high heels', 'bald man',
        'person in white shirts', 'person in plaid shirts', 'person with curly hair', 'person in green']

        
        self.dancebook = ['woman laying on the floor', 'person raising her arms', 'person lifting her leg', 'person on the right', 'person on the left', 'person being the second from the left',
        'person being the third from the left', 'person being the second from the right', 'person being the third from the right', 'person at the front of the line', 'person with reddish brown hair',
        'person with grey hair', 'person with yellow-brown hair', 'person with black hair', 'person who is shaking heads', 'person who are twisting', 'person in white T-shirts',
        'person in black T-shitts', 'person who is jumping', 'person who is pouting ass', 'person who is shaking hands', 'person in the middle', 'person squatting on the floor',
        'person who is occluded', 'person stretching arms', 'person located on top', 'person located below', 'person on one knee', 'person whose back is turned', 'person with straight hair',
        'person with curly hair', 'person with red hair', 'person in a golden hat', 'person dancing ballet', 'person on tiptoe', 'person spinning in circles', 'person pacing', 'a person bowing',
        'person with green hair', 'person with glasses', 'person in jeans', 'person in a dress', 'person in high heels', 'person in white shoes', 'person with long hair', 'person with short hair']
        query_context = self.context_decoder(query_feat, visual_context).unsqueeze(1)
        self.texts = torch.cat([tokenize(c, context_length=self.context_length-1) for c in self.base_class]).to(query_context.device).long()
        self.track_texts = torch.cat([tokenize('A photo of a ' + c, context_length=self.context_length) for c in self.trackbook]).to(query_context.device).long()
        # B, L, K, C = query_context.shape

        # text_embeddings = self.text_encoder(self.texts, torch.cat([self.contexts.expand(B, -1, K, -1), query_context], dim=1), 'querytext')    # concat a learnable context
        # text_embeddings = self.text_encoder(self.texts, query_context, 'querytext')  # only query context
        # text_embeddings2 = self.text_encoder(self.track_texts, None, 'onlytext')     # onlt text context
        text_embeddings, text_embeddings2 = self.text_encoder([self.track_texts, self.texts], query_context, 'query_concat_text')   # query concat track text
        adapter_embedding = self.clipadapter(torch.cat([text_embeddings, text_embeddings2], dim=1))

        query_feat3 = self.query_attn(adapter_embedding[:, :text_embeddings.shape[1], :], adapter_embedding[:, text_embeddings.shape[1]:, :])
        # query_feat3 = self.query_attn(text_embeddings, text_embeddings2)
        # query_feat3 = self.dropout_feat3(self.activation(self.linear_feat3(text_embeddings)))
        # query_feat = self.featresize(torch.cat([query_feat, query_feat3.squeeze(0)], dim=-1))

        query_feat = self.norm_feat(query_feat3)
        track_instances.query_pos[:, dim//2:] = query_feat

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances, data['context'])
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances

class ClipEncQueryInteractionModule_Ada(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        # CLIP
        # query attention: 10, 11
        # only query: 10, 11
        # learnable context + query: 10, 21
        # query concat track text: 10, 10
        self.context_length = 15
        self.context_decoder = ContextDecoder(transformer_width=256, transformer_heads=4, transformer_layers=3, visual_dim=256, dropout=0.1, outdim=512)
        self.text_encoder = CLIPTextContextEncoder(context_length=15, transformer_width=512, transformer_heads=8, transformer_layers=12, embed_dim=1024)
        self.text_encoder.init_weights(pretrained='./pre_trained/RN50.pt')
        self.query_attn = query_attention(input_dim = 1024, hidden_dim = 256, output_dim = 256, dropout = 0.1)
        # self.clipadapter = ClipAdapter(input_dim=1024, hidden_dim=512, dropout=0.1)
        self.featresize = FeatureResizer(input_feat_size=512, output_feat_size=256, dropout=0.1)

        # learnable textual contexts
        # self.contexts = nn.Parameter(torch.randn(1, self.context_length, 1, 512))
        # nn.init.trunc_normal_(self.contexts)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.linear_feat3 = nn.Linear(512, 256)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.dropout_feat3 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]

            # add fp for each active track in a specific probability.
            fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
            selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

            if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
                num_fp = len(selected_active_track_instances)
                if num_fp >= len(inactive_instances):
                    fp_track_instances = inactive_instances
                else:
                    inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                    selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                    ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                    # select the fp with the largest IoU for each active track.
                    fp_indexes = ious.max(dim=0).indices

                    # remove duplicate fp.
                    fp_indexes = torch.unique(fp_indexes)
                    fp_track_instances = inactive_instances[fp_indexes]

                merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
                return merged_track_instances

            return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances, visual_context) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim//2:]
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos[:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)

        self.base_class = ['person']
        self.trackbook = ['person by the streetlight', 'person on the lawn', 'person in front of the car', 'person on the sidewalk', 'person next to a building', 'person under the tree', 
        'person in the restaurant', 'person at the crosswalk', 'person by the signboard', 'person at bus stops', 'person in the shopping mall', 'person in a sunny day', 
        'person in the night', 'person in the crowd', 'person by the window', 'person by the traffic light', 'walking person', 'running person', 'standing person', 'sitting person', 
        'talking person', 'shopping person', 'person riding bicycle', 'person who is waiting', 'person driving car', 'person crossing the road', 'person who is looking around',
        'person who is drinking', 'person who is eating', 'person who is smiling', 'person who is meeting', 'person watching phones', 'man in a suit', 'woman with long hair', 
        'person carrying a bag', 'person wearing glasses', 'woman with a scarf', 'person in white', 'person in black', 'perosn in orange', 'person in blue', 'person in gray', 
        'person in red', 'occluded person', 'person in a hat', 'person in jeans', 'person in the coat', 'woman in a dress', 'person with brown hair', 'woman in high heels', 'bald man',
        'person in white shirts', 'person in plaid shirts', 'person with curly hair', 'person in green']
        query_context = self.context_decoder(query_feat, visual_context).unsqueeze(1)
        self.texts = torch.cat([tokenize(c, context_length=self.context_length-1) for c in self.base_class]).to(query_context.device).long()
        self.track_texts = torch.cat([tokenize('A photo of a ' + c, context_length=self.context_length) for c in self.trackbook]).to(query_context.device).long()
        # B, L, K, C = query_context.shape

        # text_embeddings = self.text_encoder(self.texts, torch.cat([self.contexts.expand(B, -1, K, -1), query_context], dim=1), 'querytext')    # concat a learnable context
        # text_embeddings = self.text_encoder(self.texts, query_context, 'querytext')  # only query context
        # text_embeddings2 = self.text_encoder(self.track_texts, None, 'onlytext')     # onlt text context
        text_embeddings, text_embeddings2 = self.text_encoder([self.track_texts, self.texts], query_context, 'query_concat_text')   # query concat track text
        # adapter_embedding = self.clipadapter(torch.cat([text_embeddings, text_embeddings2], dim=1))

        # query_feat3 = self.query_attn(adapter_embedding[:, :text_embeddings.shape[1], :], adapter_embedding[:, text_embeddings.shape[1]:, :])
        query_feat3 = self.query_attn(text_embeddings, text_embeddings2)
        # query_feat3 = self.dropout_feat3(self.activation(self.linear_feat3(text_embeddings)))
        # query_feat = self.linear_feat3(torch.cat([query_feat, query_feat3.squeeze(0)], dim=-1))
        # query_feat = self.featresize(torch.cat([query_feat, query_feat3.squeeze(0)], dim=-1))

        query_feat = self.norm_feat(query_feat3)
        track_instances.query_pos[:, dim//2:] = query_feat

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances, data['context'])
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances

def build(args, layer_name, dim_in, hidden_dim, dim_out):
    interaction_layers = {
        'QIM': QueryInteractionModule,
        'CLIP_QIM': ClipQueryInteractionModule,
        'CLIP_QIM_ENC': ClipEncQueryInteractionModule,
        'CLIP_QIM_ENC_V2': ClipEncQueryInteractionModuleV2,
        'CLIP_QIM_ADA': ClipEncQueryInteractionModule_Ada,
    }
    assert layer_name in interaction_layers, 'invalid query interaction layer: {}'.format(layer_name)
    return interaction_layers[layer_name](args, dim_in, hidden_dim, dim_out)
