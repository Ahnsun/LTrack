# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr import build as build_deformable_detr
from .motr import build as build_motr
from .motr_text2query import build as build_motr_text_query
from .text2query_aspp import build as build_text_query_aspp
from .text2query_attnpool import build as build_text_query_attnpool
from .text2query_enc import build as build_text_query_enc


def build_model(args):
    arch_catalog = {
        'deformable_detr': build_deformable_detr,
        'motr': build_motr,
        'text_query': build_motr_text_query,
        'text_query_aspp': build_text_query_aspp,
        'text_query_attnpool': build_text_query_attnpool,
        'text_query_enc': build_text_query_enc,
    }
    assert args.meta_arch in arch_catalog, 'invalid arch: {}'.format(args.meta_arch)
    build_func = arch_catalog[args.meta_arch]
    return build_func(args)