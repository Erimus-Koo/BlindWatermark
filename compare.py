#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Erimus'
# 用来比较修改前后的算法

from .BlindWatermark_debug import watermark as old
from .watermark import watermark as new
import logging as log
import os

# ═══════════════════════════════════════════════


log.basicConfig(level=log.DEBUG)

here = os.path.abspath(os.path.dirname(__file__))
src = os.path.join(here, 'pic/lena512.png')
wm = os.path.join(here, 'pic/wm64.png')
out_new = os.path.join(here, 'test/out_new.png')
out_old = os.path.join(here, 'test/out_old.png')
out_new_wm = os.path.join(here, 'test/out_new_wm.png')
out_old_wm = os.path.join(here, 'test/out_old_wm.png')

bwm1 = new(4399, 2333, 32)

# bwm1.embed(src=src, wm=wm, output=out_new)

# bwm1.extract(src=out_new, wm_shape=(32, 32), output=out_new_wm)

bwm2 = old(4399, 2333, 32, dwt_deep=1, color_mod='YUV')
bwm2.read_ori_img(src)
bwm2.read_wm(wm)
# bwm2.embed(out_old)
bwm2.extract(out_old, out_old_wm)
# printDir(bwm2)
