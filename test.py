#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Erimus'

import numpy as np
import cv2
import os
import logging as log
import re
from .watermark import watermark
from .cv2_tools import cv_text

# ═══════════════════════════════════════════════
# 创建输出文件夹
here = os.path.abspath(os.path.dirname(__file__))
test_folder = os.path.join(here, 'test')
plot_folder = os.path.join(test_folder, 'plot')
for _folder in [test_folder, plot_folder]:
    if not os.path.exists(_folder):
        os.mkdir(_folder)
# ═══════════════════════════════════════════════


def format_filename(*, src_img, wm, wm_seed, block_seed, mod, fmt, jpg_quality):
    # 参数设定
    here = os.path.abspath(os.path.dirname(__file__))
    if not src_img.endswith('.jpg'):
        src_img += '.png'  # 默认png后缀
    img_name = src_img[:-4]
    img_file = os.path.join(here, f'pic/{src_img}')
    wm_file = os.path.join(here, f'pic/wm{wm}.png')

    wm_map_method = 1  # 水印映射方式

    src_shape = cv2.imread(img_file).shape[:2]  # 原图短边（自动计算）
    temp = watermark()
    block, dwt_deep = temp.auto_block(src_shape)
    kwargs = {'wm_seed': wm_seed, 'block_seed': block_seed, 'mod': mod,
              'wm_map_method': wm_map_method}
    # dwt_deep, block = 0, 1  # 指定
    # kwargs.update({'dwt_deep': dwt_deep, 'block': block})

    # 格式化输出文件名
    _fmt = fmt + (f'{jpg_quality}' if fmt == 'jpg' else '')
    _wms = '' if wm_seed is None else f'_wms{wm_seed}'
    _blks = '' if block_seed is None else f'_blks{block_seed}'
    out = (f'test/{img_name}_wm{wm}_map{wm_map_method}_dwt{dwt_deep}'
           f'_block{block}_mod{mod}_{_fmt}{_wms}{_blks}')
    out_img = os.path.join(here, f'{out}.{fmt}')
    out_wm = os.path.join(here, f'{out}_wm.png')

    return img_file, wm_file, out_img, out_wm, kwargs


def jpg_quality_mod_grid(src_img, wm, wm_seed, block_seed, jpg_limit=50):
    jpg_list = list(range(jpg_limit, 101, 10))
    mod_list = list(range(16, 33, 4))

    # 新建画布
    wm_size = 64
    w = (len(mod_list) + 1) * wm_size
    h = (len(jpg_list) + 1) * wm_size
    plot = np.zeros((w, h, 3), np.uint8)
    plot.fill(255)
    cvtext = cv_text()

    for jidx, jpg in enumerate(jpg_list):
        x = (jidx + 1) * wm_size  # 当前行
        y = wm_size - 4
        plot = cvtext.put(plot, str(jpg), x, y, align=('left', 'bottom'))
        for midx, mod in enumerate(mod_list):
            print(f'{jpg=} | {mod=}')
            if jidx == 0:
                _x, _y = wm_size, (midx + 1) * wm_size
                plot = cvtext.put(plot, str(mod), _x, _y, align=('right', 'top'))
            y = int((midx + 1) * wm_size)  # 当前列

            kw = {'src_img': src_img, 'wm': wm, 'wm_seed': wm_seed,
                  'block_seed': block_seed, 'mod': mod,
                  'fmt': 'jpg', 'jpg_quality': jpg}
            img_file, wm_file, out_img, out_wm, kwargs = format_filename(**kw)
            bwm = watermark(**kwargs)
            if not os.path.exists(out_img):
                bwm.embed(src=img_file, wm=wm_file,
                          output=out_img, jpg_quality=jpg)

            if not os.path.exists(out_wm):
                bwm.extract(src=out_img, output=out_wm)

            wm_img = cv2.imread(out_wm)
            plot[y:y + wm_size, x:x + wm_size] = wm_img

    # cv2.imshow('image', plot)
    # cv2.waitKey(0)
    path, file = os.path.split(out_img)
    info, ext = os.path.splitext(file)
    info = re.sub(r'_jpg\d+', '', info)
    info = re.sub(r'_mod\d+', '', info)
    info = (f'{info}_jpg{jpg_list[0]}-{jpg_list[-1]}'
            f'_mod{mod_list[0]}-{mod_list[-1]}.png')

    # 绘制标题
    brk = info[len(info) // 2:].index('_') + len(info) // 2 + 1  # 后半段第一个下划线
    plot = cvtext.put(plot, info[:brk], 64, 5, 10, align=('left', 'top'))
    plot = cvtext.put(plot, info[brk:-4], 64, 25, 10, align=('left', 'top'))

    plot_name = os.path.join(path, 'plot', info)
    print(f'{plot_name = }')
    cv2.imwrite(plot_name, plot)


def jpg_quality_mod_grid_batch():
    from multiprocessing import Pool, Value, Process, Manager

    params = []
    for src in ['lena512', 'mini512', 'mini512i', 'noy512',
                'comic.jpg', 'drama.jpg', 'manga.jpg', 'paint.jpg']:
        for wm in ['64', '64i']:
            params.append((src, wm, 1234, 5678, 30))
    print(f'{len(params) = }')
    p = Pool(len(params))  # 设置进程数
    for arg in params:
        p.apply_async(jpg_quality_mod_grid, arg)
    p.close()
    p.join()


def test_single_file():
    jpg_quality = 50
    kw = {
        'src_img': 'lena512',  # 原图
        # 'src_img': 'mini512',
        # 'src_img' :'noy512',
        'wm': '32',
        'wm_seed': 1234,  # 水印随机种子 (0~4294967296)
        'block_seed': 5678,  # block随机种子 (0~4294967296)
        'mod': 24,  # 对齐除数
        'fmt': 'png',  # 输出格式
        'fmt': 'jpg',  # 输出格式
        'jpg_quality': jpg_quality,    # 输出jpg质量
    }
    img, wm, out_img, out_wm, kwargs = format_filename(**kw)

    # 加水印
    bwm = watermark(**kwargs)
    bwm.embed(src=img, wm=wm, output=out_img, jpg_quality=jpg_quality)

    # 解水印
    bwm = watermark(**kwargs)
    bwm.extract(src=out_img, output=out_wm)


# ═══════════════════════════════════════════════

if __name__ == '__main__':

    # test_single_file()

    jpg_quality_mod_grid('comic.jpg', '64', 1234, 5678, jpg_limit=30)  # 单文件

    # jpg_quality_mod_grid_batch()  # 所有
