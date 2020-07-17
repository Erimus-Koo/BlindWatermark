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
from multiprocessing import Pool, Value, Process, Manager

# ═══════════════════════════════════════════════
# 创建输出文件夹
here = os.path.abspath(os.path.dirname(__file__))
test_folder = os.path.join(here, 'test')
plot_folder = os.path.join(test_folder, 'plot')
for _folder in [test_folder, plot_folder]:
    if not os.path.exists(_folder):
        os.mkdir(_folder)
# ═══════════════════════════════════════════════
SAMPLE_LIST = ['lena512', 'mini512', 'mini512i', 'noy512',
               'comic.jpg', 'drama.jpg', 'manga.jpg', 'paint.jpg']
# ═══════════════════════════════════════════════


def format_filename(*, src_img, wm, wm_seed=1234, block_seed=5678, mod=24,
                    fmt='png', jpg_quality=80, block='auto', dwt_deep=3):
    # 参数设定
    here = os.path.abspath(os.path.dirname(__file__))
    if not src_img.endswith('.jpg'):
        src_img += '.png'  # 默认png后缀
    img_name = src_img[:-4]
    img_file = os.path.join(here, f'pic/{src_img}')
    wm_file = os.path.join(here, f'pic/wm{wm}.png')

    wm_map_method = 1  # 水印映射方式

    if block == 'auto':
        src_shape = cv2.imread(img_file).shape[:2]  # 原图短边（自动计算）
        temp = watermark()
        block, dwt_deep = temp.auto_block(src_shape)
    # dwt_deep, block = 0, 1  # 指定

    kwargs = {'wm_seed': wm_seed, 'block_seed': block_seed, 'mod': mod,
              'dwt_deep': dwt_deep, 'block': block,
              'wm_map_method': wm_map_method}

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
    cvtxt = cv_text()
    fsize = wm_size / 6

    for m_idx, mod in enumerate(mod_list):
        x = wm_size - 4  # 当前行
        y = (m_idx + 1) * wm_size
        plot = cvtxt.put(plot, f'mod{mod}', x, y,
                          size=fsize, align=('top', 'right'))

        caculated_img = None  # 同一个配置只计算一次 其余直接保存为不同精度jpg
        for j_idx, jpg in enumerate(jpg_list):
            print(f'{mod=} | {jpg=}')
            if m_idx == 0:
                _x, _y = (j_idx + 1) * wm_size, wm_size - 4
                plot = cvtxt.put(plot, str(jpg), _x, _y,
                                  size=fsize, align=('bottom', 'left'))
            x = int((j_idx + 1) * wm_size)  # 当前列

            kw = {'src_img': src_img, 'wm': wm, 'wm_seed': wm_seed,
                  'block_seed': block_seed, 'mod': mod,
                  # 'dwt_deep': 3, 'block': 1,
                  'fmt': 'jpg', 'jpg_quality': jpg}
            img_file, wm_file, out_img, out_wm, kwargs = format_filename(**kw)
            bwm = watermark(**kwargs)
            if not os.path.exists(out_img):
                if caculated_img is None:
                    caculated_img = bwm.embed(src=img_file, wm=wm_file,
                                              output=out_img, jpg_quality=jpg)
                    print(f'!!! First caculated {block=} | {multiple=}')
                else:
                    bwm.save_image(out_img, caculated_img, jpg)

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
    info = (f'{info}_mod{mod_list[0]}-{mod_list[-1]}'
            f'_jpg{jpg_list[0]}-{jpg_list[-1]}.png')

    # 绘制标题
    brk = info[len(info) // 2:].index('_') + len(info) // 2 + 1  # 后半段第一个下划线
    plot = cvtxt.put(plot, info[:brk], 64, 4,
                      size=fsize, align=('top', 'left'))
    plot = cvtxt.put(plot, info[brk:-4], 64, int(4 + fsize * 1.5),
                      size=fsize, align=('top', 'left'))

    plot_name = os.path.join(path, 'plot', info)
    print(f'{plot_name = }')
    cv2.imwrite(plot_name, plot)


def jpg_quality_mod_grid_batch():
    params = []
    for src in SAMPLE_LIST:
        for wm in ['64', '64i']:
            params.append((src, wm, 1234, 5678, 30))
    print(f'{len(params) = }')
    p = Pool(len(params))  # 设置进程数
    for arg in params:
        p.apply_async(jpg_quality_mod_grid, arg)
    p.close()
    p.join()


def multiple_test(src_img, wm, dwt_deep=1):
    kw = {'src_img': src_img, 'wm': wm}
    img_file, wm_file, out_img, out_wm, kwargs = format_filename(**kw)
    src_size = min(cv2.imread(img_file).shape[:2])
    wm_size = cv2.imread(wm_file).shape[0]
    wm_h, wm_w = cv2.imread(wm_file).shape[:2]
    fsize = wm_size / 6

    # 用deep=0，512原图，计算不同block来应对不同multiple。
    jpg_list = list(range(30, 101, 10))

    # 最大block
    max_block = int(src_size / 2**dwt_deep / wm_size)
    block_list = range(1, max_block + 1)

    # 新建画布
    w = (len(jpg_list) + 1) * wm_size
    h = (len(block_list) + 1) * wm_size
    plot = np.zeros((h, w, 3), np.uint8)
    plot.fill(255)
    cvtxt = cv_text()

    for b_idx, block in enumerate(block_list):  # 先行后列
        x = wm_size - 4  # 文字右侧
        y = (b_idx + 1) * wm_size  # 当前行

        # 计算multiple和block
        multiple = src_size / 2**dwt_deep / block / wm_size
        plot = cvtxt.put(plot, f'd{dwt_deep}b{block}', x, y,
                          size=fsize, align=('top', 'right'))
        plot = cvtxt.put(plot, f'f{2**dwt_deep*block}', x, int(y + fsize * 1.5),
                          size=fsize, align=('top', 'right'))
        plot = cvtxt.put(plot, f'm{multiple:.1f}', x, int(y + fsize * 3),
                          size=fsize, align=('top', 'right'))

        caculated_img = None  # 同一个配置只计算一次 其余直接保存为不同精度jpg
        for j_idx, jpg in enumerate(jpg_list):
            print(f'{block=} | {multiple=:.1f} | {jpg=}')
            if b_idx == 0:
                _x, _y = (j_idx + 1) * wm_size, wm_size - 4
                plot = cvtxt.put(plot, str(jpg), _x, _y,
                                  size=fsize, align=('bottom', 'left'))  # 打印列名
            x = (j_idx + 1) * wm_size  # 当前列

            kw = {'src_img': src_img, 'wm': wm,
                  'dwt_deep': dwt_deep, 'block': block,
                  'fmt': 'jpg', 'jpg_quality': jpg}
            img_file, wm_file, out_img, out_wm, kwargs = format_filename(**kw)
            bwm = watermark(**kwargs)
            if not os.path.exists(out_img):
                if caculated_img is None:
                    print(kw)
                    caculated_img = bwm.embed(src=img_file, wm=wm_file,
                                              output=out_img, jpg_quality=jpg)
                    print(f'!!! First caculated {block=} | {multiple=}')
                else:
                    bwm.save_image(out_img, caculated_img, jpg)

            if not os.path.exists(out_wm):
                bwm.extract(src=out_img, wm_w=wm_w, wm_h=wm_h, output=out_wm)

            wm_img = cv2.imread(out_wm)
            plot[y:y + wm_h, x:x + wm_w] = wm_img

            # cv2.imshow('image', plot)
            # cv2.waitKey(0)
    path, file = os.path.split(out_img)
    info, ext = os.path.splitext(file)
    info = re.sub(r'_jpg\d+', '', info)
    info = re.sub(r'_block\d+', '', info)
    info = re.sub(r'_dwt\d+', '', info)
    info = (f'{info}_multiple_test_dwt{dwt_deep}.png')

    # 绘制标题
    brk = info[len(info) // 2:].index('_') + len(info) // 2 + 1  # 后半段第一个下划线
    plot = cvtxt.put(plot, info[:brk], 4, 4, fsize, align=('top', 'left'))
    plot = cvtxt.put(plot, info[brk:-4], 4, int(4 + fsize * 1.5),
                      size=fsize, align=('top', 'left'))

    plot_name = os.path.join(path, 'plot', info)
    print(f'{plot_name = }')
    cv2.imwrite(plot_name, plot)


def multiple_test_batch():
    params = []
    for src in SAMPLE_LIST:
        for dwt_deep in range(2):
            params.append((src, 64, dwt_deep))
    print(f'{len(params) = }')
    p = Pool(len(params))  # 设置进程数
    for arg in params:
        p.apply_async(multiple_test, arg)
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
        # 'dwt_deep': 0,
        # 'block': 7,
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

    test_single_file()

    # jpg_quality_mod_grid('comic.jpg', '64', 1234, 5678, jpg_limit=30)  # 单文件

    # jpg_quality_mod_grid_batch()  # 所有

    # multiple_test('lena512', '32')  # 单文件

    # multiple_test_batch()
