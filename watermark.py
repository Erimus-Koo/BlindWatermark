#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Erimus'
'''
这个部分主要参考了
https://github.com/fire-keeper/BlindWatermark

修改的部分详见readme
请尽量使用64x64像素的黑白水印图
'''

import numpy as np
import cv2
from pywt import dwt2, idwt2
import os
import logging as log

# ═══════════════════════════════════════════════


class watermark():
    def __init__(self,
                 wm_seed=None,  # 水印随机种子 (0~4294967296)
                 block_seed=None,  # block随机种子 (0~4294967296)
                 mod=24,  # 可传入int或tuple(多次mod)
                 block='auto',  # (宽, 高) or 'auto' 自动目前仅支持64x64水印
                 dwt_deep=3,  # 小波次数 (最大值 后续如果图片过小可能自动降低)
                 color_mode='YUV',  # 嵌入水印的通道 'YUV', 'RGB'
                 wm_map_method=1  # 水印和block的映射方式 1平铺 2纵横分别映射
                 ):
        # svd逆运算不是正方形有点搞 干脆先用正方形
        self.block = block
        self.block_shape = (block, block) if isinstance(block, int) else block
        self.wm_seed = wm_seed
        self.block_seed = block_seed
        self.mod = [mod] if isinstance(mod, int) else mod
        self.color_mode = color_mode
        self.dwt_deep = dwt_deep
        self.wm_map_method = wm_map_method

    def load_image(self, src_file):
        '''
        读取原图，返回cv2对象。
        '''
        img = cv2.imread(src_file).astype(np.float32)

        # 转换色彩空间
        if self.color_mode.upper() == 'YUV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # 绘制原图
        # imshow('img', img)
        return img

    def save_image(self, output_filename, img, jpg_quality):
        # imshow('img', img)
        if self.color_mode.upper() == 'YUV':
            img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_YUV2BGR)
        # imshow('Convert', img)
        img[img > 255] = 255
        img[img < 0] = 0
        # imshow('Limit', img)
        cv2.imwrite(output_filename, img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        print(f'\nWatermarked image saved: {output_filename}\n')

    def dwt_image(self, img):
        '''
        将图片进行小波变换
        因为会改变图片尺寸，所以会返回原图宽高。
        返回变换后的图片。
        为了逆变换，还要返回各级的hvd数据。
        '''
        # 宽高凑整为 2**dwt_deep 的倍数(扩展画布补黑色) (zeros默认类型是float)
        src_h, src_w = img.shape[:2]
        log.debug(f'src img: w={src_w} | h={src_h}')
        dwt_unit = 2**self.dwt_deep
        if src_w % dwt_unit:  # 补横向
            w_gap = dwt_unit - src_w % dwt_unit
            right = np.ones((src_h, w_gap, 3)).astype(np.float32)
            img = np.hstack((img, right))
        if src_h % dwt_unit:  # 补纵向
            h_gap = dwt_unit - src_h % dwt_unit
            bottom = np.ones((h_gap, img.shape[1], 3)).astype(np.float32)
            img = np.vstack((img, bottom))

        # 绘制原图各通道
        # imshow(f'ori channels ({self.color_mode})', np.hstack(dwt_data[0]['a']))

        # 拆分通道 [[h*w*单通道] * 3channel] (其实小波时会把数据变为浮点)
        dwt_img = cv2.split(img)
        # imshow(f'dwt img', np.hstack(dwt_img), wait=0)

        # 小波变换(多次)
        hvd_list = []  # index=dwt_deep, defautl=src=0, 内含a,hvd,block
        for _deep in range(self.dwt_deep):
            ahvd_all = [dwt2(_channel, 'haar') for _channel in dwt_img]
            dwt_img = [a for a, hvd in ahvd_all]
            hvd = [hvd for a, hvd in ahvd_all]
            hvd_list.append(hvd)  # add this level data

            # 绘制各通道小波变换结果
            # for _idx, (_a, _hvd) in enumerate(ahvd_all):
            #     imshow(f'dwt c{_idx} AHVD', np.hstack((_a, *_hvd)), wait=0)
            # cv2.waitKey(0)
        return dwt_img, hvd_list

    def idwt_image(self, src_h, src_w, dwt_img, hvd_list):
        '''
        逆向小波，返回一个cv2图片对象。
        '''
        for hvd in hvd_list[::-1]:
            dwt_img = [idwt2((dwt_img[_i], hvd[_i]), 'haar')
                       for _i in range(len(dwt_img))]

        out_img = cv2.merge(dwt_img)
        # imshow('idwt', out_img)
        out_img = out_img[:src_h, :src_w]  # 还原原图尺寸
        return out_img

    def load_watermark(self, wm_file):
        '''
        读取水印文件，转为灰度，根据seed乱序。
        返回二维数组。
        '''
        wm = cv2.imread(wm_file, cv2.IMREAD_GRAYSCALE)
        # imshow('read watermark', wm)

        if self.wm_seed:
            wm_flatten = wm.flatten()  # 展开为一维数组
            random_wm = np.random.RandomState(self.wm_seed)
            random_wm.shuffle(wm_flatten)
            wm = np.reshape(wm_flatten, wm.shape)

        return wm

    def save_watermark(self, wm_data, wm_w, wm_h, out_wm_file, channel):
        '''
        读取水印的二维数组，根据seed逆向乱序。
        保存为图片。
        '''
        if self.wm_seed:
            for chn_idx, chn in enumerate(wm_data):
                if not channel or chn_idx == channel:
                    wm_flatten = chn.flatten()
                    wm_index = np.arange(len(wm_flatten))
                    random_wm = np.random.RandomState(self.wm_seed)
                    random_wm.shuffle(wm_index)
                    wm_flatten[wm_index] = wm_flatten.copy()
                    wm_data[chn_idx] = np.reshape(wm_flatten, (wm_h, wm_w))

        # 拼接各通道水印
        wm_data = np.hstack(wm_data) if channel is None else wm_data[channel]

        # imshow('wm', wm)
        cv2.imwrite(out_wm_file, wm_data)
        print(f'\nWatermark saved: {out_wm_file}\n')

    def check_block(self, dwt_img, wm_data):
        curve_h, curve_w = dwt_img[0].shape[:2]
        block_w, block_h = self.block_shape
        block_h_num, block_v_num = curve_w // block_w, curve_h // block_h
        wm_h, wm_w = wm_data.shape[:2]
        if ((self.wm_map_method == 1
             and block_h_num * block_v_num < wm_w * wm_h)
                or (self.wm_map_method == 2
                    and (block_h_num < wm_w or block_v_num < wm_h))):
            raise ValueError(
                f'Error: 水印的宽高超过对应的block数量\n'
                f'curve size={curve_w}x{curve_h} | block={block_w}x{block_h}\n'
                f'block num: {block_h_num}x{block_v_num} | '
                f'watermark: {wm_w}x{wm_h}')

    def auto_block(self, src_shape, wm_shape=(64, 64), multiple=4):
        '''
        自动计算block大小
        这里考虑有限保证画质，尽量 2^dwt_deep * block >=8(jpg 压缩以8px为单位)
        dwt size(256) = wm(64) * capa(1) * block(4) * deep(0)
        dwt size(512) = wm(64) * capa(1) * block(4) * deep(0)

        multiple的意义。水印约大，误差越小。反之亦然。
        在jpg50的情况下，64*2 = 32*4 = 16*8 = 边长128个block能达到较好效果。
        64px的水印需要128个block来记录，每像素用2倍边长的block来还原(4个block)。
        32px的水印也用128个block来记录，每像素点需要4倍来还原以提高辨识度。
        multiple其实是根据水印大小在动态变化的。
        multiple<1等于图片不足以承载水印的数据。
        '''
        log.debug(f'===auto block===\n{src_shape=}\n{wm_shape=}')
        multiple_target = 1
        block_list = []

        for i, edge in enumerate(src_shape):
            result = None
            # 获得所有合法的参数组合
            prm = []  # params list
            for _deep in range(self.dwt_deep, -1, -1):
                for block in range(1, edge + 1):
                    final = 2**_deep * block  # 最终映射到原图的块尺寸
                    multiple = edge / final / wm_shape[i]
                    if multiple < 1:
                        break  # quit block loop
                    prm.append({'dwt_deep': _deep, 'block': block,
                                'final': final, 'multiple': multiple})

            def mini_sort(_list, _key, reverse=True):  # 缩写sort 纯为了pep8
                return sorted(_list, key=lambda x: x[_key], reverse=reverse)
            # 最终块越大 计算次数越少
            prm = mini_sort(prm, 'final')
            # 倍数越大 还原品质越高
            prm = mini_sort(prm, 'multiple')
            # [print(i) for i in prm]

            # 优先选择符合倍数的 jpg压缩画质会比卡中间的好很多
            grid_match = [i for i in prm if i['final'] % 8 == 0]
            if grid_match and grid_match[0]['multiple'] >= 1:
                # 这里尝试限制最小的block范围
                # dwt3blk1和dwt1blk4区别，blk4色偏移更小，解码效果更好。
                # 但极端情况下解码效果不如d3b1 (mini512i大片黑底)
                big_block = [i for i in grid_match if i['block'] >= 4]
                if big_block:  # 满足8的倍数时 block不要过小 奇异值会更准
                    result = big_block[0]
                else:
                    result = grid_match[0]
                log.debug(f'Match Grid: {result}')
            else:
                # 优先选择还原能力强的
                for _target in [1.5, 1]:
                    match = [i for i in prm if i['multiple'] >= _target]
                    if match:
                        # 选择块数最少的
                        biggest = mini_sort(match, 'final')
                        result = biggest[0]
                        log.debug(f'BIG: {biggest[0]}')
                        break
            block_list.append(result if result else None)

        if None in block_list:  # 有一边未找到合适参数
            final_block = dwt_deep = block = 0
        else:
            r = mini_sort(block_list, 'block', reverse=False)[0]
            dwt_deep, block = r.get('dwt_deep', 0), r.get('block', 0)
            final_block, multiple = r.get('final', 0), r.get('multiple', 0)
            print(f'dwt_deep({dwt_deep}) x block({block}) = {final_block}'
                  f' ({multiple=:.2f})')

        if block == 0:
            print(f'⬇⬇⬇\nError: Source image too small ({src_shape=})\n⬆⬆⬆')
        elif final_block < 2:
            print(f'⬇⬇⬇\nWarning: Block too small ({final_block=})\n⬆⬆⬆')

        return block, dwt_deep

    def block_add_wm(self, block, order_index, wm_point):
        '''
        获取block数据，嵌入该block对应的水印的像素的数据。
        返回混合后的block数据。
        '''
        block_dct = cv2.dct(block)  # 二维离散余弦变换

        # 伪随机乱序
        if order_index is None:
            block_dct_sf = block_dct  # 不使用随机
        else:
            block_dct_flt = block_dct.flatten()
            block_dct_flt = block_dct_flt[order_index]  # 按index排序
            block_dct_sf = block_dct_flt.reshape(self.block_shape)  # 重组合

        # 加mod
        U, sigma, V = np.linalg.svd(block_dct_sf)  # 奇异值分解
        for _i, mod in enumerate(self.mod):
            sigma[_i] = sigma[_i] - sigma[_i] % mod + 1 / 2 * mod
            sigma[_i] += (1 if wm_point > 127 else -1) * 1 / 4 * mod

        block_dct_sf_mod = np.dot(U, np.dot(np.diag(sigma), V))

        # 逆伪随机乱序
        if order_index is None:
            block_dct_mod = block_dct_sf_mod  # 不使用随机
        else:
            block_dct_sf_mod_flt = block_dct_sf_mod.flatten()
            block_dct_sf_mod_flt[order_index] = block_dct_sf_mod_flt.copy()
            block_dct_mod = block_dct_sf_mod_flt.reshape(self.block_shape)

        block_mod = cv2.idct(block_dct_mod)

        # 绘制转换过程
        # imshow('result', np.hstack((block, block_dct, block_dct_mod, block_mod)))
        return block_mod

    def block_get_wm(self, block, order_index, get_sigma=False):
        '''
        获取block数据，返回block对应的水印像素点的灰度值。
        '''
        block_dct = cv2.dct(block)

        # 伪随机乱序
        if order_index is None:
            block_dct_sf = block_dct  # 不使用随机
        else:
            block_dct_flt = block_dct.flatten()
            block_dct_flt_sf = block_dct_flt[order_index]
            block_dct_sf = block_dct_flt_sf.reshape(self.block_shape)

        U, sigma, V = np.linalg.svd(block_dct_sf)

        if get_sigma:
            return sigma[0]

        wm = []
        for _i, mod in enumerate(self.mod):
            wm.append(255 if sigma[_i] % mod > mod / 2 else 0)

        # wm = wm[0] if len(wm)==1 else (wm[0] * 3 + wm[1] * 1) / 4
        wm = np.mean(wm)
        return wm

    def embed(self, *, src, wm, output, jpg_quality=80):
        '''
        嵌入水印
        需要源图，水印图（最好黑白），输出文件名。其他参数在init时输入。
        '''
        print(f'---\nStart embed watermark\nsrc: {src}\nwm:  {wm}')
        # 载入原图
        src_img = self.load_image(src)
        src_h, src_w = src_img.shape[:2]  # 原图高/宽

        # 载入水印
        wm_data = self.load_watermark(wm)
        wm_h, wm_w = wm_data.shape[:2]

        # 自动计算block的大小和小波次数dwt_deep
        if str(self.block).lower() == 'auto':
            block, self.dwt_deep = self.auto_block(src_shape=(src_h, src_w),
                                                   wm_shape=(wm_h, wm_w))
            if block == 0:  # 直接输出无水印原图
                self.save_image(output, src_img, jpg_quality)
                print('★★★ Output with NO WATERMARK')
                return
            self.block_shape = (block, block)

        # 小波变换
        dwt_img, hvd_list = self.dwt_image(src_img)  # dwt图/hvd列表
        curve_h, curve_w = dwt_img[0].shape[:2]  # 最终dwt后的宽高
        block_w, block_h = self.block_shape  # block宽高
        block_h_num, block_v_num = curve_w // block_w, curve_h // block_h

        self.check_block(dwt_img, wm_data)  # 分块数是否足以记录水印

        # 加随机种子
        order_index = None
        if self.block_seed is not None:
            random_dct = np.random.RandomState(self.block_seed)
            order_index = np.arange(self.block_shape[0] * self.block_shape[1])

        # 处理各个block
        for y in range(0, curve_h - block_h + 1, block_h):
            for x in range(0, curve_w - block_w + 1, block_w):
                # 计算水印坐标
                block_x, block_y = int(x / block_w), int(y / block_h)  # blk idx
                if self.wm_map_method == 1:  # 横向平铺到底换行
                    index = (block_y * block_v_num + block_x) % (wm_w * wm_h)
                    wm_y, wm_x = index % wm_h, index // wm_w
                elif self.wm_map_method == 2:  # 横竖分别对应
                    wm_y, wm_x = block_y % wm_h, block_x % wm_w

                wm_point = wm_data[wm_y, wm_x]  # 水印的取样点
                for chn_idx, this_channel in enumerate(dwt_img):
                    if order_index is not None:
                        random_dct.shuffle(order_index)  # 伪随机index
                    block = this_channel[y:y + block_h, x:x + block_w]
                    embed_data = self.block_add_wm(block, order_index, wm_point)
                    this_channel[y:y + block_h, x:x + block_w] = embed_data

        # 逆小波变换
        out_img = self.idwt_image(src_h, src_w, dwt_img, hvd_list)

        # 保存图片
        self.save_image(output, out_img, jpg_quality)
        return out_img  # 为了测试时保存不同质量的jpg 避免重复计算

    def extract(self, *, src, wm_w=64, wm_h=64, output=None, channel=1):
        '''
        提取水印。
        需要嵌有水印的图，水印尺寸（高,宽），输出的水印图片名。
        channel=1 默认获取Y通道的水印图。None:全部通道, 0:3合1的通道。
        下面的 not channel 是个 cheating 的写法。
        '''
        print(f'---\nStart extract watermark\nsrc: {src}')
        output = output or os.path.splitext(src_img)[0] + '.png'

        # 载入含水印的图片
        src_img = self.load_image(src)
        src_h, src_w = src_img.shape[:2]  # 原图高/宽

        # 自动计算block的大小和小波次数dwt_deep
        if str(self.block).lower() == 'auto':
            block, self.dwt_deep = self.auto_block(src_shape=(src_h, src_w),
                                                   wm_shape=(wm_h, wm_w))
            if block == 0:  # 无水印
                return
            self.block_shape = (block, block)

        # 小波变换
        dwt_img, hvd_list = self.dwt_image(src_img)  # 原图高/宽/dwt图/hvd列表
        curve_h, curve_w = dwt_img[0].shape[:2]  # 最终dwt后的宽高
        block_w, block_h = self.block_shape  # block宽高
        block_h_num, block_v_num = curve_w // block_w, curve_h // block_h

        # 加随机种子
        order_index = None
        if self.block_seed is not None:
            random_dct = np.random.RandomState(self.block_seed)
            order_index = np.arange(self.block_shape[0] * self.block_shape[1])

        # [1+3 channel][row][column]
        wm_data = [[[[] for _ in range(wm_w)] for _ in range(wm_h)]
                   for _ in range(1 + len(dwt_img))]

        # 处理各个block
        for y in range(0, curve_h - block_h + 1, block_h):
            for x in range(0, curve_w - block_w + 1, block_w):
                # 计算水印坐标
                block_x, block_y = int(x / block_w), int(y / block_h)  # blk idx
                if self.wm_map_method == 1:  # 横向平铺到底换行
                    index = (block_y * block_v_num + block_x) % (wm_w * wm_h)
                    wm_y, wm_x = index % wm_h, index // wm_w
                elif self.wm_map_method == 2:  # 横竖分别对应
                    wm_y, wm_x = block_y % wm_h, block_x % wm_w

                for chn_idx, this_channel in enumerate(dwt_img):
                    chn_idx += 1  # 合成后0是混合其他颜色的通道
                    if order_index is not None:
                        random_dct.shuffle(order_index)  # 伪随机index
                    block = this_channel[y: y + block_h, x: x + block_w]
                    if not channel or chn_idx == channel:  # 只读取需要的通道
                        point = self.block_get_wm(block, order_index)
                    else:
                        point = 0
                    wm_data[chn_idx][wm_y][wm_x].append(point)

                # 添加混合3通道的点（如果要输出全通道水印的话）
                if not channel:
                    this_pt = [wm_data[i + 1][wm_y][wm_x][-1] for i in range(3)]
                    wm_data[0][wm_y][wm_x].append(np.mean(this_pt))

        # 各点取平均值
        for chn_idx, chn in enumerate(wm_data):
            if not channel or chn_idx == channel:
                wm_data[chn_idx] = np.array([[np.mean(_p) for _p in row]
                                             for row in chn])

        # 保存水印图
        self.save_watermark(wm_data, wm_w, wm_h, output, channel)


# ═══════════════════════════════════════════════


if __name__ == "__main__":

    from .cv2_tools import imshow
    from .test import format_filename

    log.basicConfig(level=log.DEBUG)
    # log.basicConfig(level=log.INFO)

    # 参数设定
    jpg_quality = 80
    kw = {
        'src_img': 'lena512',  # 原图
        # 'src_img': 'mini512',
        # 'src_img' :'noy512',
        'wm': '64',
        'wm_seed': 1234,  # 水印随机种子 (0~4294967296)
        'block_seed': 5678,  # block随机种子 (0~4294967296)
        'mod': 24,  # 对齐除数
        'fmt': 'png',  # 输出格式
        'fmt': 'jpg',  # 输出格式
        'jpg_quality': jpg_quality,  # 输出jpg质量
    }
    img, wm, out_img, out_wm, kwargs = format_filename(**kw)

    # 加水印
    bwm = watermark(**kwargs)
    bwm.embed(src=img, wm=wm, output=out_img, jpg_quality=jpg_quality)

    # 解水印
    bwm = watermark(**kwargs)
    bwm.extract(src=out_img, output=out_wm)
