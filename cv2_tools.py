#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Erimus'

import cv2
import numpy as np

# ═══════════════════════════════════════════════


def imshow(window_name, img_data, wait=True):
    '''
    打印并显示图片数据
    '''
    img_data = img_data.copy()
    print(f'\n{window_name}\n{img_data}')
    print(f'{window_name} {img_data.max() = }')
    print(f'{window_name} {img_data.min() = }')
    img_data[img_data < 0] = 0
    img_data[img_data > 255] = 255
    cv2.imshow(window_name, img_data.astype(np.uint8))
    if wait:
        cv2.waitKey()


class cv_text():
    '''
    简化cv2上的文字打印，增加对齐支持。
    '''
    def __init__(self, font_family=None, font_size=14):
        self.font = font_family or 0
        (w, h), baseline = cv2.getTextSize('Bag', self.font, 1, 1)
        print(f'{w=} {h=} {baseline=}')
        self.base_size = h

    def put(self, canvas, text, x, y, size=14, align=('left', 'bottom')):
        scale = size / self.base_size
        (w, h), bl = cv2.getTextSize(text, self.font, fontScale=scale, thickness=1)
        h_dict = {'left': 0, 'right': w, 'mid': w / 2}
        x -= int(h_dict[align[0]])
        v_dict = {'bottom': 0, 'top': h, 'mid': h / 2}
        y += int(v_dict[align[1]])
        canvas = cv2.putText(canvas, text, (x, y), self.font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas


def text_test():
    # 试绘文字相关的 字体 宽高 基线
    font = 0
    black, blue, red = (0, 0, 0), (255, 0, 0), (0, 0, 255)
    pd = 50
    cvs_list = []
    W = H = 0  # total
    for font in range(8):
        (w, h), bl = cv2.getTextSize(f'Font Family: {font}', font, 1, 1)
        ch, cw = h + pd * 2, w + pd * 2
        W = max(W, cw)
        H += ch
        cvs = np.zeros((ch, cw, 3), np.uint8)
        cvs.fill(255)
        cvs = cv2.putText(cvs, f'Font Family: {font}', (pd, pd + h),
                          font, 1, black, 1, cv2.LINE_AA)
        cvs = cv2.rectangle(cvs, (1, 1), (cw - 2, ch - 2), black)
        cvs = cv2.rectangle(cvs, (pd, pd), (w + pd, h + pd), blue)
        cvs = cv2.line(cvs, (pd, pd + h + bl), (pd + w, pd + h + bl), red)
        cvs_list.append(cvs)
        # cv2.imshow("cvs", cvs)
        # cv2.waitKey(0)

    print(f'{W = } | {H = }')
    CVS = np.zeros((H, W, 3), np.uint8)
    CVS.fill(255)
    this_y = 0
    for cvs in cvs_list:
        h, w = cvs.shape[:2]
        CVS[this_y:this_y + h, (W - w) // 2:(W - w) // 2 + w] = cvs
        this_y += h
    cv2.imshow("cvs", CVS)
    cv2.waitKey(0)


# ═══════════════════════════════════════════════

if __name__ == '__main__':

    text_test()
