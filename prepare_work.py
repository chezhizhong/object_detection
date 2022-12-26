# -*- coding:utf-8 -*-
import cv2
import math
import numpy as np
import csv
import json
import os
import matplotlib.pyplot as plt


"""
该脚本的目的是计算训练数据的均值和方差，以及计算目标基本尺寸的数据分布
"""


def get_image_pixel_mean(img_dir, img_list, img_size):
    """
    求数据集的r,g,b的均值
    :param img_dir: 影像路径
    :param img_list:
    :param img_size: resize到统一大小
    :return: r_mean,g_mean,b_mean
    """
    r_sum = 0
    g_sum = 0
    b_sum = 0
    count = 0

    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (img_size, img_size))
        r_sum = r_sum + img[:, :, 0].mean()
        g_sum = g_sum + img[:, :, 1].mean()
        b_sum = b_sum + img[:, :, 2].mean()
        count += 1

    r_sum = r_sum/count
    g_sum = g_sum/count
    b_sum = b_sum/count

    img_mean = [r_sum, g_sum, b_sum]
    return img_mean


def get_image_pixel_std(img_dir, img_mean, img_list, img_size):
    """
    求所有样本的R、G、B标准差
    :param img_dir:
    :param img_mean:
    :param img_list:
    :param img_size:
    :return: r_std,g_std,b_std
    """
    r_squared_mean = 0
    g_squared_mean = 0
    b_squared_mean = 0
    count = 0
    img_mean = np.array(img_mean)

    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        if not os.path.isdir(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img=cv2.resize(img,(img_size, img_size))
            img = img-img_mean
            # 求单张图片的方差
            r_squared_mean += np.mean(np.square(img[:, :, 0].flatten()))
            g_squared_mean += np.mean(np.square(img[:, :, 1].flatten()))
            b_squared_mean += np.mean(np.square(img[:, :, 2].flatten()))
            count += 1

    r_std = math.sqrt(r_squared_mean / count)
    g_std = math.sqrt(g_squared_mean / count)
    b_std = math.sqrt(b_squared_mean / count)
    img_std = [r_std, g_std, b_std]
    return img_std


def get_mean_std():
    img_dir = "/code/cage_data/train/"
    img_list = os.listdir(img_dir)
    img_size = 640
    img_mean = get_image_pixel_mean(img_dir, img_list, img_size)
    img_std = get_image_pixel_std(img_dir, img_mean, img_list, img_size)
    print("img_mean:{}".format(img_mean))
    print("img_std:{}".format(img_std))


def get_box_size(json_path, save_path):
    # json_path = '/code/cage_data/train.json'
    with open(json_path, 'r') as fcc_file:
        json_data = json.load(fcc_file)
    annotations_data_list = json_data['annotations']
    size_list = []
    for ann in annotations_data_list:
        size_list.append(math.sqrt(ann['bbox'][2] * ann['bbox'][3]))
    min_size = int(min(size_list)) - 1
    max_size = int(max(size_list)) + 1
    bins = range(min_size, max_size, 20)
    print(len(size_list))

    plt.hist(size_list, bins=bins, color="g", histtype="bar", rwidth=1, alpha=0.6)

    plt.xlabel("size(pixel)")
    plt.ylabel("number")

    plt.show()
    plt.savefig(save_path)


if __name__ == '__main__':
    get_box_size('/code/cage_data/train.json', '/code/cage_data/train.png')
