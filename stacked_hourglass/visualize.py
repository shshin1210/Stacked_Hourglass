import math
import random

import cv2
import numpy as np
import torch

import mpii_data as ds

from PIL import Image


ds.init()
checkpoint_path = 'BEST_checkpoint.tar'

idx = 0


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def get_kpts(maps, img_h=368.0, img_w=368.0):
    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]

    kpts = []
    for m in map_6:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x, y])
    return kpts


def draw_paint(img_path, kpts):
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [255, 0, 255], [125, 0, 255]]
    # limbSeq = [[13, 12], [12, 9], [12, 8], [9, 10], [8, 7], [10, 11], [7, 6], [12, 3], [12, 2], [2, 1], [1, 0], [3, 4],
    # [4, 5]]
    limbSeq = [[0, 1], [1, 2],
               [3, 4], [4, 5],
               [2, 6], [3, 6],
               [6, 7], [7, 8], [8, 9],
               [10, 11], [11, 12],
               [13, 14], [14, 15],
               [12, 7], [13, 7]]
    # img = transforms.ToPILImage()(img[0].cpu())
    # im = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    im = cv2.imread(img_path)
    # im = cv2.resize(im, (256, 256))
    # draw points
    for k in kpts:
        print(k)
        x = k[0]
        y = k[1]
        cv2.circle(im, (int(x), int(y)), radius=3, thickness=2, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]
        mX = np.mean([X0, X1])
        mY = np.mean([Y0, Y1])
        length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_im, polygon, colors[i])
        im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)
    # cv2.imshow('', im)
    # cv2.waitKey(0)
    global idx
    cv2.imwrite('visualize/{0}.jpg'.format(idx), im)
    idx += 1


def test_example(model, img_path, kpts0):
    # print(kpts0)
    # img = cv2.imread(img_path)
    # h, w = img.shape[0:2]
    # img = np.array(cv2.resize(img, (256, 256)), dtype=np.float32)
    # img = torch.from_numpy(img.transpose((2, 0, 1)))
    # # cv2.circle(img, (int(center_x), int(center_y)), radius=3, thickness=3, color=(0, 0, 255))
    # # cv2.imshow('', img)
    # # cv2.waitKey(0)
    # # normalize
    # mean = [128.0, 128.0, 128.0]
    # std = [256.0, 256.0, 256.0]
    # for t, m, s in zip(img, mean, std):
    #     # t.sub_(m).div_(s)
    #     t.div_(s)
    img = torch.unsqueeze(img, 0)

    heat = model(img)
    kpts = get_kpts(heat[0], img_h=h, img_w=w)
    print(kpts)
    kpts = get_kpts(heat[2], img_h=h, img_w=w)
    print(kpts)
    kpts = get_kpts(heat[-1], img_h=h, img_w=w)
    print(kpts)
    draw_paint(img_path, kpts)
    draw_paint(img_path, kpts0)


def visualize(model=None):
    if not model:
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint['model']

    train, valid = ds.setup_val_split()
    sample = random.sample(list(train), 32)
    for idx in sample:
        img_path = ds.get_path(idx)
        kpts = ds.get_kps(idx)
        print(img_path)
        print(kpts.shape)
        test_example(model, img_path, kpts[0][:, 0:2])


if __name__ == '__main__':
    visualize()
