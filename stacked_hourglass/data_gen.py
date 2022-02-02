import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
import utils


class GenerateHeatmap(): # heatmap 만들기
    def __init__(self, output_res, num_parts): # 64, part 개수 16
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res / 64 # sigma 값 : 1
        self.sigma = sigma
        size = 6 * sigma + 3 # size = 9

        x = np.arange(0, size, 1, float) # 0~9 까지 1칸씩 : 즉 0,1,2 ... , 9 // (9,) = [1,2,3,4,5,6,7,8,9]
        y = x[:, np.newaxis] # 9행 1열 (9,1) 2차원
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1 # x0 = 4, y0 =4
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints): # hms.shape = (16, 64, 64) 0으로 채움
        hms = np.zeros(shape=(self.num_parts, self.output_res, self.output_res), dtype=np.float32)
        sigma = self.sigma # sigma = 1
        for p in keypoints:
            for idx, pt in enumerate(p): # keypoints에는 index와 point들이 있다. pt = (x,y)
                if pt[0] > 0: # x좌표가 0보다 크면
                    x, y = int(pt[0]), int(pt[1]) # x좌표 = p[0], y좌표 = p[1]
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res: # x <0, y<0 , x > 64,y>64 이면 continue
                        continue
                    ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1) # ul = ( x-4, y-4 )
                    br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2) # br = ( x+5, y+5 )

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


class mpii_dataset(torch.utils.data.Dataset):
    def __init__(self, ds, index, input_size=256, output_size=64, num_parts=16):
        self.input_res = input_size
        self.output_res = output_size
        self.generateHeatmap = GenerateHeatmap(self.output_res, num_parts)
        self.ds = ds
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.loadImage(self.index[idx % len(self.index)])

    def loadImage(self, idx):
        ds = self.ds

        ## load + crop
        orig_img = ds.get_img(idx)
        orig_keypoints = ds.get_kps(idx)
        kptmp = orig_keypoints.copy()
        c = ds.get_center(idx)
        s = ds.get_scale(idx)

        cropped = utils.crop(orig_img, c, s, (self.input_res, self.input_res))
        for i in range(np.shape(orig_keypoints)[1]):
            if orig_keypoints[0, i, 0] > 0:
                orig_keypoints[0, i, :2] = utils.transform(orig_keypoints[0, i, :2], c, s, (self.input_res, self.input_res))
        keypoints = np.copy(orig_keypoints)

        ## augmentation -- to be done to cropped image
        height, width = cropped.shape[0:2]
        center = np.array((width / 2, height / 2))
        scale = max(height, width) / 200
        aug_rot = (np.random.random() * 2 - 1) * 30.
        aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
        scale *= aug_scale
        mat_mask = utils.get_transform(center, scale, (self.output_res, self.output_res), aug_rot)[:2]
        mat = utils.get_transform(center, scale, (self.input_res, self.input_res), aug_rot)[:2]
        inp = cv2.warpAffine(cropped, mat, (self.input_res, self.input_res)).astype(np.float32) / 255
        keypoints[:, :, 0:2] = utils.kpt_affine(keypoints[:, :, 0:2], mat_mask)
        if np.random.randint(2) == 0:
            inp = self.preprocess(inp)
            inp = inp[:, ::-1]
            keypoints = keypoints[:, ds.flipped_parts['mpii']]
            keypoints[:, :, 0] = self.output_res - keypoints[:, :, 0]
            orig_keypoints = orig_keypoints[:, ds.flipped_parts['mpii']]
            orig_keypoints[:, :, 0] = self.input_res - orig_keypoints[:, :, 0]

        ## set keypoints to 0 when were not visible initially (so heatmap all 0s)
        for i in range(np.shape(orig_keypoints)[1]):
            if kptmp[0, i, 0] == 0 and kptmp[0, i, 1] == 0:
                keypoints[0, i, 0] = 0
                keypoints[0, i, 1] = 0
                orig_keypoints[0, i, 0] = 0
                orig_keypoints[0, i, 1] = 0

        ## generate heatmaps on outres
        heatmaps = self.generateHeatmap(keypoints)
        # inp.transpose
        return torch.tensor(inp.astype(np.float32)).permute(2, 0, 1), torch.tensor(heatmaps.astype(np.float32))

    def preprocess(self, data):
        # random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV);
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:, :, 0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        data[:, :, 1] *= delta_sature
        data[:, :, 1] = np.maximum(np.minimum(data[:, :, 1], 1), 0)
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        return data


def get_mpii_dataset(mode='train'):
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    import mpii_data as ds

    ds.init()
    train, valid = ds.setup_val_split()
    dataset = {key: mpii_dataset(ds, data) for key, data in zip(['train', 'valid'], [train, valid])}
    return dataset[mode]


if __name__ == '__main__':

    train_set = get_mpii_dataset('train')
    img, heatmap = train_set[0]
    print(img.shape)
    print(heatmap.shape)
    cv2.imshow('',img.permute(1,2,0).numpy())
    cv2.waitKey(0)

    # from PIL import Image
    #
    # img = img.permute(1, 2, 0) * 255
    # print(img.shape)
    # img = Image.fromarray(np.uint8(img))
    # img.show()
    #
    # heatmap = heatmap * 255
    # for i in range(len(heatmap)):
    #     hm = heatmap[i, :, :]
    #     hm = Image.fromarray(np.uint8(hm))
    #     hm.show()

        # from PIL import Image
        # import numpy
        # # img=cv2.imread('data/mpii_dataset/images/015601864.jpg')
        # # print(img.size)
        # # for line in img[140]:
        # #     print(line)
        # # cv2.imshow('',img)
        # # cv2.waitKey(10000)
        #
        # img1=Image.open('data/mpii_dataset/images/015601864.jpg')
        # data=numpy.asarray(img1)
        # print(data.dtype)
        # img1=Image.fromarray(data)
        # img1.show()
