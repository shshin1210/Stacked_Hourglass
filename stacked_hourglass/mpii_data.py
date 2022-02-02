import numpy as np
import h5py
import cv2
import os
import time

annot_dir = 'data/mpii_dataset/joints'
img_dir = 'data/mpii_dataset/images'

assert os.path.exists(img_dir)
mpii, num_examples_train, num_examples_val = None, None, None


class MPII:
    def __init__(self):
        print('loading data...')
        tic = time.time()

        # train_f 변수에 train.h5 read하기
        train_f = h5py.File(os.path.join(annot_dir, 'train.h5'), 'r')
        val_f = h5py.File(os.path.join(annot_dir, 'valid.h5'), 'r')

        # train_f 의 scale, part, visible, normalize, imgname 변수에 저장
        self.t_center = train_f['center'][()]
        t_scale = train_f['scale'][()]
        t_part = train_f['part'][()]
        t_visible = train_f['visible'][()]
        t_normalize = train_f['normalize'][()]
        t_imgname = [None] * len(self.t_center)

        for i in range(len(self.t_center)):     # t_imgname에 imgname 넣어주기
            t_imgname[i] = train_f['imgname'][i].decode('UTF-8')

        # val_f 의 scale, part, visible, normalize, imgname 변수에 저장
        self.v_center = val_f['center'][()]
        v_scale = val_f['scale'][()]
        v_part = val_f['part'][()]
        v_visible = val_f['visible'][()]
        v_normalize = val_f['normalize'][()]
        v_imgname = [None] * len(self.v_center)
        for i in range(len(self.v_center)): # v_imgname에에 imgnam 넣어주기
            v_imgname[i] = val_f['imgname'][i].decode('UTF-8')

        # train과 val set의 cale, part, visible, normalize, imgname 변수에 저장
        self.center = np.append(self.t_center, self.v_center, axis=0)
        self.scale = np.append(t_scale, v_scale)
        self.part = np.append(t_part, v_part, axis=0)
        self.visible = np.append(t_visible, v_visible, axis=0)
        self.normalize = np.append(t_normalize, v_normalize)
        self.imgname = t_imgname + v_imgname

        print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def getAnnots(self, idx):
        '''
        returns h5 file for train or val set
        '''
        return self.imgname[idx], self.part[idx], self.visible[idx], self.center[idx], self.scale[idx], self.normalize[idx] # train과 val의 h5 file 정보 return

    def getLength(self): # val과 train의 center값 길이 반환하기 / *2 하면 총 가로 세로 길이가 된다
        return len(self.t_center), len(self.v_center)


def init():
    global mpii, num_examples_train, num_examples_val
    mpii = MPII()
    num_examples_train, num_examples_val = mpii.getLength() # t_center 와 v_center 값 넣기


# Part reference
parts = {'mpii': ['rank', 'rkne', 'rhip',
                  'lhip', 'lkne', 'lank',
                  'pelv', 'thrx', 'neck', 'head',
                  'rwri', 'relb', 'rsho',
                  'lsho', 'lelb', 'lwri']} # 총 16개

flipped_parts = {'mpii': [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

part_pairs = {'mpii': [[0, 5], [1, 4], [2, 3], [6], [7], [8], [9], [10, 15], [11, 14], [12, 13]]} # 짝이 있는 parts 들의 index list

pair_names = {'mpii': ['ankle', 'knee', 'hip', 'pelvis', 'thorax', 'neck', 'head', 'wrist', 'elbow', 'shoulder']} # pair가 있는 것들의 name


def setup_val_split():
    '''
    returns index for train and validation imgs
    index for validation images starts after that of train images
    so that loadImage can tell them apart
    '''
    valid = [i + num_examples_train for i in range(num_examples_val)] # 예를들어 train이 10000장 있으면 10000+1장부터 10000+i 등까지 val 의 idx값
    train = [i for i in range(num_examples_train)] # t_center 의 길이?
    return np.array(train), np.array(valid)


def get_img(idx): # 이미지 가져오기
    imgname, __, __, __, __, __ = mpii.getAnnots(idx)
    path = os.path.join(img_dir, imgname)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_path(idx): # path 가져오기
    imgname, __, __, __, __, __ = mpii.getAnnots(idx)
    path = os.path.join(img_dir, imgname)
    return path


def get_kps(idx): #
    __, part, visible, __, __, __ = mpii.getAnnots(idx)
    kp2 = np.insert(part, 2, visible, axis=1) #part의 2번째 index에 visible 넣기 (세로축 2번째 index)
    kps = np.zeros((1, 16, 3)) # 16행 3열
    kps[0] = kp2
    return kps


def get_normalized(idx): # normalize 값
    __, __, __, __, __, n = mpii.getAnnots(idx)
    return n


def get_center(idx): # center 값
    __, __, __, c, __, __ = mpii.getAnnots(idx)
    return c


def get_scale(idx): # scale 값
    __, __, __, __, s, __ = mpii.getAnnots(idx)
    return s


if __name__ == '__main__':
    train_f = h5py.File(os.path.join(annot_dir, 'train.h5'), 'r')
    val_f = h5py.File(os.path.join(annot_dir, 'valid.h5'), 'r')
    print(train_f['center'])
    print(val_f['center'])
