

import glob
from collections import defaultdict
import os
import numpy as np
import random
from tqdm import tqdm
from collections import Counter

import torchvision.transforms as transforms

from PIL import Image

dir_pth = os.path.dirname(os.path.abspath(__file__))

UCM_ML_npy = os.path.join(dir_pth, 'ucm_ml.npy')
AID_ML_npy = os.path.join(dir_pth, 'aid_ml.npy')
DFC_ML_npy = os.path.join(dir_pth, 'dfc15_ml.npy')

def default_loader(path):
    return Image.open(path).convert('RGB')


class DataGeneratorML:

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        
        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase

        self.CreateIdx2fileDict()

    def CreateIdx2fileDict(self):
        random.seed(42)

        if self.dataset == 'AID_multilabel':
            data = np.load(AID_ML_npy, allow_pickle=True).item()
        elif self.dataset == 'UCMerced':
            data = np.load(UCM_ML_npy, allow_pickle=True).item()
        elif self.dataset == 'DFC15_multilabel':
            data = np.load(DFC_ML_npy, allow_pickle=True).item()

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        if self.dataset != 'DFC15_multilabel':
            for _, scenePth in enumerate(self.sceneList):

                subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
                random.shuffle(subdirImgPth)

                train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
                val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
                test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

                self.train_numImgs += len(train_subdirImgPth)
                self.test_numImgs += len(test_subdirImgPth)
                self.val_numImgs += len(val_subdirImgPth)

                for imgPth in train_subdirImgPth:
                    
                    multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                    self.train_idx2fileDict[train_count] = (imgPth, multi_hot)
                    train_count += 1
                
                for imgPth in test_subdirImgPth:

                    multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                    self.test_idx2fileDict[test_count] = (imgPth, multi_hot)
                    test_count += 1

                for imgPth in val_subdirImgPth:

                    multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                    self.val_idx2fileDict[val_count] = (imgPth, multi_hot)
                    val_count += 1
        else:

            imgPths = sorted(glob.glob(os.path.join(self.datadir, '*.'+self.imgExt)))
            random.shuffle(imgPths)

            train_subdirImgPth = imgPths[:int(0.7*len(imgPths))]
            val_subdirImgPth = imgPths[int(0.7*len(imgPths)):int(0.8*len(imgPths))]
            test_subdirImgPth = imgPths[int(0.8*len(imgPths)):]

            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                    
                multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                self.train_idx2fileDict[train_count] = (imgPth, multi_hot)
                train_count += 1
            
            for imgPth in test_subdirImgPth:

                multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                self.test_idx2fileDict[test_count] = (imgPth, multi_hot)
                test_count += 1

            for imgPth in val_subdirImgPth:

                multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                self.val_idx2fileDict[val_count] = (imgPth, multi_hot)
                val_count += 1

        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))

    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        elif self.phase == 'val':
            idx = self.valDataIndex[index]
        else:
            idx = self.testDataIndex[index]
        
        return self.__data_generation(idx)
    
    def __data_generation(self, idx):

        if self.phase == 'train':
            imgPth, multi_hot = self.train_idx2fileDict[idx]
        elif self.phase == 'val':
            imgPth, multi_hot = self.val_idx2fileDict[idx]
        else:
            imgPth, multi_hot = self.test_idx2fileDict[idx]
        
        img = default_loader(imgPth)
        
        bin2int = int(''.join(list(map(str, multi_hot.tolist()))), 2)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        return {'img': img, 'idx':idx, 'multiHot':multi_hot.astype(np.float32), 'bin2int':bin2int}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)

class DataGeneratorContrasSin:
    """ 
    data generator for contrastive loss, if the pair shares more than one label in common, 
    it is indicated as 1, otherwise, 0
    """
    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        
        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.train_idx2multiLbIdx = None

        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase

        self.CreateIdx2fileDict()
        self.CreateLbIdx()

    def CreateIdx2fileDict(self):
        random.seed(42)

        if self.dataset == 'AID_multilabel':
            data = np.load(AID_ML_npy, allow_pickle=True).item()
        elif self.dataset == 'UCMerced':
            data = np.load(UCM_ML_npy, allow_pickle=True).item()
        elif self.dataset == 'DFC15_multilabel':
            data = np.load(DFC_ML_npy, allow_pickle=True).item()

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        if self.dataset != 'DFC15_multilabel':
            for _, scenePth in enumerate(self.sceneList):

                subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
                random.shuffle(subdirImgPth)

                train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
                val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
                test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

                self.train_numImgs += len(train_subdirImgPth)
                self.test_numImgs += len(test_subdirImgPth)
                self.val_numImgs += len(val_subdirImgPth)

                for imgPth in train_subdirImgPth:
                    
                    multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                    self.train_idx2fileDict[train_count] = (imgPth, multi_hot)
                    train_count += 1
                
                for imgPth in test_subdirImgPth:

                    multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                    self.test_idx2fileDict[test_count] = (imgPth, multi_hot)
                    test_count += 1

                for imgPth in val_subdirImgPth:

                    multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                    self.val_idx2fileDict[val_count] = (imgPth, multi_hot)
                    val_count += 1
        else:

            imgPths = sorted(glob.glob(os.path.join(self.datadir, '*.'+self.imgExt)))
            random.shuffle(imgPths)

            train_subdirImgPth = imgPths[:int(0.7*len(imgPths))]
            val_subdirImgPth = imgPths[int(0.7*len(imgPths)):int(0.8*len(imgPths))]
            test_subdirImgPth = imgPths[int(0.8*len(imgPths)):]

            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                    
                multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                self.train_idx2fileDict[train_count] = (imgPth, multi_hot)
                train_count += 1
            
            for imgPth in test_subdirImgPth:

                multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                self.test_idx2fileDict[test_count] = (imgPth, multi_hot)
                test_count += 1

            for imgPth in val_subdirImgPth:

                multi_hot = data['nm2label'][os.path.basename(imgPth).split('.')[0]]

                self.val_idx2fileDict[val_count] = (imgPth, multi_hot)
                val_count += 1

        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))

    def CreateLbIdx(self):
        
        lb_multihots = []

        for i in tqdm(self.trainDataIndex):
            _, a_multihot = self.train_idx2fileDict[i]
            lb_multihots.append(list(a_multihot))
        
        lb_multihots = np.asarray(lb_multihots)
        # print(lb_multihots)

        self.train_idx2multiLbIdx = np.matmul(lb_multihots, lb_multihots.transpose())
        self.train_idx2multiLbIdx[self.train_idx2multiLbIdx>=1] = 1

    def __getitem__(self, index):

        if self.phase == 'train':
            target = np.random.randint(0, 2)
            idx = self.trainDataIndex[index]
            indicators = self.train_idx2multiLbIdx[idx,:]
            sm_idxs = np.where(indicators==1)[0]
            dif_idxs = np.where(indicators==0)[0]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(sm_idxs)
            else:
                siamese_index = np.random.choice(dif_idxs)
            return self.__data_generation_p(idx, siamese_index), target
        elif self.phase == 'val':
            idx = self.valDataIndex[index]
            return self.__data_generation(idx)
        else:
            idx = self.testDataIndex[index]
            return self.__data_generation(idx)

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)

    def __data_generation(self, idx):

        # if self.phase == 'train':
        #     imgPth, multi_hot = self.train_idx2fileDict[idx]
        if self.phase == 'val':
            imgPth, multi_hot = self.val_idx2fileDict[idx]
        else:
            imgPth, multi_hot = self.test_idx2fileDict[idx]
        
        img = default_loader(imgPth)
        
        bin2int = int(''.join(list(map(str, multi_hot.tolist()))), 2)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        return {'img': img, 'idx':idx, 'multiHot':multi_hot.astype(np.float32), 'bin2int':bin2int}
    
    def __data_generation_p(self, idx, siamese_index):
        img1Pth, _ = self.train_idx2fileDict[idx]
        img2Pth, _ = self.train_idx2fileDict[siamese_index]

        img1 = default_loader(img1Pth)
        img2 = default_loader(img2Pth)

        if self.imgTransform is not None:
            img1 = self.imgTransform(img1)
            img2 = self.imgTransform(img2)

        return (img1, img2)

if __name__ == "__main__":

    dataGen = DataGeneratorML(
        data='/home/jkang/Documents/data/scene',
        # dataset = 'UCMerced',
        # dataset = 'AID_multilabel',
        dataset = 'DFC15_multilabel',
        # imgExt='jpg',
        imgExt='png',
        phase='train'
    )

    # # bin2int_list = []
    # multi_hots = []
    # for i in tqdm(range(len(dataGen))):
    #     # bin2int_list.append(dataGen[i]['bin2int'])
    #     multi_hots.append(dataGen[i]['multiHot'])

    # # multiHot_counts = Counter(bin2int_list)

    # # print(multiHot_counts)
    # multi_hots = np.asarray(multi_hots)

    # print(multi_hots.shape)

    # multi_hots[multi_hots==0] = -1

    # out = np.matmul(multi_hots, multi_hots.transpose())

    # hamming_dist = (out + multi_hots.shape[1]) / 2
    # weights = hamming_dist / multi_hots.shape[1]

    # print(weights.max(), weights.min())










