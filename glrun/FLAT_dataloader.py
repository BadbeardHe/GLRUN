'''
Author: Jin Zeng
Date: 2023-04-03 
LastEditTime: 2023-08-07
Description: GLRUN for iToF denoising
'''
import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def sqrt_ldr(correlations):
    tof_conf = np.abs(correlations[0,:,:]) + np.abs(correlations[1,:,:]) 
    tof_conf_l = 16*np.sqrt(tof_conf+36)-96
    tof_conf[tof_conf==0]=1
    i_tmp = tof_conf_l*correlations[0,:,:]/tof_conf
    q_tmp = tof_conf_l*correlations[1,:,:]/tof_conf
    
    return np.stack((i_tmp, q_tmp), axis=0)


def load_ideal(scene):
    """
    :param scene: path of ideal file
    :return: ideal depth with LIB2, in unit meter
    """
    size = [424, 512]
    nimg = np.fromfile(scene, dtype = np.float32).reshape(size[0], size[1]) / 1e3
    data_expanded = np.expand_dims(nimg.astype(np.float32), 0)
    return data_expanded


def load_raw(scene,sqrt_in):
    shape = [424, 512, 9]
    correlations = np.fromfile(scene, dtype = np.float32).reshape(shape)

    tof_IQ_40 = np.stack((correlations[:,:,1], correlations[:,:,0]), axis=0) #IQ
    tof_IQ_30 = np.stack((correlations[:,:,4], correlations[:,:,3]), axis=0) #IQ
    tof_IQ_58 = np.stack((correlations[:,:,7], correlations[:,:,6]), axis=0) #IQ
    if sqrt_in:
        tof_IQ_40 = sqrt_ldr(tof_IQ_40)
        tof_IQ_30 = sqrt_ldr(tof_IQ_30)
        tof_IQ_58 = sqrt_ldr(tof_IQ_58)

    tof_IQs = np.concatenate((tof_IQ_30, tof_IQ_40, tof_IQ_58), axis=0)    # [6, 424, 512]
    tof_IQs = tof_IQs/500

    return tof_IQs.astype(np.float32)


class FLAT_Dataset(Dataset):
    """
    Dataset loader
    """

    def __init__(self, img_dir, mode='train', sqrt_ldr=True, transform=None):
        """
        Args:
            img_dir (string): Path to the image files with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.sqrt_ldr = sqrt_ldr
        self.mode = mode
        if mode != 'train' and mode != 'test':
            raise NotImplementedError  

        self.list = []
        self.raw_paths = []
        self.ideal_paths = []
        self.ideal_d_paths = []

        self.transform = transform    

        self.noise_path = os.path.join(img_dir, "noise")
        self.ideal_path = os.path.join(img_dir, "ideal")
        self.ideal_d_path = os.path.join(img_dir, "ideal_depth") 

        if mode == 'train':
            file_list = os.path.join(img_dir, "list/train.txt")
        elif mode == 'test':
            file_list = os.path.join(img_dir, "list/test.txt")
        
        with open(file_list, 'r') as f:
            for line in f:
                self.list.append(line.strip('\n'))
        
        for file in self.list:
            self.raw_paths.append(f"{self.noise_path}/{file}")
            self.ideal_paths.append(f"{self.ideal_path}/{file}")
            self.ideal_d_paths.append(f"{self.ideal_d_path}/{file}")

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):

        tof_raw_IQ = load_raw(self.raw_paths[idx], self.sqrt_ldr) 
        tof_ideal_IQ = load_raw(self.ideal_paths[idx], self.sqrt_ldr)
        ideal_d = load_ideal(self.ideal_d_paths[idx])        

        tof_raw_IQ = np.nan_to_num(tof_raw_IQ, nan=0.0, posinf=0.0, neginf=0.0)
        tof_ideal_IQ = np.nan_to_num(tof_ideal_IQ, nan=0.0, posinf=0.0, neginf=0.0)

        tof_raw_IQ_tensor = torch.from_numpy(tof_raw_IQ).float() # [6, 424, 512]
        tof_ideal_IQ_tensor = torch.from_numpy(tof_ideal_IQ).float() # [6, 424, 512]
        ideal_d_tensor = torch.from_numpy(ideal_d).float() # [1, 424, 512]
        
        # if self.transform:
        #     sample = self.transform(sample)

        return tof_raw_IQ_tensor, tof_ideal_IQ_tensor, ideal_d_tensor


if __name__ == "__main__":
    batch_size = 10
    img_dir='../../FLAT'
    train_data = FLAT_Dataset(img_dir=img_dir, mode='test')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    count = 0

    print(train_dataloader.__len__())
    for i, mini_batch in enumerate(train_dataloader):
        raw_IQ, ideal_IQ, ideal_d = mini_batch

        np_raw = raw_IQ.cpu()
        np_raw = np_raw[0].detach().numpy()
        np_raw.tofile("./rawIQ/"+str(count))

        np_ideal = ideal_IQ.cpu()
        np_ideal = np_ideal[0].detach().numpy()
        np_ideal.tofile("./idealIQ/"+str(count))

        np_ideal = ideal_d.cpu()
        np_ideal = np_ideal[0][0].detach().numpy()
        np_ideal.tofile("./ideald/"+str(count))

        print(f"Shape of nimg [N, C, H, W]: {raw_IQ.shape}")
        print(f"Shape of rimg [N, C, H, W]: {ideal_IQ.shape}")
        print(f"Shape of rimg [N, C, H, W]: {ideal_d.shape}")

        print(raw_IQ.abs().mean())
        print(ideal_IQ.abs().mean())

        print(raw_IQ.max())
        print(ideal_IQ.max())

        print(raw_IQ.min())
        print(ideal_IQ.min())



        break
