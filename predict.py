##################################
# Train glrun with FLAT dataset
# Dataset: FLAT
# Input: kinect iq
# Output: denoised kinect iq
# Updated by Jin Zeng, 20230808
##################################
import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from glrun.GLRUN import GLRUN
from glrun.FLAT_dataloader import load_raw
from utils import iq2depth

import warnings
warnings.filterwarnings("ignore")


def sqrt_hdr(correlations):
    correlations = correlations[0].detach().numpy()
    correlations = 500 * correlations

    tof_conf = np.abs(correlations[0, :, :]) + np.abs(correlations[1, :, :])
    tof_conf_h = (tof_conf / 16 + 6) ** 2 - 36

    tof_conf[tof_conf == 0] = 1
    iq_0 = tof_conf_h * correlations[0, :, :] / tof_conf
    iq_1 = tof_conf_h * correlations[1, :, :] / tof_conf

    return np.stack((iq_0, iq_1), axis=0)


def get_input(scene):
    tof_raw_IQ = load_raw(scene, sqrt_in=True)
    tof_raw_IQ = np.nan_to_num(tof_raw_IQ, nan=0.0, posinf=0.0, neginf=0.0)
    tof_raw_IQ_tensor = torch.from_numpy(tof_raw_IQ).float()  # [6, 424, 512]

    return tof_raw_IQ_tensor


def predict_single_scene(scene, device, model):
    with torch.no_grad():
        raw_IQ = get_input(scene)
        raw_IQ = raw_IQ.to(device)
        raw_IQ = raw_IQ.unsqueeze(0)

        t0 = time.time()
        out_0, mu0 = model(raw_IQ[:, 0:2, :, :])
        out_1, mu1 = model(raw_IQ[:, 2:4, :, :])
        out_2, mu2 = model(raw_IQ[:, 4:6, :, :])
        t1 = time.time()

    # sqrt in
    out_0 = sqrt_hdr(out_0.cpu())
    out_1 = sqrt_hdr(out_1.cpu())
    out_2 = sqrt_hdr(out_2.cpu())

    outputs = np.concatenate((out_0, out_1, out_2), axis=0)
    outputs_mu = np.concatenate((mu0.cpu().detach().numpy(), mu1.cpu().detach().numpy(), mu2.cpu().detach().numpy()),
                                axis=0)
    
    raw = np.fromfile(scene, dtype=np.float32).reshape((424, 512, 9))
    outputs_d = iq2depth(outputs, raw)

    return outputs, outputs_mu, outputs_d, t1-t0


def predict(args):
    cudaid = "cuda:" + str(args.dev)
    device = torch.device(cudaid)

    # args
    raw_dir = args.train_path
    out_dir = args.destination
    out_mu_dir = args.destination_mu
    out_d_dir = args.destination_depth
    list_path = args.list_path
    model_path = args.model

    if not os.path.exists(raw_dir):
        print(f"Dataset path '{raw_dir}' does not exist!")
        raise FileNotFoundError
    if not os.path.exists(model_path):
        print(f"Model '{model_path}' does not exist!")
        raise FileNotFoundError
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_mu_dir, exist_ok=True)
    os.makedirs(out_d_dir, exist_ok=True)
    print(device, model_path)

    # load file list
    predict_list = []
    with open(list_path, 'r') as f:
        for line in f:
            path = line.strip('\n')
            predict_list.append(path)

    # model
    glrun = GLRUN()
    checkpoint = torch.load(model_path)
    glrun.load_state_dict(checkpoint['model_state_dict'])
    glrun.to(device)
    glrun = glrun.eval()

    # predict
    print("[ Start Predicting ]")
    t_total = 0
    num_sample = len(predict_list)
    for file in tqdm(predict_list, desc="Predicting"):
        out_iq, out_mu, out_d, t_last = predict_single_scene(f"{raw_dir}/{file}", device, glrun)
        t_total += t_last
        out_iq.tofile(f"{out_dir}/{file}")
        out_mu.tofile(f"{out_mu_dir}/{file}")
        out_d.tofile(f"{out_d_dir}/{file}")
    
    t_avg = t_total / num_sample
    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))
    print("[ End Predicting ]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dev', type=int, default=0, help='device id')

    parser.add_argument("-in", "--train_path", type=str, default='../FLAT/noise_IQ', help="Train set directory")
    parser.add_argument("-ls", "--list_path", type=str, default='../FLAT/test.txt', help='Path to the test list file')
    parser.add_argument("-out", "--destination", type=str, default='./predict_result_iq', help="Output IQ destination.")
    parser.add_argument("-out_d", "--destination_depth", type=str, default='./predict_result_d', help="Output destination.")
    parser.add_argument("-out_mu", "--destination_mu", type=str, default='./predict_result_mu', help="Output destination.")
    parser.add_argument("-m", "--model", type=str, default='./result_v2/checkpoint_best.pth', help="Path to the trained DeepGLR.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    predict(args)
