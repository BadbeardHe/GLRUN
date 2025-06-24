##################################
# Predict single scene with GLRUN
# Dataset: FLAT
# Input: noisy iq
# Output: denoised iq and depth
# Updated by Changyong He, Jin Zeng, 20240623
##################################
import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from glrun.GLRUN import GLRUN
from predict import predict_single_scene


def predict_single(args):
    device = torch.device("cuda")
    scene = args.scene_path
    model_path = args.model
    out_path = args.predict_iq_path
    out_d_path = args.predict_depth_path

    if not os.path.exists(scene):
        print(f"Scene file {scene} does not exist!")
        raise FileNotFoundError
    if not os.path.exists(model_path):
        print(f"Model '{model_path}' does not exist!")
        raise FileNotFoundError

    model = GLRUN()
    checkpoint = torch.load(model_path, map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model = model.eval()

    iq, _, depth, _ = predict_single_scene(scene, device, model)
    iq.tofile(out_path)
    depth.tofile(out_d_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dev', type=int, default=0, help='device id')

    parser.add_argument("-in", "--scene_path", type=str, default=None, help="Noisy raw path")
    parser.add_argument("-out_iq", "--predict_iq_path", type=str, default=None, help="Denoised iq path.")
    parser.add_argument("-out_d", "--predict_depth_path", type=str, default=None, help="Denoised depth path.")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to the pretrained model.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    predict_single(args)
