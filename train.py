##################################
# Train GLRUN with FLAT dataset
# Dataset: FLAT
# Input: kinect iq
# Output: depth with ideal_depth produced by LIB2 as GT
# Updated by Jin Zeng, 20240623
##################################
import os
import time
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from glrun.GLRUN import GLRUN
from glrun.FLAT_dataloader import FLAT_Dataset
from loss import GLoss, GLoss_test

import warnings
warnings.filterwarnings("ignore")


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger


def train(args):
    cudaid = "cuda:" + str(args.dev)
    device = torch.device(cudaid)

    # args
    batch_size = args.batch_size
    lr = args.learning_rate
    total_epoch = args.epoch
    out_path = args.destination
    debug_path = args.debug

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)

    out_model = os.path.join(out_path, args.name)
    print(device, out_model)

    # dataset
    train_data = FLAT_Dataset(img_dir=args.train_path, mode='train')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data = FLAT_Dataset(img_dir=args.train_path, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    # model
    glrun = GLRUN()
    glrun.to(device)
    if args.model:
        print("Continue training from: ", args.model)
        try:
            # glrun.load_state_dict(torch.load(args.model, map_location=device))
            glrun.load_state_dict(torch.load(args.model))
        except Exception:
            print("Can't load model")
            return

    # optimizer
    optimizer = optim.Adam(glrun.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,20,30,50], gamma=0.5)
    loss_fn = GLoss(device)
    loss_fn.to(device)
    loss_fn_test = GLoss_test()
    loss_fn_test.to(device)

    os.makedirs('./logs', exist_ok=True)
    logger = get_logger(f'./logs/exp_{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}.log')
    logger.info('Start logging...\n')

    tstart = time.time()
    best_loss = 100
    ld = len(train_dataloader)

    for epoch in range(total_epoch):  # loop over the dataset multiple times

        scheduler.step()
        train_step = 0
        glrun.train()
        train_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"[Train Epoch {epoch}]")
        for i, data in enumerate(pbar, 0):
            # get the inputs; data is a list of [inputs, labels, ideal_d]
            raw_IQ, ideal_IQ, ideal_d = data
            raw_IQ = raw_IQ.to(device)  # [batch_size, 6, H, W]
            ideal_IQ = ideal_IQ.to(device)  # [batch_size, 1, H, W]
            ideal_d = ideal_d.to(device)  # [batch_size, 1, H, W]

            out_0, mu0 = glrun(raw_IQ[:, 0:2, :, :])
            out_1, mu1 = glrun(raw_IQ[:, 2:4, :, :])
            out_2, mu2 = glrun(raw_IQ[:, 4:6, :, :])

            optimizer.zero_grad()
            loss = loss_fn(out_0, out_1, out_2, ideal_IQ, ideal_d)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_step += 1
            avg_loss = train_loss / train_step
            pbar.set_postfix(loss=avg_loss)

        train_loss /= train_step
        time.ctime()
        info = f"[Epoch {epoch}/{total_epoch}] [Train Loss: {train_loss}] [time eclapsed {time.time() - tstart}]"
        logger.info(f"{info}\n")

        glrun.eval()
        test_step = 0
        test_loss = 0
        with torch.no_grad():
            pbar = tqdm(test_dataloader, desc=f"[Test Epoch {epoch}]")
            for i, data in enumerate(pbar, 0):
                # get the inputs; data is a list of [inputs, labels, ideal_d]
                raw_IQ, ideal_IQ, ideal_d = data
                raw_IQ = raw_IQ.to(device)  # [batch_size, 6, H, W]
                ideal_IQ = ideal_IQ.to(device)  # [batch_size, 1, H, W]
                ideal_d = ideal_d.to(device)  # [batch_size, 1, H, W]

                out_0, mu0 = glrun(raw_IQ[:, 0:2, :, :])
                out_1, mu1 = glrun(raw_IQ[:, 2:4, :, :])
                out_2, mu2 = glrun(raw_IQ[:, 4:6, :, :])
                loss = loss_fn_test(out_0, out_1, out_2, ideal_IQ, ideal_d)
                test_loss += loss.item()
                test_step += 1
                avg_test_loss = test_loss / test_step
                pbar.set_postfix(loss=avg_test_loss)

                if (epoch % 10 == 0) & (i % 5 == 0):
                    outputs = torch.concat((out_0, out_1, out_2), axis=1).cpu()
                    outputs = outputs[0].detach().numpy()
                    outputs.tofile(args.debug + '/epoch_' + str(epoch) + '_' + str(i))

        test_loss /= test_step
        info = f"[Epoch {epoch}/{total_epoch}] [Test Loss: {test_loss}]"
        logger.info(f"{info}\n")

        if test_loss < best_loss:
            best_loss = test_loss
            model_state_dict = glrun.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            checkpoint = {
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'epoch': epoch,
                'loss': train_loss
            }
            torch.save(checkpoint, f"{out_path}/checkpoint_best.pth")

        if epoch % 10 == 0:
            print("save @ epoch ", epoch + 1)
            torch.save(glrun.state_dict(), f"{out_path}/checkpoint_{epoch}.pth")

    # torch.save(glrun.state_dict(), out_model+'_{0}'.format(total_epoch))
    # print("Total running time: {0:.3f}".format(time.time() - tstart))
    logger.info("End logging.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dev', type=int, default=0, help='device id')
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                        help="Training learning rate. Default is 1e-3, or 2e-4 for FT")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 loss on parameters).')

    parser.add_argument("-in", "--train_path", type=str, default='../FLAT', help="Train set directory")
    parser.add_argument("-out", "--destination", type=str, default='./result', help="Output destination.")
    parser.add_argument("-d", "--debug", type=str, default='./result_debug', help="Result directory.")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to the trained DeepGLR.")
    parser.add_argument("-n", "--name", type=str, default='glrun.pkl', help="Name of model.",
                        )

    parser.add_argument("-e", "--epoch", type=int, default=200, help="Total epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="Training batch size. Default is 1")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train(args)
