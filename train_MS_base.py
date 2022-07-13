use_wandb = False
Project_name = "SPVSR_V2"
This_name = "EDVR_SP_L5FSR"

import torch
import torch.nn as nn
import wandb
import argparse
import os
import random
from data import get_loader
from rootmodel.MS_base import *
from torch import optim
from utils import *


parser = argparse.ArgumentParser(description="PyTorch Data_Pre")
parser.add_argument("--train_image_root", default='datasets/train_ori/train_images/', type=str, help="train root path")
parser.add_argument("--train_gt_root", default='datasets/train_ori/train_masks/', type=str, help="train root path")
parser.add_argument("--train_depth_root", default='datasets/train_ori/train_depth/', type=str, help="train root path")
parser.add_argument("--test_root_path", default='datasets/test/', type=str, help="test root path")
parser.add_argument("--trainsize",default=256, type=int)
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--frame", default=100, type=int, help="use cuda?")
parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")  # default 16
parser.add_argument("--nEpochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0004, help="Learning Rate. Default=1e-4")
parser.add_argument("--threads", type=int, default=16, help="number of threads for data loader to use")
opt = parser.parse_args()


def main():
    global model, opt
    if use_wandb:
        wandb.init(project=Project_name, name=This_name, entity="karledom")
    print(opt)

    print("===> Find Cuda")
    cuda = opt.cuda
    torch.cuda.set_device(0)
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    # cudnn.benchmark = True

    print("===> Loading datasets")
    train_loader = get_loader(opt.train_image_root, opt.train_gt_root, opt.train_depth_root, batchsize=opt.batchSize, trainsize=opt.trainsize)

    print("===> Building model")
    model = MSSOD()
    criterion = torch.nn.BCEWithLogitsLoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    print("===> Do Resume Or Skip")
    # checkpoint = torch.load("checkpoints/edvr_deblur/model_epoch_212_psnr_33.3424.pth", map_location='cpu')
    # model.load_state_dict(checkpoint.state_dict())
    # model = get_yu(model)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    print("===> Training")

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(optimizer, model, criterion, epoch, train_loader)

def train(optimizer, model, criterion, epoch, train_loader):
    global opt
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    avg_loss = AverageMeter()
    if opt.cuda:
        model = model.cuda()
    for iteration, batch in enumerate(train_loader):
        images, gts, depths = batch
        if opt.cuda:
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
            depths = torch.cat([depths, depths, depths], dim=1)
        out = model(images, depths)
        loss = criterion(out, gts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())
        if iteration % 50 == 0:
            if use_wandb:
                wandb.log({'epoch': epoch, 'iter_loss': avg_loss.avg})
            print('epoch_iter_{}_loss is {:.10f}'.format(iteration, avg_loss.avg))


if __name__ == "__main__":
    main()