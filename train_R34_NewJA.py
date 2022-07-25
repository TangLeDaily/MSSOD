use_wandb = False
Project_name = "MSSOD_V1"
This_name = "ResNet34_NewJA"

test_dataset_name = ['DUT', 'NJUD', 'NLPR']
# ['DUT', 'NJUD', 'NLPR', 'SSD', 'STEREO', 'LFSD', 'RGBD135']

import torch
import torch.nn as nn
import wandb
import argparse
import os
import random
from data import get_loader, test_dataset
from rootmodel.ResNet34_NewJA import *
from torch import optim
from utils import *


parser = argparse.ArgumentParser(description="PyTorch Data_Pre")
parser.add_argument("--train_image_root", default='datasets/train_ori/train_images/', type=str, help="train root path")
parser.add_argument("--train_gt_root", default='datasets/train_ori/train_masks/', type=str, help="train root path")
parser.add_argument("--train_depth_root", default='datasets/train_ori/train_depth/', type=str, help="train root path")
parser.add_argument("--test_root_path", default='datasets/test_data/', type=str, help="test root path")
parser.add_argument("--trainsize", default=256, type=int)
parser.add_argument("--savename", default='DUT', type=str)
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--frame", default=100, type=int, help="use cuda?")
parser.add_argument("--start_epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")  # default 16
parser.add_argument("--nEpochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0004, help="Learning Rate. Default=1e-4")
parser.add_argument("--threads", type=int, default=16, help="number of threads for data loader to use")
opt = parser.parse_args()


def get_yu(model):
    pretrained_dict = torch.load(
        "/home/tangle/code/MSSOD/checkpoints/MSSOD_base_nopool_Center/EP_DUT_Em_0.8884_Sm_0.8664_Fm_0.8878_MAE_0.0620_lr_0.00013107200000000006.pth",
        map_location='cpu')
    pretrained_dict = pretrained_dict['model']

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # 利用预训练模型的参数，更新模型
    model.load_state_dict(model_dict)
    return model

def get_yu2(model):
    kk = torch.load("checkpoints/default/net_g_300000.pth")
    model_dict = model.state_dict()
    kk = {k: v for k, v in kk.items() if k in model_dict}
    model_dict.update(kk)  # 利用预训练模型的参数，更新模型
    model.load_state_dict(model_dict)
    return model

def main():
    global model, opt
    if use_wandb:
        wandb.init(project=Project_name, name=This_name, entity="karledom")
    print(opt)

    print("===> Find Cuda")
    cuda = opt.cuda
    torch.cuda.set_device(2)
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
    # model = get_yu(model)
    # state_dict = torch.load("/home/tangle/code/MSSOD/checkpoints/MSSOD_V1_base/EP_DUT_MAE_0.1426_Em_0.7644_Sm_0.7810_Fm_0.8161_lr_0.00032.pth",
    #                         map_location='cpu')
    # state_dict = state_dict['model']
    # model.load_state_dict(state_dict)

    # state_dict = torch.load('model_name.pth', map_location='cpu')
    # model.load_state_dict(state_dict['model'])
    # model = get_yu(model)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    print("===> Training")
    last_Em = 0
    Now_best_Em = 0
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(optimizer, model, criterion, epoch, train_loader)
        if (epoch+1)%10 == 0:
            save_mae, save_Em, save_Sm, save_Fm = test(model, epoch, opt.savename)
            save_checkpoint(model, opt.savename, optimizer.param_groups[0]["lr"], save_mae, save_Em, save_Sm, save_Fm)
        if (epoch+1)%80 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.8
            # Now_best_Em = save_Em
            # if save_Em > Now_best_psnr:
            #     Now_best_psnr = save_Em
            # if Now_best_Em <= last_Em:
            #     for p in optimizer.param_groups:
            #         p['lr'] *= 0.8
            # last_Em = Now_best_Em

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

def get_test_dataloader(dataset_path, dataset, size=256):
    image_root = dataset_path + dataset + '/test_images/'
    gt_root = dataset_path + dataset + '/test_masks/'
    depth_root = dataset_path + dataset + '/test_depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, size)
    print('Now test dataset: ', dataset)

    return test_loader

def test(model, epoch, savename='DUT'):
    print(" -- Start eval --")
    model.eval()
    if not os.path.exists("checkpoints/{}/".format(This_name)):
        os.makedirs("checkpoints/{}/".format(This_name))
    log_write("checkpoints/{}/Test_log.txt".format(This_name), "===> Epoch_{}:".format(epoch))
    save_mae, save_Em, save_Sm, save_Fm = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for dataset_name in test_dataset_name:
            test_dataloader = get_test_dataloader(opt.test_root_path, dataset_name, 256)
            T_mae = AverageMeter()
            T_Fm = AverageMeter()
            T_Em = AverageMeter()
            T_Sm = AverageMeter()
            for i in range(test_dataloader.size):
                image, gt, depth, name = test_dataloader.load_data()
                if opt.cuda:
                    image = image.cuda()
                    depth = depth.cuda()
                    depth = torch.cat([depth, depth, depth], dim=1)
                    gt = gt.cuda()
                out = model(image, depth)
                out = out.sigmoid()
                T_mae.update(MAE(out, gt))
                T_Em.update(Em(out, gt))
                T_Sm.update(Sm(out, gt))
                T_Fm.update(Fm(out, gt))
            avg_mae = T_mae.avg
            avg_Em = T_Em.avg
            avg_Sm = T_Sm.avg
            avg_Fm = T_Fm.avg
            if dataset_name == savename:
                save_mae = avg_mae
                save_Em = avg_Em
                save_Sm = avg_Sm
                save_Fm = avg_Fm
            if use_wandb:
                wandb.log({'{}_MAE'.format(dataset_name): avg_mae,
                           '{}_Em'.format(dataset_name): avg_Em,
                           '{}_Sm'.format(dataset_name): avg_Sm,
                           '{}_Fm'.format(dataset_name): avg_Fm,
                           'Epoch':epoch})
            print("===> dataset_name:{} Em:{:.4f} Sm:{:.4f} Fm:{:.4f} MAE:{:.4f}".format(dataset_name, avg_Em, avg_Sm, avg_Fm, avg_mae))
            log_write("checkpoints/{}/Test_log.txt".format(This_name), "dataset_name:{} Em:{:.4f} Sm:{:.4f} Fm:{:.4f} MAE:{:.4f}".format(dataset_name, avg_Em, avg_Sm, avg_Fm, avg_mae))
    return save_mae, save_Em, save_Sm, save_Fm


def save_checkpoint(model, epoch, lr, save_mae, save_Em, save_Sm, save_Fm):
    global opt
    model_folder = "checkpoints/{}/".format(This_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "EP_{}_Em_{:.4f}_Sm_{:.4f}_Fm_{:.4f}_MAE_{:.4f}_lr_{}.pth".format(epoch, save_Em, save_Sm, save_Fm, save_mae, lr)
    torch.save({'model': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()