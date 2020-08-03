""" 
GRN-SNDL-BCE
"""

import os
import random
import math
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import shutil


import argparse
from tensorboardX import SummaryWriter


import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import sys
sys.path.append('../')

from utils.dataGen import DataGeneratorML
from utils.NCA_ML import NCA_ML_CrossEntropy
from utils.model import ResNet18_cls, ResNet50_cls, WideResNet50_2_cls, InceptionV3_cls
from utils.metrics import KNNClassification, MetricTracker
from utils.LinearAverage import LinearAverage

model_choices = ['ResNet18', 'ResNet50', 'WideResNet50', 'InceptionV3']

parser = argparse.ArgumentParser(description='PyTorch SNCA Training for ML RS')
parser.add_argument('--data', metavar='DATA_DIR',  default='../data',
                        help='path to dataset (default: ../data)')
parser.add_argument('--dataset', metavar='DATASET',  default='ucmerced',
                        help='learning on the dataset (ucmerced)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='num_workers for data loading in pytorch, (default:8)')
parser.add_argument('--epochs', default=130, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--dim', default=128, type=int,
                    metavar='D', help='embedding dimension (default:128)')
parser.add_argument('--imgEXT', metavar='IMGEXT',  default='tif',
                        help='img extension of the dataset (default: tif)')
parser.add_argument('--temperature', default=0.05, type=float,
                    metavar='T', help='temperature parameter')
parser.add_argument('--memory-momentum', '--m-mementum', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('--margin', default=0.0, type=float,
                    help='classification margin')
parser.add_argument('--model', default='ResNet18', type=str, metavar='M',
                    choices=model_choices,
                        help='choose model for training, choices are: ' \
                         + ' | '.join(model_choices) + ' (default: ResNet18)')




args = parser.parse_args()

sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
print('saving file name is ', sv_name)

checkpoint_dir = os.path.join('./', sv_name, 'checkpoints')
logs_dir = os.path.join('./', sv_name, 'logs')

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))

def save_checkpoint(state, is_best, name):

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth.tar')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name + '_model_best.pth.tar'))



def main():
    global args, sv_name, logs_dir, checkpoint_dir

    write_arguments_to_file(args, os.path.join('./', sv_name, sv_name+'_arguments.txt'))

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_data_transform = transforms.Compose([
                                        transforms.Resize((256,256)),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])
    
    val_data_transform = transforms.Compose([
                                            transforms.Resize((256,256)),
                                            transforms.ToTensor(),
                                            normalize])
    

    train_dataGen = DataGeneratorML(data=args.data, 
                                            dataset=args.dataset, 
                                            imgExt=args.imgEXT,
                                            imgTransform=train_data_transform,
                                            phase='train')


    val_dataGen = DataGeneratorML(data=args.data, 
                                            dataset=args.dataset, 
                                            imgExt=args.imgEXT,
                                            imgTransform=val_data_transform,
                                            phase='val')


    train_data_loader = DataLoader(train_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    trainloader_wo_shuf = DataLoader(train_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    if args.dataset == 'DFC15_multilabel':
        clsNum = 8
    else:
        clsNum = 17
    
    if args.model == "ResNet18":
        model = ResNet18_cls(clsNum=clsNum, dim=args.dim)
    elif args.model == "ResNet50":
        model = ResNet50_cls(clsNum=clsNum, dim=args.dim)
    elif args.model == 'WideResNet50':
        model = WideResNet50_2_cls(clsNum=clsNum, dim=args.dim)
    elif args.model == 'InceptionV3':
        model = InceptionV3_cls(clsNum=clsNum, dim=args.dim)
    else:
        print('no model')

    if use_cuda:
        model.cuda()
        multiLabelLoss = torch.nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4, nesterov=True)

    best_acc = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            lemniscate = checkpoint['lemniscate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # define lemniscate and loss function (criterion)
        ndata = len(train_dataGen)
        lemniscate = LinearAverage(args.dim, ndata, args.temperature, args.memory_momentum).cuda()

    y_true = []

    for idx, data in enumerate(tqdm(trainloader_wo_shuf, desc="extracting training labels")):
        
        multiHot_batch = data['multiHot'].to(torch.device("cpu"))

        y_true += list(np.squeeze(multiHot_batch.numpy()).astype(np.float32))

    y_true = np.asarray(y_true)

    # print(y_true)

    criterion = NCA_ML_CrossEntropy(torch.tensor(y_true),
            args.margin / args.temperature).cuda()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))

    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        adjust_memory_update_rate(lemniscate, epoch)

        # train for one epoch
        train(train_data_loader, model, lemniscate, criterion, multiLabelLoss, optimizer, epoch, train_writer)
        acc = val(val_data_loader, model, lemniscate, y_true, epoch, val_writer)

        is_best_acc = acc > best_acc
        best_acc = max(best_acc, acc)
        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best_acc, sv_name)

        scheduler.step()


def train(trainloader, model, lemniscate, criterion, MLLoss, optimizer, epoch, train_writer):

    losses = MetricTracker()
    sncalosses = MetricTracker()
    celosses = MetricTracker()


    model.train()

    for idx, data in enumerate(tqdm(trainloader, desc="training")):

        imgs = data['img'].to(torch.device("cuda"))
        index = data["idx"].to(torch.device("cuda"))
        multiHots = data['multiHot'].to(torch.device("cuda"))

        feature, logits = model(imgs)

        output = lemniscate(feature, index)

        loss_snca = criterion(output, index)
        loss_ce = MLLoss(logits, multiHots)
        loss = loss_snca + loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))
        sncalosses.update(loss_snca.item(), imgs.size(0))
        celosses.update(loss_ce.item(), imgs.size(0))

    info = {
        "Loss": losses.avg,
        "SNCALoss": sncalosses.avg,
        "CELoss": celosses.avg
    }
    for tag, value in info.items():
        train_writer.add_scalar(tag, value, epoch)

    print('Train TotalLoss: {:.6f} SNCALoss: {:.6f} CELoss: {:.6f}'.format(
            losses.avg,
            sncalosses.avg,
            celosses.avg
            ))

def val(valloader, model, lemniscate, y_true, epoch, val_writer):

    train_features = lemniscate.memory.cpu().numpy()

    knn_classifier = KNNClassification(train_features, y_true)

    model.eval()

    y_val_true = []
    val_features = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valloader, desc="validation")):

            imgs = data['img'].to(torch.device("cuda"))
            multiHot_batch = data['multiHot'].to(torch.device("cpu"))

            feature, _ = model(imgs)

            val_features += list(feature.cpu().numpy().astype(np.float32))
            y_val_true += list(np.squeeze(multiHot_batch.numpy()).astype(np.float32))

    y_val_true = np.asarray(y_val_true)
    val_features = np.asarray(val_features)

    acc = knn_classifier(val_features, y_val_true)

    val_writer.add_scalar('KNN-SampleF1', acc, epoch)

    print('Validation KNN-SampleF1: {:.6f} '.format(
            acc,
            # hammingBallRadiusPrec.val,
            ))
    
    return acc

def adjust_memory_update_rate(lemniscate, epoch):
    if epoch >= 80:
        lemniscate.params[1] = 0.8
    if epoch >= 120:
        lemniscate.params[1] = 0.9


if __name__ == "__main__":
    main()


