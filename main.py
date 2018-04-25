import matplotlib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import cocodataset
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import numpy as np
import torchvision.models as models
import shutil
from mylogger import MyLogger
import os
from plot_log import plot_logger
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import math
import os


# get a list of the models that can be used
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='start epoch (use if resuming)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum (default=0.9)')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='WD', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size (default: 32)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='initialize pre-trained weights on Imagenet')
parser.add_argument('--resume', type=str, default='', metavar='PATH',
                    help='path to the latest checkpoint (default: none)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--logdir', default='logs', metavar='PATH',
                    help='path where to store the log and checkpoints')
parser.add_argument('--dataset', default='deepemotion', metavar='NAME',
                    help='dataset to use')
parser.add_argument('--write_freq', default=1, type=int, metavar='N',
                    help='write log every N epochs')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(42)
if args.cuda:
    torch.cuda.manual_seed(42)


kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
args.expname = ('{}_ep_{}_lr_{}_mom_{}'
                '_wd_{}_bs_{}_pre_{}'.format(args.model, args.epochs, args.lr,
                                             args.momentum, args.wd,
                                             args.batch_size, str(args.pretrained)))
save_path = os.path.join(args.logdir, args.expname)
writer = SummaryWriter(save_path)



# computed with utils.compute_mean()
def main():
    num_classes = 0
    if args.model=='inception_v3':
        size=(299,341)
    else:
        size=(224,256)


    train_transforms = [transforms.Scale(size[1]),
                        transforms.RandomSizedCrop(size[0]),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),  # sets input to range [0,1]
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


# create train data loader
    train_dataset = cocodataset('/home/pbehjati/Crop_Images/Train/train_Images', '/home/pbehjati/Crop_Images/Train/listfile1_train.txt', transform=transforms.Compose(train_transforms))
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, **kwargs)

    print('Loading Train Images...')
    val_transforms = [transforms.Scale(size[1]),
                      transforms.CenterCrop(size[0]),
                      transforms.ToTensor(),  # sets input to range [0,1]
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ]
    test_dataset = cocodataset('/home/pbehjati/Crop_Images/Train/test_images', '/home/pbehjati/Crop_Images/Train/listfile1_test.txt', transform=transforms.Compose(val_transforms))
    val_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=True, **kwargs)
    print('Loading Test Images...')

    num_classes=train_dataset.num_labels()

#create model
    if args.pretrained:
        print("=> using pre-trained '{}' model".format(args.model))
        model = models.__dict__[args.model](pretrained=True)
    else:
        print("=> using '{}' model from scratch".format(args.model))
        model = models.__dict__[args.model](pretrained=False)

    # change the last layer for resnet and Inception_v3
    if 'resnet' or 'inception' in args.model:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, train_dataset.num_labels())


    if args.cuda:
        model.cuda()

    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # prepare trainable parameters for the optimizer
    if args.pretrained:
        ignored_params = list(map(id, model.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             model.parameters())

        params = [{'params': base_params, 'lr': args.lr * 0.1, 'name': 'base'},
                  {'params': model.fc.parameters(), 'lr': args.lr, 'name': 'fc'}]
    else:
        params = model.parameters()

    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_score = checkpoint["best_score"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("=> no checkpoint at '{}'".format(args.resume))
    else:
        best_score = 0

    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        # update lr
        adjust_learning_rate(optimizer, epoch)

        # train step
        train(train_loader, model, criterion, optimizer, epoch)


        # evaluate
        score = evaluate(val_loader, model, criterion)



        # remember best score and save checkpoint
        is_best = score > best_score
        best_score = max(score, best_score)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict()
        }, is_best)


        # write log if necessary
        if epoch % args.write_freq == 0:
            exp_logger.write_log()
            plot_logger(exp_logger, save=True)



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch % 30 == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            print('Params "{}" lr updated to {}'.format(param_group['name'], param_group['lr']))


def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = torch.topk(output,1, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))


    res = []
    correct = correct.view(-1).float().sum(0, keepdim=True)
    res.append(correct.mul_(100.0 / batch_size))

    return res



def train(train_loader, model, criterion, optimizer, epoch):

    # monitor metrics
    batch_time = 0
    data_time = 0
    losses = 0
    top1 = 0
    processed = 0

    # set model to train mode
    model.train()

    end = time.time()
    n_iters = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time += time.time() - end

        if args.cuda:
            input, target = input.cuda(), target.cuda()

        input_var, target_var = Variable(input), Variable(target)


        if "inception_v3" in args.model:
            output=model(input_var)[0]

        # compute output
        output = model(input_var)
        loss=criterion(output,target_var)
        # measure accuracy and loss
        current_top1 = accuracy(output.data, target)[0][0]

        top1 += current_top1
        current_loss = loss.data[0]
        losses += current_loss

        #current iteration metrics
        writer.add_scalar('train/iter_loss', current_loss, epoch*n_iters+i)
        writer.add_scalar('train/iter_acc', current_top1, epoch*n_iters+i)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time += time.time() - end
        end = time.time()

        processed += 1

        # print if necessary
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time:.3f}\t'
                  'Data {data_time:.3f}\t'
                  'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                  'Accuracy {top1_val:.3f} ({top1_avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time / processed,
                      data_time=data_time / processed, loss_avg=losses / processed,
                      top1_avg=top1 / processed, loss_val=current_loss,
                      top1_val=current_top1))

        # average epoch metrics
        writer.add_scalar('train/epoch_loss', losses/processed, epoch)
        writer.add_scalar('train/epoch_acc', top1/processed, epoch)






def evaluate(val_loader, model, criterion):
    # monitor metrics
    batch_time = 0
    losses = 0
    top1 = 0
    processed = 0

    # set model to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.cuda:
            input, target = input.cuda(), target.cuda()

        input_var, target_var = Variable(input), Variable(target)

        # compute output
        output = model(input_var)
        _, pred = torch.max(output.data,1)
        result += confusion_matrix(y_true=target,
                                  y_pred=pred)
        # Compute loss
        loss = criterion(output, target_var)

        # measure accuracy and loss

        current_top1 = accuracy(output.data, target)[0][0]


        top1 += current_top1
        current_loss = loss.data[0]
        losses += current_loss

        # measure elapsed time
        batch_time += time.time() - end
        end = time.time()

        processed += 1

        # print if necessary
        if i % args.print_freq == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time:.3f}\t'
                  'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                  'Accuracy {top1_val:.3f} ({top1_avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time / processed,
                      loss_avg=losses / processed, top1_avg=top1 / processed,
                      loss_val=current_loss, top1_val=current_top1))

    print(result)
    print(' * Val accuracy {top1:.3f}'
          .format(top1=top1 / processed))


    writer.add_scalar('val/epoch_loss', losses/processed, epoch)
    writer.add_scalar('val/epoch_acc', top1/processed, epoch)


    return top1 / processed

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    save_path = os.path.join(args.logdir, args.expname) + '/'
    torch.save(state, save_path + filename)
    if is_best:
        shutil.copyfile(save_path + filename, save_path + 'model_best.pth.tar')


if __name__ == '__main__':
    main()

