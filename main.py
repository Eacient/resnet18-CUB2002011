import time
import math
from IPython.terminal.embed import warnings
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter


from data import get_loader
from model import get_model
from opt import get_opt, get_lr_sche
from utils import Logger


import torch
import time
from tqdm import tqdm

loss_fn = torch.nn.CrossEntropyLoss()
def one_epoch(epoch, model, loader, cutmix_or_mixup, opt, logger):
    data_time = []
    infer_time = []
    if opt is None:
        model = model.eval()
    else:
        model = model.train()
    pbar = tqdm(loader)
    data_st = time.time()
    for images, labels in pbar:
        if opt is not None:
            images, labels = cutmix_or_mixup(images, labels)
        images = images.cuda()
        labels = labels.cuda(non_blocking=True)
        data_ed = time.time()
        data_time.append(data_ed - data_st)
        infer_st = time.time()
        pred = model(images)
        loss = loss_fn(pred, labels)
        if opt is not None:
            acc = (torch.argmax(pred.detach(), dim=1) == torch.argmax(labels, dim=1)).sum()
        else:
            acc = (torch.argmax(pred.detach(), dim=1) == labels).sum()
        logger.log_step(loss.detach().cpu().numpy(), acc.detach().cpu().numpy()/images.shape[0], images.shape[0])
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        infer_ed = time.time()
        infer_time.append(infer_ed - infer_st)
        if opt:
            pbar.set_description_str('Train')
        else:
            pbar.set_description_str("Eval")
        pbar.set_postfix_str(f'loss:{logger.loss_mean:.3f} acc:{logger.acc_mean:.3f} ' 
                f'data:{sum(data_time)/len(data_time):.5f} infer:{sum(infer_time)/len(infer_time):.5f}') 

        data_st = time.time()
    logger.log_epoch()
        
    return logger.epoch_loss[-1], logger.epoch_acc[-1]

from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from utils import Logger
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--data', type=str, default='/kaggle/input/cifar100/CIFAR100')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4, required=False)
    parser.add_argument('--batch_size', type=int, default=256) # swin 128
    parser.add_argument('--opt', type=str, default='sgd') # swin adamw
    parser.add_argument('--lr', type=float, default=0.1) # swin 5e-6
    parser.add_argument('--wd', type=float, default=1e-4) # swin 0.05
    parser.add_argument('--lr_sche', type=str, default='step') #swin cosine
    parser.add_argument('--epochs', type=int, default=90) # swin 300
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=5e-7)
    parser.add_argument('--step_size', type=int, default=30, required=False)
    parser.add_argument('--gamma', type=float, default=0.1, required=False)
    
    args = parser.parse_args()
#     args = {
#         'arch': 'resnet',
#         'data': '/kaggle/input/cifar100/CIFAR100',
#         'num_classes' : 100,
#         'workers': 4,
#         'epochs': 2,
#         'batch_size': 128,
#         'opt': 'sgd',
#         'lr': 0.1,
#         'wd': 1e-4,
#         'lr_sche': 'step',
#         'step_size': 30,
#         'gamma': 0.1
#     }
    
#     args = {
#         'arch': 'swin',
#         'data': '/kaggle/input/cifar100/CIFAR100',
#         'num_classes' : 100,
#         'workers': 4,
#         'epochs': 2,
#         'batch_size': 64,
#         'opt': 'adamw',
#         'lr': 5e-6,
#         'wd': 0.05,
#         'lr_sche': 'cosine',
#         'warmup_epochs': 20,
#         'base_lr': 5e-4,
#         'min_lr': 5e-7
#     }
    
    Arg = namedtuple('Arg', args.keys())
    args = Arg(**args)
    
    
    train_loader, val_loader, test_loader, cutmix_or_mixup = get_loader(args)
    
    model = get_model(args)
    opt = get_opt(args, model)
    lr_sche = get_lr_sche(args, opt)
    
    writer = SummaryWriter('./')
    train_logger = Logger()
    val_logger = Logger()
    
    model = model.cuda()
    best_acc = -1
    for epoch in range(args.epochs):
        train_loss, train_acc = one_epoch(epoch, model, train_loader, cutmix_or_mixup, opt, train_logger)
        lr_sche.step()
        val_loss, val_acc = one_epoch(epoch, model, val_loader, None, None, val_logger)
#         print("|  {:>4} |    {:.5f} |   {:.5f} |    {:.5f} |   {:.5f} |"\
#             .format(i, train_loss, train_acc, val_loss, val_acc))
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Train/acc', train_acc, epoch)
        writer.add_scalar('Val/loss', val_loss, epoch)
        writer.add_scalar('Val/acc', val_acc, epoch)
        if val_acc > best_acc:
            torch.save(model.state_dict(), 'model_best.pth')
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict, f'checkpoint_{epoch:03d}.pth')
            
    best_param = torch.load('model_best.pth', map_location='cpu')
    model.load_state_dict(best_param)
    test_loss, test_acc = one_epoch(epoch, model, test_loader, None, None, val_logger)
    print("|  {:>4} |    {:.5f} |   {:.5f} |"\
            .format(args.epochs, test_loss, test_acc))
    writer.add_scalar('Val/loss', test_loss, args.epochs)
    writer.add_scalar('Val/acc', test_acc, args.epochs)

if __name__ == "__main__":
  main()
