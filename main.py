import os
import time
import math
from tqdm import tqdm
import torch


from data import CUB200Dataset
from model import get_resnet18
from opt import get_param_group, get_opt, get_scheduler
from utils import Logger

def epoch(model, loss_fn, loader, logger, opt=None):
  if not opt:
    model.eval()
  else:
    model.train()

  data_time = []
  forward_time = []
  backward_time = []
  data_st = time.time()
  pbar = tqdm(loader)
  for data in pbar:

    img, y = data
    img = img.cuda()
    y = y.cuda(non_blocking=True)
    data_ed = time.time()
    data_time.append(data_ed - data_st)

    forward_st = time.time()

    y_pred = model(img)
    loss = loss_fn(y_pred, y)

    forward_ed = time.time()
    forward_time.append(forward_ed - forward_st)

    acc = (torch.argmax(y_pred.detach(), dim=1) == y).sum()
    logger.log_step(loss.detach().cpu().numpy(), acc.detach().cpu().numpy()/img.shape[0], img.shape[0])

    if opt:
      backward_st = time.time()

      opt.zero_grad()
      loss.backward()
      opt.step()

      backward_ed = time.time()
      backward_time.append(backward_ed - backward_st)

    if opt:
      pbar.set_description_str('Train')
      pbar.set_postfix_str(f'loss:{logger.loss_mean:.3f} acc:{logger.acc_mean:.3f} ' 
                f'data:{sum(data_time)/len(data_time):.5f} forward:{sum(forward_time)/len(forward_time):.5f} '
                f'backward:{sum(backward_time)/len(backward_time):.5f}')
    else:
      pbar.set_description_str("Eval")
      pbar.set_postfix_str(f'loss:{logger.loss_mean:.3f} acc:{logger.acc_mean:.3f} ' 
                f'data:{sum(data_time)/len(data_time):.5f} forward:{sum(forward_time)/len(forward_time):.5f}')

    data_st = time.time()
  logger.log_epoch()
  return logger.epoch_loss[-1], logger.epoch_acc[-1]

def train(n_epoch, model, train_loader, test_loader, loss_fn, opt, scheduler, save_dir):
  ckpt_path = os.path.join(save_dir, 'model_best.pth')
  model = model.cuda()
  logger = Logger()
  best_acc = -1
  for i in range(n_epoch):
    train_loss, train_acc = epoch(model, loss_fn, train_loader, logger, opt)
    test_loss, test_acc = epoch(model, loss_fn, test_loader, logger)
    print("|  {:>4} |    {:.5f} |   {:.5f} |    {:.5f} |   {:.5f} |"\
            .format(i, train_loss, train_acc, test_loss, test_acc))
    if test_acc >= best_acc:
      best_acc = test_acc
      print(f'saved epoch:{i}, best_acc:{best_acc:.3f}')
      torch.save(model.state_dict(), ckpt_path)
    scheduler.step()
  logger.plot()

if __name__ == "__main__":
  from torchvision import transforms
  from torch.utils.data import DataLoader
  import argparse
  import yaml

  from utils import set_seed

  parser = argparse.ArgumentParser()
  parser.add_argument('config', type=str, help='configuration yaml file')
  args = parser.parse_args()
  with open(args.config, 'r') as f:
    cfg = yaml.load(f, yaml.FullLoader)

  set_seed(0)

  root_dir = cfg['dataset']['root_dir']
  input_size = cfg['dataset']['input_size']
  infer_size = cfg['dataset']['infer_size']
  loader_args = cfg['dataloader']

  # Define transforms
  train_transform = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomResizedCrop(size=(input_size, input_size), scale=(0.4, 1), ratio=(0.5, 2)),
      transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
      transforms.RandomRotation(20),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # image_net
  ])

  test_transform = transforms.Compose([
      transforms.Resize((infer_size, infer_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # image_net
  ])

  train_dataset = CUB200Dataset(root_dir=root_dir, transform=train_transform, train=True)
  test_dataset = CUB200Dataset(root_dir=root_dir, transform=test_transform, train=False)

  train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
  test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

  pretrained = cfg['model']['pretrained']
  lr = cfg['optimizer']['lr']
  opt_args = cfg['optimizer']
  scheduler_args = cfg['scheduler']

  # make model
  model = get_resnet18(pretrained=pretrained)
  param_group = get_param_group(model, lr=lr, pretrained=pretrained)
  opt = get_opt(param_group, **opt_args)
  scheduler = get_scheduler(opt, **scheduler_args)
  loss_fn = torch.nn.CrossEntropyLoss()

  n_epoch = cfg['n_epochs']
  train(n_epoch, model, train_loader, test_loader, loss_fn, opt, scheduler, cfg['model']['save_dir'])

