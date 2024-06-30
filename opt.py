from torch import optim

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]

def get_opt(args, model):
    if args.opt == 'sgd':
        opt = optim.SGD(set_weight_decay(model), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    elif args.opt == 'adamw':
        opt = optim.AdamW(set_weight_decay(model), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999), eps=1e-8)
    else:
        raise NotImplementedError()
    return opt

import math
def warmup_cosine(args):
    warmup_epochs = args.warmup_epochs
    num_epochs = args.epochs
    cosine_epochs = num_epochs - warmup_epochs
    base_lr = args.base_lr
    warmup_lr = args.lr
    min_lr = args.min_lr

    def warmup_cosine_handler(epoch):
        if epoch < warmup_epochs:
            return (base_lr - warmup_lr) * epoch / warmup_epochs + warmup_lr
        else:
            return min_lr + 0.5*(base_lr-min_lr)*(1.0+math.cos(math.pi*(epoch-warmup_epochs)/cosine_epochs))
        
    return warmup_cosine_handler


def get_lr_sche(args, opt):
    if args.lr_sche == 'step':
        lr_sche = optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_sche == 'cosine':
        lr_sche = optim.lr_scheduler.LambdaLR(opt, lr_lambda = warmup_cosine(args))
    else:
        return NotImplementedError()
    return lr_sche
