from torch import optim

def get_param_group(model, lr, pretrained=False):

  def get_base_param(named_params):
    for t in named_params:
      if t[0] == 'fc.weight' or t[0] == 'fc.bias':
        continue
      else:
        yield t[1]

  if not pretrained:
    return model.parameters()

  base_param = get_base_param(model.named_parameters())
  fc_param = [model.fc.weight, model.fc.bias]

  return [{'params': base_param, 'lr': 0.1 * lr},
    {'params': fc_param},]

def get_opt(param_group, lr=1e-2, weight_decay=1e-4, name='SGD', momentum=0, nesterov=False):
  if name == 'SGD':
    return optim.SGD(param_group, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
  elif name == 'Adam':
    return optim.Adam(param_group, lr=lr, weight_decay=weight_decay)
  else:
    raise NotImplementedError()

def get_scheduler(opt, name='Step', **kwargs):
  if name == 'Step':
    return optim.lr_scheduler.StepLR(opt, step_size=kwargs['step_size'], gamma=0.5)
  elif name == 'Poly':
    return optim.lr_scheduler.PolynomialLR(opt, max_epoch=kwargs['max_epoch'], power=2)
  elif name == 'Cosine':
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_max=kwargs['max_epoch'], eta_min=0)
  elif name == 'rlop':
    return optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=4)
