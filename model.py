import torch
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import torch.nn as nn

def get_resnet18(pretrained=False):

  if not pretrained:
    model = resnet18(num_classes=200)
    _ = nn.init.kaiming_normal_(model.fc.weight, mode='fan_out')
    _ = nn.init.zeros_(model.fc.bias)

  else:
    weights = ResNet18_Weights.verify('IMAGENET1K_V1')
    state_dict = weights.get_state_dict(progress=True, check_hash=True)
    _ = state_dict.pop('fc.weight')
    _ = state_dict.pop('fc.bias')

    model = resnet18(num_classes=200)
    model.load_state_dict(state_dict, strict=False)
    _ = nn.init.kaiming_normal_(model.fc.weight, mode='fan_out')
    _ = nn.init.zeros_(model.fc.bias)

  return model
