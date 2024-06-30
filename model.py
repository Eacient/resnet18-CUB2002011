from torchvision.models import swin_t, resnet34

def get_model(args):
    if args.arch == 'resnet':
        model = resnet34(num_classes=100)
    elif args.arch == 'swin':
        model= swin_t(num_classes=100)
    else:
        raise NotImplementedError()
    return model
