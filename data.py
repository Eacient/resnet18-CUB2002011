from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2

def get_loader(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    augmentations = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,   
    ]

    val_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]

    train_dir = args.data + '/TRAIN'
    val_dir = args.data + '/VALID'
    test_dir = args.data + '/TEST'

    train_dataset = datasets.ImageFolder(train_dir, transforms.Compose(augmentations))
    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose(val_transforms))
    test_dataset = datasets.ImageFolder(test_dir, transforms.Compose(val_transforms))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,)

    cutmix = v2.CutMix(num_classes=args.num_classes)
    mixup = v2.MixUp(num_classes=args.num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    return train_loader, val_loader, test_loader, cutmix_or_mixup
