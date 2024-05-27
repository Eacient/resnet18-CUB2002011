import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with all the images and annotation files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        # Load the images and labels from the annotations file
        self.image_files, self.labels = self.load_annotations()

    def load_annotations(self):
        # Paths to the annotation files
        images_txt = os.path.join(self.root_dir, 'images.txt')
        labels_txt = os.path.join(self.root_dir, 'image_class_labels.txt')
        train_test_split_txt = os.path.join(self.root_dir, 'train_test_split.txt')

        # Load annotations
        images = pd.read_csv(images_txt, sep=' ', header=None, names=['img_id', 'filepath'])
        labels = pd.read_csv(labels_txt, sep=' ', header=None, names=['img_id', 'label'])
        train_test_split = pd.read_csv(train_test_split_txt, sep=' ', header=None, names=['img_id', 'is_training_img'])

        # Merge the annotations into a single dataframe
        data = pd.merge(images, labels, on='img_id')
        data = pd.merge(data, train_test_split, on='img_id')

        # Filter for training or testing images
        is_training = data['is_training_img'] == 1
        if self.train:
            data = data[is_training]
        else:
            data = data[~is_training]

        image_files = data['filepath'].tolist()
        labels = data['label'].tolist()

        return image_files, labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx] - 1  # Convert labels to 0-based index

        if self.transform:
            image = self.transform(image)

        return image, label