# dataset for MinneApple dataset
# dataset in D:\DATA\APPLE\MinneApple\counting\

# folder structure:
# MinneApple/
#     counting/
#         train/
#             images/
#             train_ground_truth.txt
#                 "Image,count
#                 images_0000.png,0
#                 ...
#                 "
#         val/
#             images/
#             valid_ground_truth.txt
#                 "Image,count
#                 images_0000.png,0
#                 ...
#                 "
#         test/
#             images/

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class count_MinneApple(Dataset):
    def __init__(self, root_dir, split = 'train', transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        self.image_dir = os.path.join(self.root_dir, self.split, 'images')
        self.gt_file = os.path.join(self.root_dir, self.split, self.split + '_ground_truth.txt')

        self.image_list = os.listdir(self.image_dir)
        self.gt = self.get_gt()
    
    def get_gt(self):
        gt = {}
        with open(self.gt_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                image_name, count = line.strip().split(',')
                gt[image_name] = int(count)
        return gt

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        count = self.gt[image_name]
        if self.transform:
            image = self.transform(image)
    
        label = {
            "count": count
        }
        return image, label
    
    def _view(self, idx):
        """to visualize the image and its annotation

        """
        import matplotlib.pyplot as plt
        import numpy as np

        img, label = self.__getitem__(idx)

        fig, ax = plt.subplots(1)
        ax.imshow(np.array(img).transpose(1, 2, 0))

        #add count of objects in bottom left corner
        ax.text(0, img.size(1), f"Count: {label['count']}", fontsize=12, color="white", verticalalignment="bottom")

        plt.show()

def _collate_fn(batch):
    '''
        batch: list of tuple (image, label)
    '''
    
    images = [item[0] for item in batch]
    labels = {
        "count": [item[1]["count"] for item in batch]
    }

    images = torch.stack(images, dim=0)
    labels["count"] = torch.tensor(labels["count"])

    return images, labels



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time

    # Define paths
    root_dir = "D:/DATA/APPLE/MinneApple/counting/"

    # Define a basic transform (resizing, normalization, etc. as needed)
    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),
            transforms.ToTensor(),
        ]
    )

    # Create custom dataset
    valid_dataset = count_MinneApple(
        root_dir=root_dir,
        split = 'val',
        transform=transform,
    )

    # Get an image and its annotation
    img, label = valid_dataset[0]
    print(label)
    

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=40,
        shuffle=False,
        # num_workers=2,
        # collate_fn=_collate_fn
    )

    print(valid_dataset.__len__())
    
    # # Sample usage to test the DataLoader
    # for i, (img, label) in enumerate(valid_loader):
    #     print(i, img.shape, label)
    #     time.sleep(1)
    #     if i == 5:
    #         break
    # print("done")

    # Sample usage to visualize an image
    valid_dataset._view(1000)
    valid_dataset._view(1004)
    valid_dataset._view(2000)
