# dataset in D:\DATA\APPLE\Roboflow\

# folder structure
# /train
#   _annotations.coco.json
#   img1.jpg
#   img2.jpg
#   ...
# /valid
#   _annotations.coco.json
#   img1.jpg
#   img2.jpg
#   ...
#  maximum count of apples in any image is 27

import os
from torchvision.datasets import CocoDetection
import torch
# import cv2

def convert_modality(img, target):
    if(target == "RGB"):
        return img
    elif(target == "equalized"):
        return cv2.equalizeHist(img)
    elif(target == "CLAHE"):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    elif(target == "YUV"):
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        return y
    elif(target == "HSV"):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return h
    elif(target == "HLS"):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        return h


# custom dataset for COCO format
class CustomCocoDataset(CocoDetection):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Preprocess target as needed

        bbox = [ann["bbox"] for ann in target]  # list of [x, y, width, height] for each bounding box, could be of variable length
        category_id = [ann["category_id"] for ann in target]  # Get all category IDs

        # Return image, bbox, and category information as a dictionary
        label = { 
            "bboxes": bbox, 
            "category_ids": category_id,
            "count": len(bbox)
        }

        return img, label

    def _view(self, index):
        """to visualize the annotated image

        Args:
            index (int): for image at specific index
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        img, label = self.__getitem__(index)

        fig, ax = plt.subplots(1)
        ax.imshow(np.array(img).transpose(1, 2, 0))

        for i in range(label["count"]):
            x, y, w, h = label["bboxes"][i]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")
            ax.add_patch(rect)

        #add count of objects in bottom left corner
        ax.text(0, img.size(1), f"Count: {label['count']}", fontsize=12, color="white", verticalalignment="bottom")

        plt.show()

def _collate_fn(batch):
    '''
        batch: list of tuple (image, label)
    '''
    
    images = [item[0] for item in batch]
    labels = {
        "bboxes": [item[1]["bboxes"] for item in batch],
        "category_ids": [item[1]["category_ids"] for item in batch],
        "count": [item[1]["count"] for item in batch]
    }

    images = torch.stack(images, dim=0)
    labels["count"] = torch.tensor(labels["count"])

    return images, labels



# to show usage
if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import time

    # Define paths
    valid_images_dir = "D:/DATA/APPLE/Roboflow/train"
    valid_annotations_file = os.path.join(valid_images_dir, "_annotations.coco.json")

    # Define a basic transform (resizing, normalization, etc. as needed)
    transform = transforms.Compose(
        [
            # transforms.Resize(
            #     (224, 224)
            # ),
            transforms.ToTensor(),
        ]
    )

    # Create custom dataset
    valid_dataset = CustomCocoDataset(
        root=valid_images_dir,
        annFile=valid_annotations_file,
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
        collate_fn=_collate_fn
    )

    print(valid_dataset.__len__())

    # Sample usage to test the DataLoader
    
    time.sleep(1)
    time_start = time.time()
    for images, targets in valid_loader:
        print("Images batch shape:", images.shape)  # Shape of each image in the batch
        print("Targets batch:", targets)  # Bounding boxes and category IDs
    
    print("Time taken:", time.time() - time_start,"sec")  # Time taken to load the dataset

    # to visualize the annotated image
    valid_dataset._view(0)
    valid_dataset._view(1)
