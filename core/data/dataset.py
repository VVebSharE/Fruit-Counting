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
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
from enum import Enum

class Modality(Enum):
    RGB = "RGB"
    EQUALIZED = "equalized"
    HSV = "HSV"
    SHARPENED = "sharpened"
    GRAYSCALE = "grayscale"
    LAPLACIAN = "laplacian"
    DCT = "DCT"
    EDGES = "edges"


def convert_modality(img: Image.Image, target: str) -> np.ndarray:
    """
    Convert an image in RGB format to a target modality.

    Args:
        img (PIL.Image.Image): Input image in RGB format (3 channels).
        target (str): Target modality. One of ["RGB", "equalized", "HSV", "sharpened", "grayscale", "laplacian", "DCT", "edges"].

    Returns:
        np.ndarray: Converted image as a NumPy array in the target modality.
    """
    if target == "RGB":
        # Return as-is
        return np.array(img)

    # Convert the image to grayscale if required for further processing
    img_gray = img.convert("L")  # Grayscale version
    img_np = np.array(img)       # RGB as NumPy array

    if target == "equalized":
        # Histogram equalization
        img_eq = ImageOps.equalize(img)
        return np.array(img_eq)

    elif target == "HSV":
        # Convert to HSV
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        return img_hsv

    elif target == "sharpened":
        # Sharpen the image
        enhancer = ImageEnhance.Sharpness(img)
        img_sharpened = enhancer.enhance(2.0)  # Sharpening factor
        return np.array(img_sharpened)

    elif target == "grayscale":
        # Return grayscale as 3 channels
        img_gray_3ch = np.stack([np.array(img_gray)] * 3, axis=-1)
        return img_gray_3ch

    elif target == "laplacian":
        # Apply Laplacian filter
        img_lap = cv2.Laplacian(np.array(img_gray), cv2.CV_64F)
        img_lap = cv2.normalize(img_lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_lap_3ch = np.stack([img_lap] * 3, axis=-1)
        return img_lap_3ch

    elif target == "DCT":
        # Apply Discrete Cosine Transform (DCT)
        img_gray_np = np.array(img_gray, dtype=np.float32)
        dct = cv2.dct(img_gray_np)
        dct_norm = cv2.normalize(dct, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dct_3ch = np.stack([dct_norm] * 3, axis=-1)
        return dct_3ch

    elif target == "edges":
        # Detect edges using Canny edge detection
        edges = cv2.Canny(np.array(img_gray), threshold1=100, threshold2=200)
        edges_3ch = np.stack([edges] * 3, axis=-1)
        return edges_3ch

    else:
        raise ValueError(f"Unsupported modality: {target}")




# custom dataset for COCO format
class CustomCocoDataset(CocoDetection):
    def __init__(self, root: str, annFile: str, modality: Modality = Modality.RGB, transform = None):
        """
        Args:
            root (str): Root directory where images are stored.
            annFile (str): Path to annotation file.
            modality (Modality): Desired modality for the images.
            transform (callable, optional): A function/transform that takes in an image and target and returns a transformed version.
        """
        super().__init__(root, annFile, transform)
        self.modality = modality.value

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Preprocess target as needed

        if(self.modality != False):
            img = convert_modality(img, self.modality.value)
            img = Image.fromarray(img)

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



# to show dataset usage
# if __name__ == "__main__":
#     from torchvision import transforms
#     from torch.utils.data import DataLoader
#     import time

#     # Define paths
#     valid_images_dir = "D:/DATA/APPLE/Roboflow/train"
#     valid_annotations_file = os.path.join(valid_images_dir, "_annotations.coco.json")

#     # Define a basic transform (resizing, normalization, etc. as needed)
#     transform = transforms.Compose(
#         [
#             # transforms.Resize(
#             #     (224, 224)
#             # ),
#             transforms.ToTensor(),
#         ]
#     )

#     # Create custom dataset
#     valid_dataset = CustomCocoDataset(
#         root=valid_images_dir,
#         annFile=valid_annotations_file,
#         transform=transform,
#     )

#     # Get an image and its annotation
#     img, label = valid_dataset[0]
#     print(label)
    

#     valid_loader = DataLoader(
#         valid_dataset,
#         batch_size=40,
#         shuffle=False,
#         # num_workers=2,
#         collate_fn=_collate_fn
#     )

#     print(valid_dataset.__len__())

#     # Sample usage to test the DataLoader
    
#     time.sleep(1)
#     time_start = time.time()
#     for images, targets in valid_loader:
#         print("Images batch shape:", images.shape)  # Shape of each image in the batch
#         print("Targets batch:", targets)  # Bounding boxes and category IDs
    
#     print("Time taken:", time.time() - time_start,"sec")  # Time taken to load the dataset

#     # to visualize the annotated image
#     valid_dataset._view(0)
#     valid_dataset._view(1)

#to see effect of different modality
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    modalities = [i for i in Modality]
    
    valid_images_dir = "D:/DATA/APPLE/Roboflow/train"
    valid_annotations_file = os.path.join(valid_images_dir, "_annotations.coco.json")

    # Create custom dataset
    valid_dataset = CustomCocoDataset(
        root=valid_images_dir,
        annFile=valid_annotations_file,
        transform=None,
        modality=False
    )

    img, label = valid_dataset[0]

    print(type(img))

    for modality in modalities:
        print(modality)
        img_modality = convert_modality(img, modality)
        print(img_modality.shape)
        
        plt.imshow(img_modality)
        plt.axis("off")
        plt.show()
        print("----")
