from torchvision import transforms
from .dataset import CustomCocoDataset, _collate_fn
from torch.utils.data import DataLoader
import os
import lightning as L

class AppleRoboFlowDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = None):
        super().__init__()
        self.data_dir = data_dir  # currently not used
        self.transform = transforms.Compose(
            [
                # transforms.Resize(
                #     (224, 224)
                # ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.train_images_dir = "D:/DATA/APPLE/Roboflow/train"
        self.train_annotations_file = os.path.join(self.train_images_dir, "_annotations.coco.json")
        self.valid_images_dir = "D:/DATA/APPLE/Roboflow/valid"
        self.valid_annotations_file = os.path.join(self.valid_images_dir, "_annotations.coco.json")

    def prepare_data(self):
        # download, split, etc...
        
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomCocoDataset(
                root=self.train_images_dir,
                annFile=self.train_annotations_file,
                transform=self.transform,
            )

            self.valid_dataset = CustomCocoDataset(
                root=self.valid_images_dir,
                annFile=self.valid_annotations_file,
                transform=self.transform,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test':
            self.test_dataset = CustomCocoDataset(
                root=self.valid_images_dir,
                annFile=self.valid_annotations_file,
                transform=self.transform,
            )

        if stage == 'predict':
            self.predict_dataset = CustomCocoDataset(
                root=self.valid_images_dir,
                annFile=self.valid_annotations_file,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=2,
            collate_fn=_collate_fn,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            collate_fn=_collate_fn,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            collate_fn=_collate_fn,
            persistent_workers=True
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            collate_fn=_collate_fn
        )
