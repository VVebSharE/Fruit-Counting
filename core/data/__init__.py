from torchvision import transforms
from .dataset import CustomCocoDataset, _collate_fn as _collate_fn1
from .dataset2 import count_MinneApple, _collate_fn as _collate_fn2
from torch.utils.data import DataLoader
import os
import lightning as L


apple_roboflow_kwargs_train = {
    "root": "D:/DATA/APPLE/Roboflow/train",
    "annFile": "D:/DATA/APPLE/Roboflow/train/_annotations.coco.json",
}
apple_roboflow_kwargs_valid = {
    "root": "D:/DATA/APPLE/Roboflow/valid",
    "annFile": "D:/DATA/APPLE/Roboflow/valid/_annotations.coco.json",
}

apple_minneapple_kwargs_train = {
    "root_dir": "D:/DATA/APPLE/MinneApple/counting/",
    "split": "train",
}
apple_minneapple_kwargs_valid = {
    "root_dir": "D:/DATA/APPLE/MinneApple/counting/",
    "split": "val",
}

def which_dataset(dataset):
    if dataset == "apple_roboflow":
        return CustomCocoDataset, _collate_fn1, apple_roboflow_kwargs_train, apple_roboflow_kwargs_valid
    elif dataset == "apple_minneapple":
        return count_MinneApple, _collate_fn2, apple_minneapple_kwargs_train, apple_minneapple_kwargs_valid
    else:
        raise ValueError("Invalid dataset name")

class AppleDataModule(L.LightningDataModule):
    def __init__(self, dataset: str = "apple_roboflow"):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.dataset = dataset
        self.dataset_class, self.collate_fn, self.train_kwargs, self.valid_kwargs = which_dataset(dataset)

    def prepare_data(self):
        # download, split, etc...
        
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_class(
                **self.train_kwargs,
                transform=self.transform,
            )

            self.valid_dataset = self.dataset_class(
                **self.valid_kwargs,
                transform=self.transform,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test':
            self.test_dataset = self.dataset_class(
                **self.valid_kwargs,
                transform=self.transform,
            )

        if stage == 'predict':
            self.predict_dataset = self.dataset_class(
                **self.valid_kwargs,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=2,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            collate_fn=self.collate_fn
        )