import torchvision
import torch.nn.functional as F
import lightning as L
import torch

from enum import Enum


class Backbone(Enum):
    # ResNet
    RESNET18 = torchvision.models.resnet18
    RESNET34 = torchvision.models.resnet34
    RESNET50 = torchvision.models.resnet50
    RESNET101 = torchvision.models.resnet101
    RESNET152 = torchvision.models.resnet152

    # attention based
    VIT = torchvision.models.vit_b_16
    SWIN = torchvision.models.swin_b

    # MobileNet
    MOBILENET_V2 = torchvision.models.mobilenet_v2

    # ResNext
    RESNEXT50_32X4D = torchvision.models.resnext50_32x4d
    RESNEXT101_32X8D = torchvision.models.resnext101_32x8d


def get_backbone(backbone: Backbone):
    if backbone == Backbone.RESNET18:
        return (
            torchvision.models.resnet18,
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        )
    elif backbone == Backbone.RESNET34:
        return (
            torchvision.models.resnet34,
            torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
        )
    elif backbone == Backbone.RESNET50:
        return (
            torchvision.models.resnet50,
            torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
        )
    elif backbone == Backbone.RESNET101:
        return (
            torchvision.models.resnet101,
            torchvision.models.ResNet101_Weights.IMAGENET1K_V1,
        )
    elif backbone == Backbone.RESNET152:
        return (
            torchvision.models.resnet152,
            torchvision.models.ResNet152_Weights.IMAGENET1K_V1,
        )
    elif backbone == Backbone.MOBILENET_V2:
        return (
            torchvision.models.mobilenet_v2,
            torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2,
        )
    elif backbone == Backbone.RESNEXT50_32X4D:
        return (
            torchvision.models.resnext50_32x4d,
            torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
        )
    elif backbone == Backbone.RESNEXT101_32X8D:
        return (
            torchvision.models.resnext101_32x8d,
            torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2,
        )
    elif backbone == Backbone.VIT:
        return torchvision.models.vit_b_16, torchvision.models.ViT_B_16_Weights.DEFAULT
    elif backbone == Backbone.SWIN:
        return torchvision.models.swin_b, torchvision.models.Swin_B_Weights.DEFAULT


# Model to count number of apples in an image, by regressing the count
class CountModel_reg(L.LightningModule):
    def __init__(self, backbone: Backbone = Backbone.RESNET18):
        super().__init__()
        # self.model = torchvision.models.resnet18(weights =  torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # self.model = torchvision.models.resnet101(weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

        model, weights = get_backbone(backbone)
        self.model = model(weights=weights)

        if backbone == Backbone.VIT or backbone == Backbone.SWIN:
            self.model.heads = torch.nn.Linear(self.model.heads.head.in_features, 1)
        else:
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

        self.test_step_output = {
            "gt": [],
            "pred": [],
        }

    def forward(self, x):
        return self.model(x)

    def loss(self, output, label):
        """
        function to calculate loss,
        output is output from model
        label is the ground truth
        """
        count = label["count"]
        count = count.unsqueeze(1).float()
        return F.mse_loss(output, count)

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.loss(output, label)
        self.log("train_loss", loss, batch_size=img.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0005)

        # # Decay LR on Plateau
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.5, patience=5
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "monitor": "val_loss",
        # }

        return optimizer

    ############################
    def predict(self, x):
        return torch.round(self(x))

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.loss(output, label)
        self.log("val_loss", loss, batch_size=img.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        count = label["count"]  # list of counts for batch of images
        count = count.unsqueeze(1).float()

        model_output = self.predict(img)
        self.test_step_output["gt"].extend(count)
        self.test_step_output["pred"].extend(model_output)

    def on_test_epoch_end(self):
        gt = torch.tensor(self.test_step_output["gt"])
        pred = torch.tensor(self.test_step_output["pred"])

        accuracy = torch.mean((gt == pred).float())
        self.log("accuracy", accuracy * 100)
        self.log("mae", F.l1_loss(gt, pred))

        # calculate the accuracy for each count
        unique_counts = torch.unique(gt)
        accuracy_per_count = {}
        for count in unique_counts:
            mask = gt == count
            count_accuracy = torch.mean((gt[mask] == pred[mask]).float())
            accuracy_per_count[int(count.item())] = count_accuracy.item() * 100

        for count, acc in accuracy_per_count.items():
            self.log(f"accuracy_count_{count}", acc)

        self.test_step_output = {
            "gt": [],
            "pred": [],
        }


class CountModel_cls(CountModel_reg):
    def __init__(self, backbone: Backbone = Backbone.RESNET18):
        super().__init__(backbone)
        self.max_count = 28
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.max_count)
        if backbone == Backbone.VIT or backbone == Backbone.SWIN:
            self.model.heads = torch.nn.Linear(
                self.model.heads.in_features, self.max_count
            )
        else:
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.max_count)

        # add a softmax layer to output probabilities
        self.softmax = torch.nn.Softmax(dim=1)

        self.count_freq = {
            3: 0.12296564195298372,
            7: 0.039783001808318265,
            12: 0.018083182640144666,
            2: 0.14285714285714285,
            5: 0.054249547920433995,
            17: 0.0054249547920434,
            14: 0.003616636528028933,
            4: 0.081374321880651,
            1: 0.36347197106690776,
            22: 0.0018083182640144665,
            6: 0.048824593128390596,
            10: 0.023508137432188065,
            8: 0.0325497287522604,
            15: 0.0054249547920434,
            9: 0.019891500904159132,
            18: 0.007233273056057866,
            11: 0.014466546112115732,
            13: 0.007233273056057866,
            20: 0.0018083182640144665,
            26: 0.0018083182640144665,
            16: 0.0018083182640144665,
            23: 0.0018083182640144665,
        }

        for i in range(self.max_count):
            if i not in self.count_freq:
                self.count_freq[i] = 0.0018083182640144665

        self.count_freq = [1 / self.count_freq[i] for i in range(self.max_count)]
        self.count_freq = torch.tensor(self.count_freq)
        self.count_freq = self.count_freq / torch.sum(self.count_freq)
        self.count_freq = self.count_freq.float().cuda()

    def forward(self, x):
        return self.model(x)

    def loss(self, output, label):
        count = label["count"]

        return F.cross_entropy(output, count)

    def predict(self, x):
        return torch.argmax(self(x), dim=1)

early_stop_callback = L.pytorch.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, verbose=True, mode="min"
)
