# Model to do object detection
import torch
import torchvision
import torch.optim as optim
import lightning as L
import torch.nn.functional as F

class CountModel_det(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT", box_score_thresh=0.8)

        self.test_step_output = {
            "gt": [],
            "pred": [],
        }

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)

    def training_step(self, batch, batch_idx):
        img, label = batch

        bbox = label["bboxes"]

        targets = []
        for i in range(len(bbox)):
            targets.append({
                "boxes": bbox[i],
                "labels": torch.tensor([53]),
            })

        loss_dict = self.model(img, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses, batch_size=img.size(0))
        return losses

    def validation_step(self, batch, batch_idx):
        img, label = batch

        bbox = label["bboxes"]

        targets = []
        for i in range(len(bbox)):
            targets.append({
                "boxes": torch.tensor(bbox[i]),
                "labels": torch.tensor([53]),
            })

        loss_dict = self.model(img, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("val_loss", losses, batch_size=img.size(0))
        return losses

    def test_step(self, batch, batch_idx):
        img, label = batch
        count = label["count"]
        # count = count.unsqueeze(1).float()

        model_output = self.model(img)
        model_count = [
            len(model_output[i]["boxes"]) for i in range(len(model_output))
        ]

        count = count.tolist()

        self.test_step_output["gt"].extend(count)
        self.test_step_output["pred"].extend(model_count)

    def on_test_epoch_end(self):
        gt = torch.tensor(self.test_step_output["gt"])
        pred = torch.tensor(self.test_step_output["pred"])

        accuracy = torch.mean((gt == pred).float())
        self.log("accuracy", accuracy * 100)
        self.log("mae", F.l1_loss(gt.float(), pred.float()))

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

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0005)
