from core.Network import CountModel_reg, early_stop_callback, CountModel_cls, Backbone
from core.data import AppleDataModule

import lightning as L

print("#"*50)
print("CountModel_reg","resnet18")
print("#"*50)

dm = AppleDataModule(dataset="apple_minneapple")
model = CountModel_reg()
trainer = L.Trainer(max_epochs=20, log_every_n_steps=1)
trainer.fit(model, dm)
trainer.test(model, dataloaders=dm.train_dataloader())
trainer.test(model, datamodule=dm)

print("#"*50)
print("CountModel_reg","resnet50")
print("#"*50)


dm = AppleDataModule(dataset="apple_minneapple")
model = CountModel_cls(backbone=Backbone.RESNET50)
trainer = L.Trainer(max_epochs=20, log_every_n_steps=1)
trainer.fit(model, dm)
trainer.test(model, dataloaders=dm.train_dataloader())
trainer.test(model, datamodule=dm)


print("#"*50)
print("CountModel_reg","resnet101")
print("#"*50)


dm = AppleDataModule(dataset="apple_minneapple")
model = CountModel_cls(backbone=Backbone.RESNET101)
trainer = L.Trainer(max_epochs=20, log_every_n_steps=1)
trainer.fit(model, dm)
trainer.test(model, dataloaders=dm.train_dataloader())
trainer.test(model, datamodule=dm)

