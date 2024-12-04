from core.Network import CountModel_reg, early_stop_callback, CountModel_cls, Backbone
from core.data import AppleDataModule, Modality

import lightning as L

for modality in Modality:
    print("#"*50)
    print(modality)
    print("#"*50)


    dm = AppleDataModule(dataset="apple_minneapple", modality=modality)
    model = CountModel_reg()
    trainer = L.Trainer(max_epochs=20, log_every_n_steps=1)
    trainer.fit(model, dm)
    trainer.test(model, dataloaders=dm.train_dataloader())
    trainer.test(model, datamodule=dm)


for modality in Modality:
    print("#"*50)
    print(modality)
    print("#"*50)

    dm = AppleDataModule(dataset="apple_minneapple", modality=modality)
    model = CountModel_cls()
    trainer = L.Trainer(max_epochs=20, log_every_n_steps=1)
    trainer.fit(model, dm)
    trainer.test(model, dataloaders=dm.train_dataloader())
    trainer.test(model, datamodule=dm)