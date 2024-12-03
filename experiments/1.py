import os
from core.Network import CountModel_reg, early_stop_callback, CountModel_cls, Backbone
from core.data import AppleDataModule

import lightning as L

def experiment_(dataset,bacbone,modeling):

    trainer = L.Trainer(max_epochs=200,log_every_n_steps=1)

    if(modeling == 'regression'):
        model = CountModel_reg(backbone=bacbone)
    else:
        model = CountModel_cls(backbone=bacbone)

    dm = AppleDataModule(dataset)

    trainer.fit(model, dm)
    return trainer.test(model, dm)

def experiment(dataset,backbone,modeling):
    T = 5

    os.makedirs('results',exist_ok=True)


    #save 5 results for each experiment in a file

    file = f'results/{dataset}_{backbone}_{modeling}.txt'

    #if file exists return
    if(os.path.exists(file)):
        return

    with open(file,'w') as f:
        f.write('loss\taccuracy\n')

    for t in range(T):
        results = experiment_(dataset,backbone,modeling)
        with open(file,'a') as f:
            f.write(f'{results[0]}\t{results[1]}\n')


datasets = ['apple_roboflow','apple_minneapple']
backbones = [Backbone.RESNET18,Backbone.RESNET50,Backbone.RESNET152,Backbone.VIT]
modelings = ['regression','classification']
modalities = ['rgb','depth','rgbd']

# take first option as baseline
for dataset in datasets:
    experiment(dataset,Backbone.RESNET18,'regression')

for backbone in backbones:
    experiment('apple_roboflow',backbone,'regression')

for modeling in modelings:
    experiment('apple_roboflow',Backbone.RESNET18,modeling)

for modality in modalities:
    experiment('apple_roboflow',Backbone.RESNET18,'regression')
