import torch

import models
import preprocessing as prep
import train


experiment_path = '../../experiments/digits/exp1/'
data_path = '../../data/zip.train'

# set device to cuda enabled device (GPU)
dtype = torch.float
device = torch.device("cuda:0")


# load data
digits = prep.Digits(data_path)
inputs, targets = digits.load_data()


# initialize model
model = models.Cnn('CNN', 16, int(targets.max() + 1), device=device)

# make sure model parameters are in GPU memory
model.cuda()


# run training
training = train.TrainClassifier(model, inputs, targets, experiment_path)
training.run_train(100)
