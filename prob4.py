import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt

import os
import time

from PIL import Image

transformer = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      mean=[.485, .456, .406],
                                      std=[.229, .224, .225]
                                  )])

batch_size = 8
num_classes = len(os.listdir(valid_directory))

data = {
  "train": datasets.ImageFolder("./datasets/train", transformer),
  "valid": datasets.ImageFolder("./datasets/validation", transformer)
}

idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}

train_size, valid_size = len(data["train"]), len(data["valid"])

train_loader = DataLoader(data["train"], batch_size, True)
valid_loader = DataLoader(data["valid"], batch_size, True)

alexnet = models.alexnet(pretrained=True)

for param in alexnet.parameters():
  param.requires_grad = False

alexnet.classifier[6] = nn.Linear(4096, num_classes)
alexnet.classifier.add_module("7", nn.LogSoftmax(dim = 1))

loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(alexnet.parameters())

def train_and_validate(model, loss_criterion, optimizer, epochs=25):
  start = time.time()
  history = []
  best_acc = .0

  for epoch in range(epochs):
    epoch_start = time.time()
    model.train()

    train_loss, train_acc = .0, .0
    valid_loss, valid_acc = .0, .0
    
    for i, (inputs, labels) in enumerate(train_data_loader):
      inputs = inputs
      labels = labels
      
      optimizer.zero_grad()
      
      outputs = model(inputs)
      
      loss = loss_criterion(outputs, labels)
      
      loss.backward()
      
      optimizer.step()
      
      train_loss += loss.item() * inputs.size(0)
      
      ret, predictions = torch.max(outputs.data, 1)
      correct_counts = predictions.eq(labels.data.view_as(predictions))
      
      acc = torch.mean(correct_counts.type(torch.FloatTensor))
      
      train_acc += acc.item() * inputs.size(0)
        
    with torch.no_grad():
      model.eval()

      for j, (inputs, labels) in enumerate(valid_data_loader):
        inputs = inputs
        labels = labels

        outputs = model(inputs)

        loss = loss_criterion(outputs, labels)

        valid_loss += loss.item() * inputs.size(0)

        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))

        acc = torch.mean(correct_counts.type(torch.FloatTensor))

        valid_acc += acc.item() * inputs.size(0)

        
    avg_train_loss = train_loss/train_data_size 
    avg_train_acc = train_acc/train_data_size

    avg_valid_loss = valid_loss/valid_data_size 
    avg_valid_acc = valid_acc/valid_data_size

    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
            
    epoch_end = time.time()

    print("Epoch : {}, trn_loss: {:.4f}, trn_acc: {:.4f}%, val_loss : {:.4f}, val_acc: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
    torch.save(model, "model_" + str(epoch) + ".pt")
      
  return model, history

num_epochs = 30
trained_model, history = train_and_validate(alexnet, loss_func, optimizer, num_epochs)

filenames = os.listdir('./datasets/test')
imgs = [Image.open('./datasets/test/' + fname) for fname in filenames]

transformed_imgs = torch.empty(0, 3, 224, 224)

for img in imgs:
  tf_img = transformer(img)
  tensor_tf_img = torch.unsqueeze(tf_img, 0)
  transformed_imgs = torch.cat((transformed_imgs, tensor_tf_img), dim=0)

outputs = trained_model(transformed_imgs)
softout = nn.functional.softmax(outputs, dim=-1)
top3_softmax, top3_idx = torch.topk(softout, 3, dim=-1)

count_top1 = 0
count_top3 = 0
N = top3_idx.shape[0]

for i in range(N):
  for j in range(3):
    if filenames[i].startswith(idx_to_class[top3_idx[i, j].item()]):
      count_top3 += 1
      break

  count_top1 += 1 if filenames[i].startswith(idx_to_class[top3_idx[i, 0].item()]) else 0

print(count_top1, count_top3)
print("top-1 error rate", 100 * (1 - count_top1/N))
print("top-3 error rate", 100 * (1 - count_top3/N))