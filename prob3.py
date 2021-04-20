from torchvision import transforms
from torchvision import models
import torch

from PIL import Image
import os

filenames = os.listdir("./images")
labels = [fname.split(".")[0] for fname in filenames]

imgs = [Image.open("./images/" + fname) for fname in filenames]

transformer = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      mean=[.485, .456, .406],
                                      std=[.229, .224, .225]
                                  )])
        
transformed_imgs = torch.empty(0, 3, 224, 224)

for img in imgs:
  tf_img = transformer(img)
  tensor_tf_img = torch.unsqueeze(tf_img, 0)
  transformed_imgs = torch.cat((transformed_imgs, tensor_tf_img), dim=0)

googlenet = models.googlenet(pretrained=True)

googlenet.eval()

outputs = googlenet(transformed_imgs)
softout = torch.nn.functional.softmax(outputs, dim=-1)

with open("imagenet_classes.txt") as f:
  classes = [line.strip() for line in f.readlines()]

top5_softmax, top5_idx = torch.topk(softout, 5, dim=-1)

count_top1 = 0
count_top5 = 0
N = top5_idx.shape[0]

for i in range(N):
  for j in range(5):
    if classes[top5_idx[i, j]].startswith(labels[i]):
      count_top5 += 1
      break

  count_top1 += 1 if classes[top5_idx[i, 0]].startswith(labels[i]) else 0

print("top-1 error rate", 100 * (1 - count_top1/N))
print("top-5 error rate", 100 * (1 - count_top5/N))