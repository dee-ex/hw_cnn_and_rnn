import torch
from torch import nn

import numpy as np

class RNN(nn.Module):
  def __init__(self, inp_size, out_size, hidden_dim, n_layers):
    super(RNN, self).__init__()

    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    self.rnn = nn.RNN(inp_size, hidden_dim, n_layers, batch_first=True)

    self.fc = nn.Linear(hidden_dim, out_size)

  def forward(self, x):
    batch_size = x.shape[0]

    hidden = self.init_hidden(batch_size)

    out, hidden = self.rnn(x, hidden)

    out = out.contiguous().view(-1, self.hidden_dim)
    out = self.fc(out)

    return out, hidden

  def init_hidden(self, batch_size):
    hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

    return hidden

def predict(model, word):
  chars = [[char2int[chr] for chr in word]]
  onehot = one_hot_encoder(chars, dict_size, len(chars[0]), 1)

  out, hidden = model(onehot)

  prob = nn.functional.softmax(out[-1], dim=0).data

  char_idx = torch.max(prob, dim=0)[1].item()

  return int2char[char_idx], hidden

def sample(model, out_len, start):
  model.eval()

  start = start.lower()
  chars = [chr for chr in start]
  size = out_len - len(chars)

  for _ in range(size):
    chr, h = predict(model, chars)
    chars.append(chr)

  return ''.join(chars)
words = ["hello", "green", "start"]

chars = set("".join(words))

int2char = dict(enumerate(chars))

char2int = {chr: idx for idx, chr in int2char.items()}

inp, tar = [], []

for word in words:
  inp.append(word[:-1])
  tar.append(word[1:])

N = len(words)

for i in range(N):
  inp[i] = [char2int[chr] for chr in inp[i]]
  tar[i] = [char2int[chr] for chr in tar[i]]

dict_size = len(char2int)
seq_len =  len(words[0]) - 1
batch_size = len(words)

def one_hot_encoder(seq, dict_size, seq_len, batch_size):
  features = torch.zeros((batch_size, seq_len, dict_size))

  for i in range(batch_size):
    for j in range(seq_len):
      features[i, j, seq[i][j]] = 1

  return features

inp_1hot = one_hot_encoder(inp, dict_size, seq_len, batch_size)

rnn = RNN(dict_size, dict_size, 12, 1)

n_epochs = 100
lr = .01

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr)
target = torch.Tensor(tar)

for epoch in range(1, n_epochs + 1):
  optimizer.zero_grad()

  out, hidden = rnn(inp_1hot)
  loss = criterion(out, target.view(-1).long())
  loss.backward()
  optimizer.step()

  if not epoch % 10:
    print("Epoch:", epoch, "..... Loss:", loss.item())