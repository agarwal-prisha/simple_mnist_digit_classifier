import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from model import Net

# ==============================
# DEVICE (GPU/CPU)
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# LOAD IDX FILES
# ==============================

def load_images(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data[16:].reshape(-1, 28, 28)

def load_labels(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data[8:]

train_data = load_images("train-images.idx3-ubyte")
train_targets = load_labels("train-labels.idx1-ubyte")

test_data = load_images("t10k-images.idx3-ubyte")
test_targets = load_labels("t10k-labels.idx1-ubyte")

# ==============================
# HYPERPARAMETERS
# ==============================

n_epochs = 5
batch_size = 64
learning_rate = 0.001

# ==============================
# PREPROCESSING
# ==============================

train_data = np.expand_dims(train_data, axis=1) / 255.0
test_data = np.expand_dims(test_data, axis=1) / 255.0

train_batches = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]
train_target_batches = [train_targets[i:i+batch_size] for i in range(0, len(train_targets), batch_size)]

test_batches = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]
test_target_batches = [test_targets[i:i+batch_size] for i in range(0, len(test_targets), batch_size)]

# ==============================
# MODEL
# ==============================

network = Net().to(device)

summary(network, (1, 28, 28), device=str(device))

optimizer = optim.Adam(network.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# ==============================
# TRAIN
# ==============================

def train(epoch):
    network.train()
    loss_sum = 0

    pbar = tqdm(zip(train_batches, train_target_batches), total=len(train_batches))

    for index, (data, target) in enumerate(pbar, start=1):

        data = torch.from_numpy(data).float().to(device)
        target = torch.from_numpy(target).long().to(device)

        optimizer.zero_grad()

        output = network(data)
        loss = loss_function(output, target)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        pbar.set_description(f"Epoch {epoch}, loss: {loss_sum/index:.4f}")

# ==============================
# TEST
# ==============================

def test(epoch):
    network.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(zip(test_batches, test_target_batches), total=len(test_batches))

        for data, target in pbar:

            data = torch.from_numpy(data).float().to(device)
            target = torch.from_numpy(target).long().to(device)

            output = network(data)
            pred = output.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.size(0)

            pbar.set_description(f"Accuracy: {correct/total:.4f}")

# ==============================
# RUN
# ==============================

for epoch in range(1, n_epochs+1):
    train(epoch)
    test(epoch)

# ==============================
# SAVE
# ==============================

os.makedirs("Models", exist_ok=True)
torch.save(network.state_dict(), "Models/model.pt")

print("Model Saved")