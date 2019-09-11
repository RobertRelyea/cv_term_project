# -*- coding: utf-8 -*-
"""
Transfer Learning Tutorial
==========================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

In this tutorial, you will learn how to train your network using
transfer learning. You can read more about the transfer learning at `cs231n
notes <https://cs231n.github.io/transfer-learning/>`__

Quoting these notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios look as follows:

-  **Finetuning the convnet**: Instead of random initializaion, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

"""
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pdb

MODEL = 'densenet161'

freeze_indexes = {}
freeze_indexes['resnet18'] = 9
freeze_indexes['resnet152'] = 9
freeze_indexes['densenet161'] = 1

BATCH_SIZE = 256
NUM_CLASSES = 7
NUM_EPOCHS = 60
MODEL_PREFIX = MODEL + "_augmented"
path = './models/' + MODEL_PREFIX + '_{}_{}.pt'.format(freeze_indexes[MODEL], NUM_EPOCHS)

data_dir = '/home/imhs/Robert/computer_vision_2019/term_project/data'

# Define our model
if MODEL == 'resnet18':
    model_conv = torchvision.models.resnet18(pretrained=True)
elif MODEL == 'resnet152':
    model_conv = torchvision.models.resnet152(pretrained=True)
else:
    model_conv = torchvision.models.densenet161(pretrained=True)

# Freeze weights up to this layer (9 for FC fine tuning)
# Iterate through all children in the model
count = 0
for child in model_conv.children():
    # Freeze weights below the freeze index
    if count < freeze_indexes[MODEL]:
        for param in child.parameters():
            param.requires_grad = False
    # print(child)
    # print(count)
    count+=1

# Parameters of newly constructed modules have requires_grad=True by default
if MODEL == 'densenet161':
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, NUM_CLASSES, bias=True)
else:
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, NUM_CLASSES, bias=True)

plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=32)
              for x in ['train', 'val']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    stats = []

    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            stats.append([phase, epoch, epoch_loss, epoch_acc])

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, stats


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

print("Using device: {}".format(device))

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
if MODEL == 'densenet161':
    optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=0.001, momentum=0.9)
else:
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# On CPU this will take about half the time compared to previous scenario.
# This is expected as gradients don't need to be computed for most of the
# network. However, forward does need to be computed.
#
model_conv, stats = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, NUM_EPOCHS=NUM_EPOCHS)

######################################################################
#

torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': model_conv.state_dict(),
    'optimizer_state_dict': optimizer_conv.state_dict(),
    'loss': stats[-1:1]
},  path)


visualize_model(model_conv)

plt.ioff()
plt.show()

stats = np.array(stats)

stats_train = stats[stats[:,0]=='train']
epochs = stats_train[:,1]
losses_train = stats_train[:,2]
accs_train = [acc.item() for acc in stats_train[:,3]]

stats_val = stats[stats[:,0]=='val']
losses_val = stats_val[:,2]
accs_val = [acc.item() for acc in stats_val[:,3]]

train_line, = plt.plot(epochs, losses_train)
val_line, = plt.plot(epochs, losses_val)
plt.legend([train_line, val_line], ["Training Loss", "Validation Loss"])
plt.title("Loss statistics for {} Epoch{}".format(NUM_EPOCHS, 's' if (NUM_EPOCHS != 1) else ''))
plt.xlabel("Epoch")
plt.ylabel("Batch Loss")
plt.savefig("./figures/" + MODEL_PREFIX + "_{}_loss.png".format(freeze_indexes[MODEL]))
plt.show()

train_line, = plt.plot(epochs, accs_train)
val_line, = plt.plot(epochs, accs_val)
plt.legend([train_line, val_line], ["Training Accuracy", "Validation Accuracy"])
plt.title("Accuracy statistics for {} Epoch{}".format(NUM_EPOCHS, 's' if (NUM_EPOCHS != 1) else ''))
plt.xlabel("Epoch")
plt.ylabel("Batch Accuracy")
plt.savefig("./figures/" + MODEL_PREFIX + "_{}_acc.png".format(freeze_indexes[MODEL]))
plt.show()

print(MODEL_PREFIX)
print(class_names)
