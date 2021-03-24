import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pdb
import wandb
from PIL import Image

train_dir = "/media/public_dataset/ImageNet_200"
val_dir = "/media/public_dataset/ImageNet/imagenet/ILSVRC2012_img_val"
val_gt_dir = "/media/public_dataset/ImageNet/imagenet/val.txt"

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'epochs': {
            'values': [100]
        },
        'batch_size': {
            'values': [900]
        },
        'learning_rate': {
            'values': [1e-3]
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        },
    }
}
sweep_id = wandb.sweep(sweep_config, project="without self_sup testing")

config_defaults = {
        'epochs': 100,
        'batch_size': 900,
        'learning_rate': 1e-3,
        'optimizer': 'adam'
    }
wandb.init(config=config_defaults)
config = wandb.config

########################################
# Getting training set into dataloader #
########################################
print("Start loading training data!")
trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=preprocess)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
print("Finish loading training data!")

######################################################
# Getting validation set into dataloader as test set #
######################################################
print("Start loading validation data!") # get the image name which belongs to class 1-200
val_x = []
val_y = []
f = open(val_gt_dir, "r") 
for d in f:
    if int(d[29:]) <= 200:
        image_name = d[:28]
        path = os.path.join(val_dir, image_name)
        valid = Image.open(path)
        if len(np.array(valid).shape) != 3:
            valid = valid.convert("RGB")
        valid_tensor = preprocess(valid)
        val_x.append(np.array(valid_tensor))
        val_y.append(int(d[29:]))
f.close()

tensor_val_x, tensor_val_y = torch.Tensor(val_x), torch.Tensor(val_y)
validset = torch.utils.data.TensorDataset(tensor_val_x, tensor_val_y)
validloader = torch.utils.data.DataLoader(validset, batch_size=config.batch_size, shuffle=False)
print("Finish loading validation data!")

resnet50 = torchvision.models.resnet50(pretrained=False)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    resnet50 = nn.DataParallel(resnet50)

criterion = nn.CrossEntropyLoss()
if config.optimizer=='sgd':
    optimizer = optim.SGD(resnet50.parameters(),lr=config.learning_rate, momentum=0.9)
elif config.optimizer=='adam':
    optimizer = optim.Adam(resnet50.parameters(),lr=config.learning_rate, betas=(0.9, 0.999))

# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50.to(device)

print("Start training!")
for epoch in range(config.epochs):
    # training
    closs = 0
    total_acc = 0
    for i, data in enumerate(trainloader, 0):
        # get input and labels
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the gradient
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = resnet50(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item() # calculate accuracy
        pdb.set_trace()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # calculate the loss
        closs += loss.item()
        total_acc += correct
        wandb.log({"training batch loss":loss.item()})
        wandb.log({"training batch accuracy":correct/config.batch_size * 100})
    wandb.log({"training loss":closs/(1300 * 200 / config.batch_size)})
    wandb.log({"training accuracy":total_acc/(1300 * 200)})
    print('epoch %d loss: %.3f' % (epoch + 1, closs / 5200))

    # saving check point
    string1 = './checkpoint/resnet50_without_self_epoch'
    string2 = str(epoch + 1)
    string3 = '.pth'
    PATH = string1 + string2 + string3
    torch.save(resnet50.state_dict(), PATH)
    
    # validating
    closs = 0
    total_acc = 0
    for i, data in enumerate(validloader, 0):
        # get input and labels
        inputs, labels = data[0].to(device), data[1].to(device, dtype=torch.int64)
        # only forward
        outputs = resnet50(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item() # calculate accuracy
        loss = criterion(outputs, labels)
        # calculate the loss
        closs += loss.item()
        total_acc += correct
    wandb.log({"validating loss":closs/(10000 / config.batch_size)})
    wandb.log({"validating accuracy":total_acc/(10000)})

print("Finish training!")
# pdb.set_trace()