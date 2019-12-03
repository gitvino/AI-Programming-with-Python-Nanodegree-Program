import torch 
from torchvision import datasets, transforms, models 
import os
from datautils import *
from collections import OrderedDict
from torch import nn
from torch import optim
import time
import argparse

def prepare_data():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(train_dir, data_transforms)
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, data_transforms)

    dataloader = dict()
    dataloader['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle= True)
    dataloader['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle= True)
    return image_datasets, dataloader

def validation(model, validataionloader):
    test_loss = 0
    accuracy = 0
    for images, labels in validataionloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def train(model, dataloader, epochs, lr, device):
    model.to(device)
    model.train()
    for e in range(epochs):
        steps = 0
        for inputs, labels in dataloader['train']:
            train_loss = 0
            inputs, labels = inputs.to(device), labels.to(device)
            steps += 1
            start = time.time()
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if steps%print_every==0:
                model.eval()            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, dataloader['valid'])
                    train_loss += criterion(output, labels).item()
                    print(f"Epoch= {e}; Batch = {steps}; Train Loss = {train_loss/len(dataloader['train'])} Test Loss = {test_loss/len(dataloader['valid'])}; Accuracy = {accuracy/len(dataloader['valid'])}")
                    
def save_checkpoint():
    model.class_to_idx =dataloader["train"].dataset.class_to_idx
    model.epochs = epochs
    checkpoint = {'input_size': [3, 224, 224],
                 'batch_size': dataloader["train"].batch_size,
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': model.epochs}
    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16()
    num_features = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, 512)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('hidden', nn.Linear(512, 100)),                       
                              ('fc2', nn.Linear(100, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['class_to_idx']

def get_model(model_id):
    if model_id == 1: 
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[0].in_features #vgg
    elif model_id ==2: 
        model = models.densenet161(pretrained=True)
        num_features = model.classifier.in_features #densenet
#    print(model)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict( 
        [('fc1', nn.Linear(num_features, 512)),
        ('relu', nn.ReLU()),
        ('drpot', nn.Dropout(p=0.5)),
        ('hidden', nn.Linear(512, 100)),                       
        ('fc2', nn.Linear(100, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    return model

parser = argparse.ArgumentParser()
parser.add_argument("model", help = "Model: 1. vgg16, 2. DenseNet", type = int)
parser.add_argument("epochs", help = "Epochs", type = int)
parser.add_argument("lr", help = "Learning Rate", type = float)
parser.add_argument("device", help = "CPU or GPU", type = str)

args = parser.parse_args()
model_id = args.model
epochs = args.epochs
lr = args.lr
print_every = 40
image_datasets, dataloader = prepare_data()
criterion = nn.NLLLoss()
selected_device = args.device
if (selected_device.lower() == 'gpu') and (torch.cuda.is_available()): 
    device = 'cuda'
else: 
    device = 'cpu'
    
model = get_model(model_id)
optimizer = optim.Adam(model.classifier.parameters(),lr=lr)
train(model,dataloader, epochs, lr, device)
save_checkpoint()
#load_checkpoint('checkpoint.pth')