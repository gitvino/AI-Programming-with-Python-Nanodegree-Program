import torch 
from torchvision import datasets, transforms, models 
import os
from datautils import *
from collections import OrderedDict
from torch import nn
from torch import optim
from PIL import Image 
import numpy as np
from torch.autograd import Variable
import json
import argparse


def load_checkpoint(filepath):
    if device == 'cuda':
        checkpoint = torch.load(filepath)    
    else:
        checkpoint = torch.load(filepath,map_location=lambda storage, loc: storage)
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

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    

    # Get original dimensions
    orig_width, orig_height = image.size

    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    image.thumbnail(size=resize_size)

    # Find pixels to crop on to create 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    image = image.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(image)/255 # Divided by 255 because imshow() expects integers (0:1)!!

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    return np_image
   

def predict(checkpoint_file, input_image, topK):
    model, class_to_idx = load_checkpoint(checkpoint_file)
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    image = Image.open(input_image)
    image = torch.FloatTensor([process_image(image)])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]
    top_idx = np.argsort(pobabilities)[-topK:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]
    return top_probability, top_class

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_file", help = "Input PyTorch Check point file", type = str)
parser.add_argument("image_file", help = "Input image to classify", type = str)
parser.add_argument("class_names_json", help = "Prediction class names JSON", type = str)
parser.add_argument("device", help = "CPU or GPU", type = str)
parser.add_argument("topK", help = "topK", type = int)

args = parser.parse_args()
checkpoint_file = args.checkpoint_file
input_image_file = args.image_file
class_names_json = args.class_names_json
topK= args.topK
selected_device = args.device

if (selected_device.lower() == 'gpu') and (torch.cuda.is_available()): 
    device = 'cuda'
else: 
    device = 'cpu'
with open(class_names_json, 'r') as f:
        cat_to_name = json.load(f)
top_probability, top_class = predict(checkpoint_file, input_image_file, topK)
for p,c in zip(top_probability, top_class): 
    print("{0} : {1}".format(cat_to_name[c], p))
    