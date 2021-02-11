import json 
import numpy as np

import torch
from PIL import Image
from collections import OrderedDict

from torchvision import datasets, transforms, models
from torch import nn, optim

import time


def create_network(arch='vgg13', hidden_units=512, pretrained=True):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=pretrained)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    return model

def load_device(use_gpu=False):
    if use_gpu:
        if torch.cuda.is_available():
            print('Using GPU')
            device = torch.device('cuda:0')
        else:
            print('GPU not available - falling back to CPU')
            device = torch.device('cpu')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    return device

def process_image(img_path):
    w, h = img_path.size
    if w > h:
        img_path = img_path.resize((int(round(w * 256/h)), 256), Image.ANTIALIAS)
    else:
        img_path = img_path.resize((256, int(round(h * 256/w))), Image.ANTIALIAS)
    w, h = img_path.size
    left = (w - 224) / 2
    top = (h - 224) / 2
    right = (w + 224) / 2
    bottom = (h + 224) / 2
    img_path = img_path.crop((left, top, right, bottom))
    np_image = np.array(img_path, dtype=np.float32) / 255
    np_image -= np.array ([0.485, 0.456, 0.406]) 
    np_image /= np.array ([0.229, 0.224, 0.225])
    np_image = np_image.transpose((2,0,1))
    return np_image

def predict(np_image, model, topk=3, use_gpu=False):
    print('Setting device')
    device = load_device(use_gpu)
    model.to(device=device)
    
    print('Processing image')
    image = torch.from_numpy(np_image).to(device=device)
    image.unsqueeze_(dim = 0) 
    model.eval()
    outputs = model.forward(image)
    
    print('Reset to training mode')
    model.train()
    
    print('Getting probabilities')
    probabilities = torch.exp(outputs)
    probs, labels = probabilities.topk(topk)
    probs = probs.tolist()[0]
    top_labels_idx = labels.tolist()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[idx] for idx in top_labels_idx]

    return probs[:topk], top_labels[:topk]

def load_cat_to_name(filepath):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(filepath, use_gpu=False):
    device = load_device(use_gpu)
    torch.device(device)
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = create_network(checkpoint['arch'], checkpoint['hidden_units'], False)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device=device)
    return model

def open_image_path(image_path):
    img = Image.open(image_path)
    print('Processing image: ' + str(image_path))
    return img


def load_train_data(train_dir):
    print('Loading training data from: {}'.format(train_dir))
    t = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(train_dir, transform=t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
    return dataset, dataloader

def load_valid_data(valid_dir):
    print('Loading validation data from: {}'.format(valid_dir))
    t = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset = datasets.ImageFolder(valid_dir, transform=t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
    return dataset, dataloader

def load_test_data(test_dir):
    print('Loading testing data from: {}'.format(test_dir))
    t = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(test_dir, transform=t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
    return dataset, dataloader


def create_criterion():
    print('Creating criterion NLLLoss')
    return nn.NLLLoss()

def create_optimizer(model, lr):
    print('Creating optimizer with LR {}'.format(lr))
    return optim.Adam(model.classifier.parameters(), lr=lr)

def train_model(model, criterion, optimizer, train_dataloader, valid_dataloader, epochs, use_gpu=True):
    device = load_device(use_gpu)
    model.to(device=device)
    print_every = 32
    print('Training for {} epochs, updating every {} training images'.format(epochs, print_every))
    for e in range(epochs):
        start = time.time()
        running_loss = 0
        for train_step, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if train_step % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for valid_step, (images, labels) in enumerate(valid_dataloader):
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        validation_loss += criterion(outputs, labels)
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                print('Epoch: {}/{}.. '.format(e+1, epochs),
                      'Epoch time: {:.2f} sec.. '.format(time.time() - start),
                      'Training Loss: {:.3f}.. '.format(running_loss/len(train_dataloader)),
                      'Validation Loss: {:.3f}.. '.format(validation_loss/len(valid_dataloader)),
                      'Accuracy: {:.3f}'.format(accuracy/len(valid_dataloader)))
                model.train()

    return model

def save_checkpoint(model, arch, hidden_units, train_dataset, save_dir):
    model.class_to_idx = train_dataset.class_to_idx
    torch.save({
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'arch': arch,
        'hidden_units': hidden_units
    }, save_dir + '/model_' + arch + '_' + str(hidden_units) + '.pth')