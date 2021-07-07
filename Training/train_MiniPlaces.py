import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as sio
import copy
import pandas as pd
import os
from PIL import Image
import time
import sys
import math

# Example Line of Code:
#python train_MiniPlaces.py AlexNet Reference-Net 1 1 2 0 120 2
#python train_MiniPlaces.py ResNet18 Reference-Net 1 2 3 0 80 2


Model_Type = sys.argv[1]
Metamer_Type = sys.argv[2]
top_k_num = int(sys.argv[3])
first_noise = int(sys.argv[4])
last_noise = int(sys.argv[5])
first_epoch = int(sys.argv[6])
last_epoch = int(sys.argv[7])
num_workers = int(sys.argv[8])


# Hyper-parameters:
device = 'cuda:0'
num_classes = 20
preTrained_Flag = False
top_k_num_str = str(top_k_num)

num_epochs = last_epoch
first_epoch = first_epoch

if Model_Type == 'ResNet18':
	learning_rate = 0.05 # 0.05
	lr_decay_factor = 0.25 # 0.5
	first_stop = 0.2
	second_stop = 0.4
elif Model_Type == 'AlexNet':
	learning_rate = 0.01
	lr_decay_factor = 0.5
	first_stop = 0.25
	second_stop = 0.5
elif Model_Type == 'VGG11':
	learning_rate =0.001

# Color Normalization:
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Training + Testing Augmentation Transforms:
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), normalize])
transform_test = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), normalize])

# Training + Validation batch size:
training_batch_size = 128
validation_batch_size = 128

class sceneTrainDataset(torch.utils.data.Dataset):
    def __init__(self,text_file,root_dir,transform=transform):
        self.name_frame = pd.read_csv(text_file,header=None,sep=" ",usecols=range(1))
        self.label_frame = pd.read_csv(text_file,header=None,sep=" ",usecols=range(1,2))
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.name_frame)
    def __getitem__(self,idx):
        img_name = self.name_frame.iloc[idx,0]
        image = Image.open(img_name)
        image = self.transform(image)
        labels = int(self.label_frame.iloc[idx,0])-1
        return image, labels

class sceneValDataset(torch.utils.data.Dataset):
    def __init__(self,text_file,root_dir,transform=transform):
        self.name_frame = pd.read_csv(text_file,header=None,sep=" ",usecols=range(1))
        self.label_frame = pd.read_csv(text_file,header=None,sep=" ",usecols=range(1,2))
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.name_frame)
    def __getitem__(self,idx):
        img_name = self.name_frame.iloc[idx,0]
        image = Image.open(img_name)
        image = self.transform(image)
        labels = int(self.label_frame.iloc[idx,0])-1
        return image, labels


scene_classes = ('aquarium','badlands','bedroom','bridge','campus','corridor','forest_path','highway','hospital','industrial_area','japanese_garden','kitchen','mansion','mountain','ocean','office','restaurant','skyscraper','train_interior','waterfall')

# Load Training + Validation Dataset:
trainset_name = '../Dataset_Files/Training/Mini_Places_' + Metamer_Type + '.txt'
valset_name = '../Dataset_Files/Validation/Mini_Places_' + Metamer_Type + '.txt'

transform_train = transform

trainset = sceneTrainDataset(text_file = trainset_name, root_dir='.', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=training_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

valset = sceneValDataset(text_file = valset_name, root_dir='.', transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=validation_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(testing_batch_size)))

#net_load_string = './' + Metamer_Type + '/Model/5.pth'
#net.load_state_dict(torch.load(net_load_string))

for rounds in range(first_noise,last_noise+1):

    # Get Noise Round we are in:
    rounds_str = str(rounds)

    # Load Model Type:	
    if Model_Type == 'VGG11':
        vgg11 = models.vgg11(pretrained=preTrained_Flag)
        net = vgg11
        net.classifier[-1] = nn.Linear(4096, num_classes)
    elif Model_Type == 'AlexNet':
        alexnet = models.alexnet(pretrained=preTrained_Flag)
        net = alexnet
        net.classifier[-1] = nn.Linear(4096, num_classes)
         
    elif Model_Type == 'ResNet18':
        resnet18 = models.resnet18(pretrained=preTrained_Flag)
        net = resnet18
        net.fc = nn.Linear(512, num_classes)
    else:
        print('Nothing')

    net.to(device)

    # Define Criterion
    criterion = nn.CrossEntropyLoss()

    #########################
    # FineTune the Network: #
    #########################

    running_loss = 0.0
    running_loss_val = 0.0

    for epoch in range(first_epoch,num_epochs):  # loop over the dataset multiple times
        epoch_str = str(epoch)

        if epoch == 0:
            # Re-Load the Reference-Net common noise seed!
            if Metamer_Type != 'Reference-Net':
                net_load_string = '../All_Networks/Reference-Net/Model/' + Model_Type + '/' + rounds_str + '/' + epoch_str +'.pth'
                net.load_state_dict(torch.load(net_load_string))
            save_str = '../All_Networks/' + Metamer_Type + '/Model/' + Model_Type + '/' + rounds_str + '/0.pth'
            torch.save(net.state_dict(), save_str)

        epoch_plus_str = str(epoch+1)

        ########################
        # Set in Training Mode #
        ########################

        net.train()
        torch.set_grad_enabled(True)

        # Only one type of Optimizer:
        if epoch<math.ceil(num_epochs*first_stop):
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005, dampening=0, nesterov=True)
            print(learning_rate)
        elif epoch<math.ceil(num_epochs*second_stop):
            optimizer = optim.SGD(net.parameters(), lr=learning_rate*lr_decay_factor, momentum=0.9, weight_decay=0.0005, dampening=0, nesterov=True)
            print(learning_rate*lr_decay_factor)
        else:
            optimizer = optim.SGD(net.parameters(), lr=learning_rate*lr_decay_factor*lr_decay_factor, momentum=0.9, weight_decay=0.0005, dampening=0, nesterov=True)
            print(learning_rate*lr_decay_factor*lr_decay_factor)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            i_str = str(i)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch, i, loss.item()))
        print('[%d] Total Training Loss: %.3f' % (epoch, running_loss))
        loss_str = '../All_Networks/' + Metamer_Type + '/Training_Loss/' + Model_Type + '/' + rounds_str + '/' + epoch_plus_str + '.npy'
        np.save(loss_str,np.array(running_loss))

        #######################################
        # Main Network File to save the Model #
        #######################################
        #
        save_str = '../All_Networks/' + Metamer_Type + '/Model/' + Model_Type + '/' + rounds_str + '/' + epoch_plus_str + '.pth'
        torch.save(net.state_dict(), save_str)
        #
        ##########################
        # Set in Validation Mode #
        ##########################
        #
        net.eval()
        torch.set_grad_enabled(False)
        running_loss_val = 0.0
        #
        class_correct_Object = list(0. for i in range(len(scene_classes)))
        class_total_Object = list(0. for i in range(len(scene_classes)))
        for i, data in enumerate(valloader,0):
            #get the inputs from te validation set
            i_str = str(i)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            #
            # Save Validation Loss
            val_loss = criterion(outputs, labels)
            running_loss_val += val_loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch, i, val_loss.item()))
            #
            # Compute Top-k Validation performance:
            _, predicted_k = torch.topk(outputs,top_k_num,1)
            c = (predicted_k == labels.repeat(top_k_num, 1).t())
            c = torch.sum(c,1)
            #
            for m in range(len(labels)):
		#for i in range(testing_batch_size):
                label = labels[m]
                if len(labels)>1:
                    class_correct_Object[label] += c[m].item()
                    class_total_Object[label] += 1
                else:
                    class_correct_Object[label] += c.item()
                    class_total_Object[label] += 1
            del data,labels,inputs,outputs
        #
        loss_str_val = '../All_Networks/' + Metamer_Type + '/Validation_Loss/' + Model_Type + '/' + rounds_str + '/' + epoch_plus_str + '.npy'
        np.save(loss_str_val,np.array(running_loss_val))
        print('[%d] Total Validation Loss: %.3f' % (epoch, running_loss_val))
        #
        # Display all accuracies
        for i in range(len(scene_classes)):
            print('Validation Accuracy of %5s: %2d %%' % (scene_classes[i], 100.0 * float(class_correct_Object[i]) / float(class_total_Object[i])))
        print('############################################## %%')
        print('Total Validation Accuracy: %5d %%' % np.mean(100.0 * float(sum(class_correct_Object)) / float(sum(class_total_Object))))
        print('############################################## %%')
        #
        # Save Per Class accuracies somewhere
        class_correct_Object_str = '../All_Networks/' + Metamer_Type + '/Accuracy/Top' + top_k_num_str +  '/' + Model_Type + '/' + rounds_str + '/' + epoch_plus_str + '_class_correct_Scene' + '.npy'
        class_total_Object_str = '../All_Networks/' + Metamer_Type + '/Accuracy/Top' + top_k_num_str + '/' + Model_Type + '/' + rounds_str + '/' + epoch_plus_str +'_class_total_Scene' + '.npy'
        #
        np.save(class_correct_Object_str,class_correct_Object)
        np.save(class_total_Object_str,class_total_Object)
        #
        # 10 second pause:
        #time.sleep(2)
    print('Finished Training')

