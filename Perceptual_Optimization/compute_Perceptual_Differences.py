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
from scipy.stats import ks_2samp
from scipy.stats import ttest_rel
import copy
import pandas as pd
import os
from PIL import Image
import time
import sys
import math
from IQA_pytorch import SSIM, GMSD, FSIM, VSI, LPIPSvgg, DISTS, MS_SSIM, CW_SSIM, NLPD, MAD

device = 'cuda:0'
#device = 'cpu'
num_classes = 20
num_workers = 2

transform_PO = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

# Batch Size:
#batch_size = 64
batch_size = 16

class sceneTestDataset(torch.utils.data.Dataset):
    def __init__(self,text_file,root_dir,transform=transform_PO):
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

# Load Validation Dataset:
reference_name = '../Dataset_Files/Testing/Mini_Places_Reference-Net.txt'
refset = sceneTestDataset(text_file = reference_name, root_dir='.', transform=transform_PO)
refloader = torch.utils.data.DataLoader(refset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

transform_name = []

transform_name.append('../Dataset_Files/Testing/Mini_Places_Reference-Net.txt')
transform_name.append('../Dataset_Files/Testing/Mini_Places_Foveation-Texture-Net.txt')
transform_name.append('../Dataset_Files/Testing/Mini_Places_Uniform-Net.txt')
transform_name.append('../Dataset_Files/Testing/Mini_Places_Foveation-Blur-Net.txt')

Name = []

Name.append('Reference')
Name.append('Foveation-Texture')
Name.append('Uniform-Blur')
Name.append('Foveation-Blur')

dataiter = []

for i in range(len(transform_name)):
	transformset = sceneTestDataset(text_file = transform_name[i], root_dir='.', transform=transform_PO)
	transformloader = torch.utils.data.DataLoader(transformset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	dataiter.append(iter(transformloader))	

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

D = SSIM(channels=3)
#D = MS_SSIM(channels = 3)
#D = CW_SSIM(channels = 3)
#D = FSIM(channels=3)
#D = VSI(channels=3)
#D = GMSD(channels=3)
#D = NLPD(channels=3)
#D = MAD(channels=3)
D.to(device)

score = torch.empty(batch_size)
total_score = torch.zeros(4)
all_scores = []
for j in range(len(transform_name)):
	all_scores.append(torch.empty(0,batch_size).to(device))
	all_scores[j].to(device)
	print(all_scores[j])

score.to(device)
total_score.to(device)

for i, data in enumerate(refloader,0):
	inputs, labels = data
	inputs, labels = inputs.to(device), labels.to(device)
	print(i)
	for j in range(len(transform_name)):
		inputs_temp,labels_temp =  dataiter[j].next()
		inputs_temp, labels_temp = inputs_temp.to(device), labels_temp.to(device)
		score = D(inputs,inputs_temp,as_loss=False)
		score.to(device)
		total_score[j] += torch.sum(score)
		print(score)
		if i != 0:
			all_scores[j] = torch.cat((all_scores[j],score),0)
		else:		
			all_scores[j] = score.clone()
		print(score)
		print(total_score)
		print(all_scores[j])
		print(len(all_scores[j]))
		print('-------------------------')
		print('Score %s : %f' % (Name[j], torch.sum(score)))
		print('Total_Score %s : %f' % (Name[j], torch.sum(total_score)))

#stat1, p1 = ks_2samp(all_scores[1].cpu().detach().numpy(),all_scores[2].cpu().detach().numpy(),alternative='two-sided')
#stat2, p2 = ks_2samp(all_scores[1].cpu().detach().numpy(),all_scores[3].cpu().detach().numpy(),alternative='two-sided')

stat1, p1 = ttest_rel(all_scores[1].cpu().detach().numpy(),all_scores[2].cpu().detach().numpy())
stat2, p2 = ttest_rel(all_scores[1].cpu().detach().numpy(),all_scores[3].cpu().detach().numpy())


print('#########################')
for j in range(len(transform_name)):
	print('Total_Score %s : %f' % (Name[j], total_score[j]/5000.0))
	print('All Scores %s : %f +/- %f' % (Name[j], torch.mean(all_scores[j],0), torch.std(all_scores[j],0)))

print('Fov-Texture vs Uniform-Blur | KS-Test stat: %f, p-value: %f' % (stat1,p1))
print('Fov-Texture vs Foveation-Blur | KS-Test stat: %f, p-value: %f' % (stat2,p2))


