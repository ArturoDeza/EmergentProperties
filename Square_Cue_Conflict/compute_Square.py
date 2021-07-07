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
import gc

# Example line:
#python compute_Robustness.py ResNet18_correct_lr Center-Fov-Net 3 1 3 Color 1
#python compute_Window.py AlexNet Reference-Net 1 1 1 50

#Model_Type = 'AlexNet'
#Metamer_Type = 'Standard-Net'
#top_k_num = 1
#start_run_id = 1
#end_run_id = 10
#epoch = 270

###############
# Parameters: #
###############
# Uncomment these lines when executing the function!

Model_Type = sys.argv[1]
Metamer_Type = sys.argv[2]
top_k_num = int(sys.argv[3])
start_run_id = int(sys.argv[4])
end_run_id = int(sys.argv[5])
epoch = int(sys.argv[6])

top_k_num_str = str(top_k_num)

Folder_Inputs = Metamer_Type

device = 'cuda:0'
epoch_str_save = str(epoch)

epoch = epoch

epoch_str = str(epoch)
preTrained_Flag = True

##################################################
# Load Networks and run through testing dataset: #
##################################################

scene_classes = ('aquarium','badlands','bedroom','bridge','campus','corridor','forest_path','highway','hospital','industrial_area','japanese_garden','kitchen','mansion','mountain','ocean','office','restaurant','skyscraper','train_interior','waterfall')

num_classes = len(scene_classes)

#####################
# Image DataLoaders #
#####################

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_general = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),normalize])

training_batch_size = 16
testing_batch_size = 16

###################################################################################
# Now do a loop to get the performances to test the robustness to all distortions #
###################################################################################

class TotalScenes20(torch.utils.data.Dataset):
	def __init__(self,text_file,root_dir,transform=transform_general):
		self.name_frame = pd.read_csv(text_file,header=None,sep=" ",usecols=range(1))
		self.label_frame = pd.read_csv(text_file,header=None,sep=" ",usecols=range(1,2))
		self.root_dir = root_dir
		self.transform = transform
	def __len__(self):
		return len(self.name_frame)
	def __getitem__(self,idx):
		img_name = os.path.join(self.root_dir,self.name_frame.iloc[idx,0])
		image = Image.open(img_name).convert('RGB')
		image = self.transform(image)
		labels = int(self.label_frame.iloc[idx,0])-1
		return image, labels

Distortion_Family = ['Square_Uniform_Conflict']

for run_id in range(start_run_id,end_run_id+1):
	run_id_str = str(run_id)
	#
	if Model_Type == 'AlexNet':
		net = models.alexnet(pretrained=preTrained_Flag)
		net.classifier[-1] = nn.Linear(4096, num_classes)
	elif Model_Type == 'ResNet18':
		net = models.resnet18(pretrained=preTrained_Flag)
		net.fc = nn.Linear(512, num_classes)
	else:
		print('Nothing')
	#
	net.to(device)
	net_load_string = '../All_Networks/' + Metamer_Type + '/Model/' + Model_Type + '/' + run_id_str + '/' + epoch_str + '.pth'
	net.load_state_dict(torch.load(net_load_string))
	net.eval()
	#
	for m in range(len(Distortion_Family)):
		m_str = str(m)
		Distortion_Type = Distortion_Family[m]
		for z in range(17):
			##########################################
			# Load dataset for each distortion level #
			##########################################
			z_str = str(z+1)
			input_file = './Data_Loader/' + Folder_Inputs +'/' + Distortion_Type + '_Fovea_' + z_str + '.txt'
			testset = TotalScenes20(text_file = input_file, root_dir='.', transform=transform_general)
			testloader = torch.utils.data.DataLoader(testset,batch_size=testing_batch_size,shuffle=False,num_workers=1,pin_memory=True)
			input_file_Periphery = './Data_Loader/' + Folder_Inputs +'/' + Distortion_Type + '_Periphery_' + z_str + '.txt'
			testset_Periphery = TotalScenes20(text_file = input_file_Periphery, root_dir='.', transform=transform_general)
			testloader_Periphery = torch.utils.data.DataLoader(testset_Periphery,batch_size=testing_batch_size,shuffle=False,num_workers=1,pin_memory=True)
			dataiter_Periphery = iter(testloader_Periphery)
			class_periphery_Object = list(0. for i in range(len(scene_classes)))
			class_fovea_Object = list(0. for i in range(len(scene_classes)))
			class_total_Object = list(0. for i in range(len(scene_classes)))
			with torch.no_grad():
				for data in testloader:
					images, labels = data
					images, labels = images.to(device), labels.to(device)
					outputs = net(images)
					images_Periphery, labels_Periphery = dataiter_Periphery.next()
					images_Periphery, labels_Periphery = images_Periphery.to(device), labels_Periphery.to(device)
					# Compute Peripheral Bias  (Correct Class) # Re-Check this!
					_, predicted_k = torch.topk(outputs,top_k_num,1)
					c = (predicted_k == labels.repeat(top_k_num, 1).t())
					c = torch.sum(c,1)
					# Get Faux (Foveal) Labels # Re-Check this!
					labels_faux = labels_Periphery
					c_faux = (predicted_k == labels_faux.repeat(top_k_num,1).t())
					c_faux = torch.sum(c_faux,1)
					for i in range(len(labels)):
					#for i in range(testing_batch_size):
						label = labels[i]
						if len(labels)>1:
							class_periphery_Object[label] += c_faux[i].item()
							class_fovea_Object[label] += c[i].item()
							class_total_Object[label] += 1
						else:
							class_periphery_Object[label] += c_faux.item()
							class_fovea_Object[label] += c.item()
							class_total_Object[label] += 1
			del data,labels,images,outputs
			# Display all accuracies
			for i in range(len(scene_classes)):
				print('Peripheral Bias Accuracy %5s: %2d %%' % (scene_classes[i], 100.0 * float(class_periphery_Object[i] / class_total_Object[i])))
				print('Fovea Bias Accuracy %5s: %2d %%' % (scene_classes[i], 100.0 * float(class_fovea_Object[i] / class_total_Object[i])))
			# Save Per Class accuracies somewhere
			class_periphery_Object_str = './Results/Top' + top_k_num_str + '/' + Metamer_Type + '/' + Model_Type + '_' + run_id_str + '_class_periphery_Object_' + z_str + '_Epoch_' + epoch_str_save +'.npy'
			class_fovea_Object_str = './Results/Top' + top_k_num_str + '/' + Metamer_Type + '/' + Model_Type + '_' + run_id_str + '_class_fovea_Object_' + z_str + '_Epoch_' + epoch_str_save +'.npy'
			class_total_Object_str = './Results/Top' + top_k_num_str + '/' + Metamer_Type + '/' + Model_Type + '_' + run_id_str + '_class_total_Object_' + z_str + '_Epoch_' + epoch_str_save + '.npy'
			np.save(class_periphery_Object_str,class_periphery_Object)
			np.save(class_fovea_Object_str,class_fovea_Object)
			np.save(class_total_Object_str,class_total_Object)

