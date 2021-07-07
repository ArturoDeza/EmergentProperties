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
import seaborn as sn
from matplotlib import gridspec

#Model_Type = 'AlexNet'
#epoch = 120 

#Model_Type = 'ResNet18'
#epoch = 60
#epoch= 0
#epoch = 1
#epoch = 4
#epoch = 20

#Model_Type = sys.argv[1]
#epoch = int(sys.argv[2])
#Stimuli_test_str = sys.argv[3]

Model_Type = 'AlexNet'
epoch = 70
#Model_Type = 'ResNet18'
#epoch = 60
#Model_Type = 'VGG11'

start_run_id = 1
end_run_id = 10
top_k_num = 1


num_runs = end_run_id - start_run_id + 1
num_runs_str = str(num_runs)

epoch_str = str(epoch)

top_k_num_str = str(top_k_num)

scene_classes = ['aquarium','badlands','bedroom','bridge','campus','corridor','forest_path','highway','hospital','industrial_area','japanese_garden','kitchen','mansion','mountain','ocean','office','restaurant','skyscraper','train_interior','waterfall']

Metamer_Type_Family = ['Reference-Net','Foveation-Texture-Net','Uniform-Net','Foveation-Blur-Net']

All_Regimes = ['Reference-Net','Foveation-Texture-Net','Uniform-Net','Foveation-Blur-Net']
Color_Scheme = ['darkorange', 'navy', 'orchid','seagreen']
Paradigm_Names = ['Reference','Foveation-Texture','Uniform-Blur','Foveation-Blur']

Color_Legend = dict(zip(All_Regimes,Color_Scheme))
Training_Names = dict(zip(All_Regimes,Paradigm_Names))

Model_Types_Input = ['AlexNet','ResNet18','VGG11']
Model_Types_Output = ['AlexNet','ResNet18','VGG11']

Model_Types_Conversion = dict(zip(Model_Types_Input,Model_Types_Output))

# Load all Generalization Models:

index = np.arange(end_run_id-start_run_id+1)
bar_width = 0.1
fig, ax = plt.subplots()

Average_class_matrix = np.zeros((len(Metamer_Type_Family),len(Metamer_Type_Family),len(index)))

Total_Mean = np.zeros((len(Metamer_Type_Family),len(Metamer_Type_Family),1))
Total_Std = np.zeros((len(Metamer_Type_Family),len(Metamer_Type_Family),1))

Type_Counter = 0


#fig, axs = plt.subplots(4, 4,sharex=True,sharey=True)
#fig.suptitle('Sharing x per column, y per row')

#for ax in fig.get_axes():
#    ax.label_outer()

for run_id in range(start_run_id,end_run_id+1):
	fig, axs = plt.subplots(4, 5, sharex=False, sharey=False, gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.08]})
	run_id_str = str(run_id)
	i=0
	for Metamer_Type in Metamer_Type_Family:
		j=0
		for Metamer_Type_Test in Metamer_Type_Family:
			#
			class_confusion_Object_str = './Results/Top' + top_k_num_str + '/' + Metamer_Type + '/' + Model_Type + '_' + run_id_str + '_class_confusion_Object_1_Stimuli_test_' + Metamer_Type_Test + '_Epoch_' + epoch_str +'.npy'
			#
			class_confusion_Object = np.load(class_confusion_Object_str)
			#axs[i,j] = sn.heatmap(class_confusion_Object,xticklabels=scene_classes,yticklabels=scene_classes)
			if j != 3:
				g = sn.heatmap(class_confusion_Object,ax=axs[i,j],cbar=False)
			else:
				axs[i,0].get_shared_y_axes().join(axs[i,1],axs[i,2],axs[i,3])
				g = sn.heatmap(class_confusion_Object,ax=axs[i,j],cbar=True,cbar_ax=axs[i,j+1],vmin=0,vmax=250)
			g.set_xlabel(None)
			g.set_ylabel(None)
			g.set(xticklabels=[])
			g.set(yticklabels=[])
			g.tick_params(bottom=False)
			g.tick_params(left=False)
			#axs[i,j].close()
			#plt.xlabel('Prediction')
			#plt.ylabel('Ground Truth')
			#plt.tight_layout()
			#plt.savefig('./Figures/' + Model_Types_Conversion[Model_Type] + '/Confusion_Matrices/' + Metamer_Type + '_' + Metamer_Type_Test + '_run_id_' + run_id_str + '_Epoch_' + epoch_str + '.svg')
			#plt.savefig('./Figures/' + Model_Types_Conversion[Model_Type] + '/Confusion_Matrices/' + Metamer_Type + '_' + Metamer_Type_Test + '_run_id_' + run_id_str + '_Epoch_' + epoch_str + '.png')
			#plt.show()
			#plt.close()
			j = j + 1
		i = i + 1
	plt.savefig('./Figures/' + Model_Types_Conversion[Model_Type] + '/Confusion_Matrices/Full_Test_run_id' + run_id_str + '_Epoch_' + epoch_str + '.svg')
	plt.savefig('./Figures/' + Model_Types_Conversion[Model_Type] + '/Confusion_Matrices/Full_Test_run_id' + run_id_str + '_Epoch_' + epoch_str + '.png')
	plt.close()

