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

Model_Type = 'AlexNet'
#epoch = 120
#epoch = 70 
#epoch = 30
#epoch = 5
#epoch = 1
epoch = 0

#Model_Type = 'ResNet18'
#epoch = 80
#epoch = 60
#epoch= 0
#epoch = 1
#epoch = 4
#epoch = 20

#Model_Type = sys.argv[1]
#epoch = int(sys.argv[2])
#Stimuli_test_str = sys.argv[3]

#Model_Type = 'AlexNet'
#epoch = 70
#Model_Type = 'ResNet18'
#Model_Type = 'VGG11'

start_run_id = 1
end_run_id = 10
top_k_num = 1


num_runs = end_run_id - start_run_id + 1
num_runs_str = str(num_runs)

epoch_str = str(epoch)

top_k_num_str = str(top_k_num)


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


ctr1 = 0
for Metamer_Type in Metamer_Type_Family:
	ctr2 = 0
	for Metamer_Type_Test in Metamer_Type_Family:
		for run_id in range(start_run_id,end_run_id+1):
			run_id_str = str(run_id)

			class_correct_Object_str = './Results/Top' + \
									   top_k_num_str + '/' + Metamer_Type + '/' + Model_Type + '_' + run_id_str + \
									   '_class_correct_Object_1_Stimuli_test_' + Metamer_Type_Test + '_Epoch_' + epoch_str + '.npy'
			class_total_Object_str = './Results/Top' + top_k_num_str \
									 + '/' + Metamer_Type + '/' + Model_Type + '_' + run_id_str + \
									 '_class_total_Object_1_Stimuli_test_' + Metamer_Type_Test + '_Epoch_' + epoch_str +'.npy'

			class_correct_Object = np.load(class_correct_Object_str)
			class_total_Object = np.load(class_total_Object_str)
			#
			Total_Accuracy = float(np.sum(class_correct_Object))/float(np.sum(class_total_Object))*100.0
			#
			Average_class_matrix[ctr1,ctr2,run_id-1] = Total_Accuracy
			#
			if Metamer_Type == Metamer_Type_Test:
				#rects = plt.bar(0+(1.1*ctr1)-np.ceil((end_run_id-start_run_id)/2)*bar_width+(run_id-1)*bar_width,Average_class,
				#				bar_width,color=Color_Legend[Metamer_Type],label=Metamer_Type,edgecolor='white')
				if Metamer_Type == 'Reference-Net':
					rects = plt.bar(0+(1.1*ctr1)-np.ceil((end_run_id-start_run_id)/2)*bar_width+(run_id-1)*bar_width,Total_Accuracy,
								bar_width,color=Color_Legend[Metamer_Type],label=Metamer_Type,edgecolor='white',hatch='//')
				else:
					rects = plt.bar(0+(1.1*ctr1)-np.ceil((end_run_id-start_run_id)/2)*bar_width+(run_id-1)*bar_width,Total_Accuracy,bar_width,color=Color_Legend[Metamer_Type],label=Metamer_Type,edgecolor='white')
			if Metamer_Type == Metamer_Type_Test:
				plt.plot(0+(1.1*ctr1)-np.ceil((end_run_id-start_run_id)/2)*bar_width+(run_id-1)*bar_width,Total_Accuracy,marker='s',color=Color_Legend[Metamer_Type_Test],markersize=bar_width*60,markeredgecolor='white')
			else:
				plt.plot(0+(1.1*ctr1)-np.ceil((end_run_id-start_run_id)/2)*bar_width+(run_id-1)*bar_width,Total_Accuracy,marker='D',color=Color_Legend[Metamer_Type_Test],markersize=bar_width*60,markeredgecolor='white')
		ctr2 = ctr2 + 1
	ctr1 = ctr1 + 1

plt.xlabel('Number of Trained Network runs')
plt.ylabel('Top-1 Accuracy (%)')

plt.title('i.i.d and o.o.d Generalization Accuracy of ' + Model_Types_Conversion[Model_Type] + ' @ ' + epoch_str + ' Epochs')

#plt.xticks(index+1,(index+1))
plt.xticks(np.arange(-0.05,-0.05+len(Metamer_Type_Family)*1.1,1.1),[Training_Names[k] for k in Metamer_Type_Family])
plt.yticks(np.arange(0,90,10),[0,10,20,30,40,50,60,70,80])
#plt.legend(Metamer_Type_Family)
#plt.show()

Family_Total = ''

for i in range(len(Metamer_Type_Family)):
	Family_Total = Family_Total + Metamer_Type_Family[i] + '_'

plt.savefig('./Figures/' + Model_Types_Conversion[Model_Type] + '/' + Family_Total + 'num_runs_' + num_runs_str + '_Epoch_' + epoch_str + '.svg')
plt.savefig('./Figures/' + Model_Types_Conversion[Model_Type] + '/' + Family_Total + 'num_runs_' + num_runs_str + '_Epoch_' + epoch_str + '.png')
plt.show()

# Other Plots

# Compute Generalization Scores:
# (diagonal entries)
Cross_Generalization_Mean = np.mean(Average_class_matrix,2)
Cross_Generalization_Std = np.std(Average_class_matrix,2)

print(Cross_Generalization_Mean)
print(Cross_Generalization_Std)


