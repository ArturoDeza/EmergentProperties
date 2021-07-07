import pandas as pd
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt

###############################
# Trained on Both Distortions #
###############################

#Model_Type = 'AlexNet'
Model_Type = 'ResNet18'
epoch_real = 60
number_runs = 10
Top_k = 1
#epoch_real = 120
#epoch_real = 70
#epoch_real = 30
#epoch_real = 5
#epoch_real = 1
#epoch_real = 0
#epoch_real = 80
Experiment_name_mod = ['Left2Right','Top2Bottom','Scotoma','Glaucoma']
#Experiment_name_mod = ['Glaucoma','Scotoma']

epoch_real_str = str(epoch_real)
number_runs_str = str(number_runs)

epoch_equivalent = epoch_real
epoch_equivalent_str = str(epoch_equivalent)

if Top_k == 1:
	Upper_lim = 0.8
else:
	Upper_lim = 1.0

Top_k_str = str(Top_k)

Network_Arch_Names_in = ['ResNet18','AlexNet','VGG11']
Network_Arch_Names_out = ['ResNet18','AlexNet','VGG11']

Network_Name_Mapping = dict(zip(Network_Arch_Names_in,Network_Arch_Names_out))

Training_Regimes = ['Reference-Net','Foveation-Texture-Net','Uniform-Net','Foveation-Blur-Net']

#Training_Regimes = ['Reference-Net','Foveation-Texture-Net','Uniform-Net']

All_Regimes = ['Reference-Net','Foveation-Texture-Net','Uniform-Net','Foveation-Blur-Net']
Color_Scheme = ['darkorange','navy','orchid','seagreen']
Alpha_Values = [1.0,1.0,1.0,1.0]
Line_Values = ['--','-','-','-']

Color_Legend = dict(zip(All_Regimes,Color_Scheme))
Alpha_Legend = dict(zip(All_Regimes,Alpha_Values))
Line_Legend = dict(zip(All_Regimes,Line_Values))

Name_Regimes = ['Reference-Net','Foveation-Texture-Net','Uniform-Blur-Net','Foveation-Blur-Net']


Training_Name_Legend = dict(zip(All_Regimes,Name_Regimes))

Experiment_name_mod_list = ''

for i in range(len(Experiment_name_mod)):
    Experiment_name_mod_list = Experiment_name_mod_list + '_' + Experiment_name_mod[i]

Training_Regimes_list = ''

for i in range(len(Training_Regimes)):
    Training_Regimes_list = Training_Regimes_list + '_' + Training_Regimes[i]


Experiment_name_in = ['Left2Right','Top2Bottom','Glaucoma','Scotoma']
Experiment_name_out = ['Left2Right','Top2Bottom','Glaucoma','Scotoma']

Experiment_Name_Legend = dict(zip(Experiment_name_in,Experiment_name_out))

Num_Distortion_Steps = 17

Acc_Regimes = np.zeros((len(Training_Regimes),number_runs,len(Experiment_name_mod),Num_Distortion_Steps))

for run_id in range(1,number_runs+1):
	run_id_str = str(run_id)
	for regime in range(len(Training_Regimes)):
		for exp in range(len(Experiment_name_mod)):
			for level in range(1,Num_Distortion_Steps + 1):
				level_str = str(level)
				file_correct = './Results/Top' + Top_k_str + '/' + Training_Regimes[regime] + '/' + Experiment_name_mod[exp] + '/' + Model_Type + '_' + run_id_str + '_class_correct_Object_' + level_str + '_Epoch_' + epoch_real_str + '.npy'
				file_total = './Results/Top' + Top_k_str + '/'+ Training_Regimes[regime] + '/' + Experiment_name_mod[exp] + '/' + Model_Type + '_' + run_id_str + '_class_total_Object_' + level_str + '_Epoch_' + epoch_real_str + '.npy'
				# Get Scores
				correct_vec = np.load(file_correct)
				total_vec = np.load(file_total)
				# Compute Accuracy
				Acc_Regimes[regime,run_id-1,exp,level-1] = float(np.sum(correct_vec))/float(np.sum(total_vec))

Acc_Regimes_mean = np.mean(Acc_Regimes,1)
Acc_Regimes_std = np.std(Acc_Regimes,1)

##################
# Use Matplotlib #
##################

########################
# Plot Aggregate Curve #
########################

fig,axs = plt.subplots(1, len(Experiment_name_mod), figsize=(24, 7.5),squeeze=False)
fig.suptitle(Model_Type)
x = np.arange(Num_Distortion_Steps)

fig.suptitle(Network_Name_Mapping[Model_Type] + ' Robustness to Occlusions @ Epoch ' + epoch_equivalent_str)

for i in range(len(Experiment_name_mod)):
	yerr_total = np.zeros((len(Training_Regimes),2,Num_Distortion_Steps))
	#
	for j in range(len(Training_Regimes)):
		yerr_total[j,0,:] = Acc_Regimes_std[j,i,:]
		yerr_total[j,1,:] = Acc_Regimes_std[j,i,:]
	#
	x_i,y_i = int(i/len(Experiment_name_mod)), i% len(Experiment_name_mod)
	#
	for j in range(len(Training_Regimes)):
		axs[x_i,y_i].errorbar(x,Acc_Regimes_mean[j,i,:],yerr=yerr_total[j,:,:],marker='s',
							  color=Color_Legend[
			Training_Regimes[j]],alpha=Alpha_Legend[Training_Regimes[j]],linestyle=Line_Legend[Training_Regimes[j]],linewidth=2)
	#
	axs[x_i,y_i].legend([Training_Name_Legend[Training_Regimes[z]] for z in range(len(Training_Regimes))])
	axs[x_i,y_i].plot(x,np.repeat(Top_k/20.0,Num_Distortion_Steps),'--y',linewidth=2)
	axs[x_i,y_i].set(xlim=(-0.5,7.5),ylim=(0,Upper_lim))
	axs[x_i,y_i].set_title(Experiment_Name_Legend[Experiment_name_mod[i]])
	axs[x_i,y_i].set_xticks(np.arange(0,Num_Distortion_Steps,step=1))
	axs[x_i,y_i].set_ylabel('Top-1 Accuracy Ratio')
	axs[x_i,y_i].set_xlabel('Level of Occlusion')

print('Got Here!')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./Plots/' + Model_Type + '/' + 'Top_' + Top_k_str + '_Epoch_' + epoch_real_str + Experiment_name_mod_list + Training_Regimes_list + '_Runs_' + number_runs_str + '.png')
plt.savefig('./Plots/' + Model_Type + '/' + 'Top_' + Top_k_str + '_Epoch_' + epoch_real_str + Experiment_name_mod_list + Training_Regimes_list + '_Runs_' + number_runs_str + '.svg')

plt.show()

##########################
# Plot Individual Curves #
##########################

fig,axs = plt.subplots(1, len(Experiment_name_mod), figsize=(24, 7.5),squeeze=False)
fig.suptitle(Model_Type)
x = np.arange(Num_Distortion_Steps)

fig.suptitle(Network_Name_Mapping[Model_Type] + ' Robustness to Occlusions @ Epoch ' + epoch_equivalent_str)

for i in range(len(Experiment_name_mod)):
	yerr_total = np.zeros((len(Training_Regimes),2,Num_Distortion_Steps))
	#
	for j in range(len(Training_Regimes)):
		yerr_total[j,0,:] = Acc_Regimes_std[j,i,:]
		yerr_total[j,1,:] = Acc_Regimes_std[j,i,:]
	#
	x_i,y_i = int(i/len(Experiment_name_mod)), i% len(Experiment_name_mod)
	#
	for z in range(number_runs):
		for j in range(len(Training_Regimes)):
			axs[x_i,y_i].plot(x, Acc_Regimes[j,z,i,:], marker='s', color=Color_Legend[Training_Regimes[j]], alpha=0.5, linestyle=Line_Legend[Training_Regimes[j]],linewidth=1)
	#
	axs[x_i,y_i].legend([Training_Name_Legend[Training_Regimes[z]] for z in range(len(Training_Regimes))])
	axs[x_i,y_i].plot(x,np.repeat(Top_k/20.0,Num_Distortion_Steps),'--y',linewidth=2)
	axs[x_i,y_i].set(xlim=(-0.5,7.5),ylim=(0,Upper_lim))
	axs[x_i,y_i].set_title(Experiment_Name_Legend[Experiment_name_mod[i]])
	axs[x_i,y_i].set_xticks(np.arange(0,Num_Distortion_Steps,step=1))
	axs[x_i,y_i].set_ylabel('Top-1 Accuracy Ratio')
	axs[x_i,y_i].set_xlabel('Level of Occlusion')

print('Got Here!')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./Plots_Individual/' + Model_Type + '/' + 'Top_' + Top_k_str + '_Epoch_' + epoch_real_str + Experiment_name_mod_list + Training_Regimes_list + '_Runs_' + number_runs_str + '.png')
plt.savefig('./Plots_Individual/' + Model_Type + '/' + 'Top_' + Top_k_str + '_Epoch_' + epoch_real_str + Experiment_name_mod_list + Training_Regimes_list + '_Runs_' + number_runs_str + '.svg')

plt.show()



