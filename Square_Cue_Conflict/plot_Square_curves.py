import pandas as pd
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import scipy
from scipy import stats, optimize, interpolate
import math

###############################
# Trained on Both Distortions #
###############################

#Model_Type = 'AlexNet'
Model_Type = 'ResNet18'
number_runs = 10
Top_k = 1
#epoch_real = 70
#epoch_real = 60
#epoch_real = 0
#epoch_real = 1
#epoch_real = 5
#epoch_real = 30
#epoch_real = 70
#epoch_real = 120
#epoch_real = 0
#epoch_real = 1
#epoch_real = 4
#epoch_real = 20
#epoch_real = 60
epoch_real = 80

epoch_real_str = str(epoch_real)
number_runs_str = str(number_runs)

epoch_equivalent = epoch_real
epoch_equivalent_str = str(epoch_equivalent)

if Top_k == 1:
	Upper_lim = 80
else:
	Upper_lim = 100

Top_k_str = str(Top_k)

Network_Arch_Names_in = ['ResNet18','AlexNet','VGG11']
Network_Arch_Names_out = ['ResNet18','AlexNet','VGG11']

Network_Name_Mapping = dict(zip(Network_Arch_Names_in,Network_Arch_Names_out))

#Training_Regimes = ['Foveation-Texture-Net','Foveation-Blur-Net']
#Training_Regimes = ['Reference-Net','Uniform-Net']
Training_Regimes = ['Reference-Net','Foveation-Texture-Net','Uniform-Net','Foveation-Blur-Net']

All_Regimes = ['Reference-Net','Foveation-Texture-Net','Uniform-Net','Foveation-Blur-Net']

Color_Scheme = ['darkorange','navy','orchid','seagreen']
Alpha_Values = [1.0,1.0,1.0,1.0]
Line_Values = ['--','-','-','-']

Color_Legend = dict(zip(All_Regimes,Color_Scheme))
Alpha_Legend = dict(zip(All_Regimes,Alpha_Values))
Line_Legend = dict(zip(All_Regimes,Line_Values))

Name_Regimes = ['Reference-Net','Foveation-Texture-Net','Uniform-Net','Foveation-Blur-Net']

Training_Name_Legend = dict(zip(All_Regimes,Name_Regimes))


Experiment_name_mod = ['Square_Uniform_Conflict']

Experiment_name_mod_list = ''

for i in range(len(Experiment_name_mod)):
    Experiment_name_mod_list = Experiment_name_mod_list + '_' + Experiment_name_mod[i]

Training_Regimes_list = ''

for i in range(len(Training_Regimes)):
    Training_Regimes_list = Training_Regimes_list + '_' + Training_Regimes[i]


Experiment_name_in = ['Square_Uniform_Conflict']
Experiment_name_out = ['Square_Uniform_Conflict']

Experiment_Name_Legend = dict(zip(Experiment_name_in,Experiment_name_out))

Acc_Regimes_Periphery = np.zeros((len(Training_Regimes),number_runs,len(Experiment_name_mod),17))
Acc_Regimes_Fovea = np.zeros((len(Training_Regimes),number_runs,len(Experiment_name_mod),17))

for run_id in range(1,number_runs+1):
	run_id_str = str(run_id)
	for regime in range(len(Training_Regimes)):
		for exp in range(len(Experiment_name_mod)):
			for level in range(1,18):
				level_str = str(level)
				file_periphery = './Results/Top' + Top_k_str + '/' + Training_Regimes[regime] + '/' + Model_Type + '_' + run_id_str + '_class_periphery_Object_' + level_str + '_Epoch_' + epoch_real_str + '.npy'
				file_fovea = './Results/Top' + Top_k_str + '/' + Training_Regimes[regime] + '/' + Model_Type + '_' + run_id_str + '_class_fovea_Object_' + level_str + '_Epoch_' + epoch_real_str + '.npy'
				file_total = './Results/Top' + Top_k_str + '/'+ Training_Regimes[regime] + '/' + Model_Type + '_' + run_id_str + '_class_total_Object_' + level_str + '_Epoch_' + epoch_real_str + '.npy'
				# Get Scores
				periphery_vec = np.load(file_periphery)
				fovea_vec = np.load(file_fovea)
				total_vec = np.load(file_total)
				# Compute Accuracy
				Acc_Regimes_Periphery[regime,run_id-1,exp,level-1] = float(np.sum(periphery_vec))/float(np.sum(total_vec))*100.0
				Acc_Regimes_Fovea[regime,run_id-1,exp,level-1] = float(np.sum(fovea_vec))/float(np.sum(total_vec))*100.0

Acc_Regimes_Periphery_mean = np.mean(Acc_Regimes_Periphery,1)
Acc_Regimes_Periphery_std = np.std(Acc_Regimes_Periphery,1)

Acc_Regimes_Fovea_mean = np.mean(Acc_Regimes_Fovea,1)
Acc_Regimes_Fovea_std = np.std(Acc_Regimes_Fovea,1)

x = np.linspace(0,1,17)

# Compute all Cross-Over Points:
def cross_over(Array_Per,Array_Fov):
	cross_array = np.empty(len(Array_Per))
	for z in range(len(Array_Per)):
		x_Mid = math.nan
		for i in range(16):
			if Array_Per[z,0][i]>Array_Fov[z,0][i] and Array_Per[z,0][i+1]<Array_Fov[z,0][i+1]:
				Point_Per_y = np.array((Array_Per[z,0][i],Array_Per[z,0][i+1]))
				Point_Per_x = np.array((x[i],x[i+1]))
				#
				m_Per = (Point_Per_y[1]-Point_Per_y[0])/(Point_Per_x[1]-Point_Per_x[0])
				b_Per = Point_Per_y[0]-m_Per*Point_Per_x[0]
				#
				Point_Fov_y = np.array((Array_Fov[z,0][i],Array_Fov[z,0][i+1]))
				Point_Fov_x = np.array((x[i],x[i+1]))
				#
				m_Fov = (Point_Fov_y[1]-Point_Fov_y[0])/(Point_Fov_x[1]-Point_Fov_x[0])
				b_Fov = Point_Fov_y[0]-m_Fov*Point_Fov_x[0]
				# Now compute the cross-over points:
				x_Mid = (b_Fov-b_Per)/(m_Per-m_Fov)
				break
		cross_array[z] = x_Mid
	return cross_array

Array_Per = Acc_Regimes_Periphery_mean
Array_Fov = Acc_Regimes_Fovea_mean

print('----------------------')
print(cross_over(Array_Per,Array_Fov))
print('----------------------')

##################
# Use Matplotlib #
##################

# Define x-ticks:
#x = np.array([0,1.55,3.43,7.59,16.8,37.18,80.77,100])/100 # These values pre-computed from the area ratio of the log-polar pooling windows
x = np.linspace(0,1,17)

def compute_Area(Acc_Matrix,regime,run_id,exp,links):
	Area_Total = 0.0
	num_points = 17
	for i in range(1,num_points-1):
		Area_i = Acc_Matrix[regime,run_id,exp,i]*np.abs(links[i+1]-links[i])-(Acc_Matrix[regime,run_id,exp,i+1]-Acc_Matrix[regime,run_id,exp,i])*np.abs(links[i+1]-links[i])/2.0
		Area_Total = Area_Total + Area_i
	return 100*Area_Total

fig,axs = plt.subplots(1, len(Experiment_name_mod), figsize=(8, 7.5),squeeze=False)
#fix,axs = plt.subplots(2, 4, figsize=(24, 15),squeeze=False)
fig.suptitle(Model_Type)

fig.suptitle(Network_Name_Mapping[Model_Type] + ' Robustness to Window @ Epoch ' + epoch_equivalent_str )

for i in range(len(Experiment_name_mod)):
	yerr_total_Periphery = np.zeros((len(Training_Regimes),2,17))
	#
	for j in range(len(Training_Regimes)):
		yerr_total_Periphery[j,0,:] = Acc_Regimes_Periphery_std[j,i,:]
		yerr_total_Periphery[j,1,:] = Acc_Regimes_Periphery_std[j,i,:]
	#
	x_i,y_i = int(i/len(Experiment_name_mod)), i% len(Experiment_name_mod)
	#
	for j in range(len(Training_Regimes)):
		axs[x_i,y_i].errorbar(x,Acc_Regimes_Periphery_mean[j,i,:],yerr=yerr_total_Periphery[j,:,:],markersize=10,marker='o',
							  color=Color_Legend[
			Training_Regimes[j]],alpha=Alpha_Legend[Training_Regimes[j]],linestyle=Line_Legend[Training_Regimes[j]],linewidth=2)
	#
	axs[x_i,y_i].legend([Training_Name_Legend[Training_Regimes[z]] for z in range(len(Training_Regimes))])
	yerr_total_Fovea = np.zeros((len(Training_Regimes),2,17))
	#
	for j in range(len(Training_Regimes)):
		yerr_total_Fovea[j,0,:] = Acc_Regimes_Fovea_std[j,i,:]
		yerr_total_Fovea[j,1,:] = Acc_Regimes_Fovea_std[j,i,:]
	#
	x_i,y_i = int(i/len(Experiment_name_mod)), i% len(Experiment_name_mod)
	#
	for j in range(len(Training_Regimes)):
		axs[x_i,y_i].errorbar(x,Acc_Regimes_Fovea_mean[j,i,:],yerr=yerr_total_Fovea[j,:,:],markersize=10,marker='*',
							  color=Color_Legend[
			Training_Regimes[j]],alpha=Alpha_Legend[Training_Regimes[j]],linestyle=Line_Legend[Training_Regimes[j]],linewidth=2)
	#
	axs[x_i,y_i].plot(x,np.repeat(Top_k/20.0*100.0,17),'--y',linewidth=2)
	axs[x_i,y_i].set(xlim=(-0.1,1.1),ylim=(0,Upper_lim))
	axs[x_i,y_i].set_title(Experiment_Name_Legend[Experiment_name_mod[i]])
	axs[x_i,y_i].set_xticks(np.arange(0,1.1,step=0.1))
	axs[x_i,y_i].set_ylabel('("o") Peripheral Scene Classification Accuracy (%)')
	ax2 = axs[x_i,y_i].twinx() # instantiate a second axes that shares the same x-axis
	ax2.set_ylabel('("*") Foveal Scene Classification Accuracy (%)')
	ax2.set(xlim=(-0.1,1.1),ylim=(0,Upper_lim))
	axs[x_i,y_i].set_xlabel('Percentage of Central Image Area with Foveal Class')


plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('./Plots/' + Model_Type + '/Top_' + Top_k_str + '_Epoch_' + epoch_real_str + Experiment_name_mod_list + Training_Regimes_list + '_Runs_' + number_runs_str + '.png')
plt.savefig('./Plots/' + Model_Type + '/Top_' + Top_k_str + '_Epoch_' + epoch_real_str + Experiment_name_mod_list + Training_Regimes_list + '_Runs_' + number_runs_str +'.svg')

plt.show()


########################
# Use Individual Plots #
########################

# Define x-ticks:
x = np.linspace(0,1,17)

fig,axs = plt.subplots(1, len(Experiment_name_mod), figsize=(8, 7.5),squeeze=False)
#fix,axs = plt.subplots(2, 4, figsize=(24, 15),squeeze=False)
fig.suptitle(Model_Type)

fig.suptitle(Network_Name_Mapping[Model_Type] + ' Robustness to Window @ Epoch ' + epoch_equivalent_str )

for i in range(len(Experiment_name_mod)):
	x_i,y_i = int(i/len(Experiment_name_mod)), i% len(Experiment_name_mod)
	#
	for z in range(number_runs):
		for j in range(len(Training_Regimes)):
			axs[x_i,y_i].plot(x,Acc_Regimes_Periphery[j,z,i,:],markersize=10,marker='o',color=Color_Legend[Training_Regimes[j]],alpha=0.5,linestyle=Line_Legend[Training_Regimes[j]],linewidth=1)
	#
	axs[x_i,y_i].legend([Training_Name_Legend[Training_Regimes[z]] for z in range(len(Training_Regimes))])
	#
	x_i,y_i = int(i/len(Experiment_name_mod)), i% len(Experiment_name_mod)
	#
	for z in range(number_runs):
		for j in range(len(Training_Regimes)):
			axs[x_i,y_i].plot(x,Acc_Regimes_Fovea[j,z,i,:],markersize=10,marker='*', color=Color_Legend[Training_Regimes[j]],alpha=Alpha_Legend[Training_Regimes[j]],linestyle=Line_Legend[Training_Regimes[j]],linewidth=2)
			print(cross_over(Acc_Regimes_Periphery[:,z,:,:],Acc_Regimes_Fovea[:,z,:,:]))
	#
	axs[x_i,y_i].plot(x,np.repeat(Top_k/20.0*100.0,17),'--y',linewidth=2)
	axs[x_i,y_i].set(xlim=(-0.1,1.1),ylim=(0,Upper_lim))
	axs[x_i,y_i].set_title(Experiment_Name_Legend[Experiment_name_mod[i]])
	axs[x_i,y_i].set_xticks(np.arange(0,1.1,step=0.1))
	axs[x_i,y_i].set_ylabel('("o") Peripheral Scene Classification Accuracy (%)')
	ax2 = axs[x_i,y_i].twinx() # instantiate a second axes that shares the same x-axis
	ax2.set_ylabel('("*") Foveal Scene Classification Accuracy (%)')
	ax2.set(xlim=(-0.1,1.1),ylim=(0,Upper_lim))
	axs[x_i,y_i].set_xlabel('Percentage of Central Image Area with Foveal Class')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('./Plots_Individual/' + Model_Type + '/Top_' + Top_k_str + '_Epoch_' + epoch_real_str + Experiment_name_mod_list + Training_Regimes_list + '_Runs_' + number_runs_str + '.png')
plt.savefig('./Plots_Individual/' + Model_Type + '/Top_' + Top_k_str + '_Epoch_' + epoch_real_str + Experiment_name_mod_list + Training_Regimes_list + '_Runs_' + number_runs_str + '.svg')

plt.show()

