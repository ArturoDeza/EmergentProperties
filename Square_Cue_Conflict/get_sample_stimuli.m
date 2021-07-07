clc; close all; clear all;

% Class_Name:
img_class{1} = 'aquarium';
img_class{2} = 'badlands';
img_class{3} = 'bedroom';
img_class{4} = 'bridge';
img_class{5} = 'campus';
img_class{6} = 'corridor';
img_class{7} = 'forest_path';
img_class{8} = 'highway';
img_class{9} = 'hospital';
img_class{10} = 'industrial_area';
img_class{11} = 'japanese_garden';
img_class{12} = 'kitchen';
img_class{13} = 'mansion';
img_class{14} = 'mountain';
img_class{15} = 'ocean';
img_class{16} = 'office';
img_class{17} = 'restaurant';
img_class{18} = 'skyscraper';
img_class{19} = 'train_interior';
img_class{20} = 'waterfall';


Cue_Conflict{1} = 'Square_Uniform_Cue_Conflict';

img_src = './Testing_Images/'
img_dst = './Sample_Stimuli/'

Network{1} = 'Reference-Net';
Network{2} = 'Foveation-Texture-Net';
Network{3} = 'Uniform-Net';
Network{4} = 'Foveation-Blur-Net';

for i=1:17
	i_str = num2str(i);
	for k=1:1
		for j=1:4
			img = [img_src Network{j} '/' img_class{1} '/' i_str '/4751.png'];
			img_final = [img_dst Cue_Conflict{k} '/' Network{j} '/' i_str '_.png'];
			system_command = ['cp ' img ' ' img_final];			
			system(system_command);
		end
	end
end

