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

Network_Name{1} = 'Reference-Net';
Network_Name{2} = 'Foveation-Texture-Net';
Network_Name{3} = 'Uniform-Net';
Network_Name{4} = 'Foveation-Blur-Net';

% To Do:
for z=1:4
	
	% Training Dataset:
	fid = fopen(['./Training/Mini_Places_' Network_Name{z} '.txt'],'w');
	for i=1:20
		i_str = num2str(i);
		for j=1:4500
			j_str = num2str(j);
			img_str = ['../All_Training_Images/' Network_Name{z} '/' img_class{i} '/' j_str '.png ' i_str '\n'];
			fprintf(fid,img_str);
		end
	end
	fclose(fid);

	% Validation Dataset:
	fid = fopen(['./Validation/Mini_Places_' Network_Name{z} '.txt'],'w');
	for i=1:20
		i_str = num2str(i);
		for j=4501:4750
			j_str = num2str(j);
			img_str = ['../All_Validation_Images/' Network_Name{z} '/' img_class{i} '/' j_str '.png ' i_str '\n'];
			fprintf(fid,img_str);
		end
	end
	fclose(fid);

	% Testing Dataset
	fid = fopen(['./Testing/Mini_Places_' Network_Name{z} '.txt'],'w');
	for i=1:20
		i_str = num2str(i);
		for j=4751:5000
			j_str = num2str(j);
			img_str = ['../All_Testing_Images/' Network_Name{z} '/' img_class{i} '/' j_str '.png ' i_str '\n'];
			fprintf(fid,img_str);
		end
	end
	fclose(fid);

end
