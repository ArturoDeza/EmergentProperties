clc;close all;clear all;

% These are free toggles/parameters:
start_Distortion = 1;
end_Distortion = 1;

System_Type{1} = 'Reference-Net';
System_Type{2} = 'Foveation-Texture-Net';
System_Type{3} = 'Uniform-Net';
System_Type{4} = 'Foveation-Blur-Net';

% Distortion_Type:
Distortion{1} = 'Window_Cue_Conflict';

Test_Type{1} = 'Cue_Window_Loader';

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

for m=1:4
	for i=start_Distortion:end_Distortion
		i_str = num2str(i);
		for z=1:8
			z_str = num2str(z);
			file_name = ['./Data_Loader/' System_Type{m} '/' Distortion{i} '_' z_str '.txt'];
			fid = fopen(file_name,'w');
			for j=1:length(img_class)
				j_str = num2str(j);
				for img_num = 4751:5000
					img_name_str = num2str(img_num);
					img_name_proxy = [img_name_str '.png'];
					name_in_database = ['./Testing_Images/' System_Type{m} '/' img_class{j} '/' z_str '/' img_name_proxy ' ' j_str '\n'];
					fprintf(fid,name_in_database);
				end
			end
			fclose(fid);
		end
	end
end

