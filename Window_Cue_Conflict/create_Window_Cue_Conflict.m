clc;close all;clear all;

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

Network_Type{1} = 'Reference-Net';
Network_Type{2} = 'Foveation-Texture-Net';
Network_Type{3} = 'Uniform-Net';
Network_Type{4} = 'Foveation-Blur-Net';


preamble_Transform{1} = 'Baseline';
preamble_Transform{2} = '0.4';
preamble_Transform{3} = 'Matched_Blur';
preamble_Transform{4} = 'Ada_Gauss';

preamble_Suffix{1} = '_Reference.png';
preamble_Suffix{2} = '_metamer_s0.4_5.png';
preamble_Suffix{3} = '_Matched_Blur.png';
preamble_Suffix{4} = '_Ada_Gauss.png';

testing_folder_suffix{1} = '';
testing_folder_suffix{2} = '_Metamer';
testing_folder_suffix{3} = '_Matched';
testing_folder_suffix{4} = '_Ada_Gauss';

mid_x = 128;
mid_y = 128;

for k=1:8
	%
	k_str = num2str(k);
	% Load window:
	foveal_window_ch = imresize(imread(['./Foveal_Mask_Collection/foveal_template_' k_str '.png']),[256 256]);
	foveal_window_ch = double(foveal_window_ch)/255.0;
	%
	foveal_window(:,:,1) = foveal_window_ch;
	foveal_window(:,:,2) = foveal_window_ch;
	foveal_window(:,:,3) = foveal_window_ch;
	%
	foveal_residual = 1-foveal_window;
	%
	for z=1:4
		z_str = num2str(z);
		for i=1:20
			if i<20
				i2 = i+1;
			else
				i2 = 1;
			end
			for j=1:250
			%for j=1:1
				img_id = 4750+j;
				img_indx_str = num2str(img_id);
				%
				img = imread(['../All_Testing_Images/' Network_Type{z} '/' img_class{i} '/' img_indx_str preamble_Suffix{z}]);
				%
				img_decoy = imread(['../All_Testing_Images/' Network_Type{z} '/' img_class{i2} '/' img_indx_str preamble_Suffix{z}]);
				%
				image_Foveal_Cue = uint8(double(img_decoy).*foveal_window + double(img).*foveal_residual);
				%
				imwrite(image_Foveal_Cue,['./Testing_Images/' Network_Type{z} '/' img_class{i} '/' k_str '/' img_indx_str '.png']);
			end
		end
	end
end
