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

mid_x = 128;
mid_y = 128;

start_Distortion = 1;
end_Distortion = 1;
Distortion{1} = 'Robustness_to_Occlusion';

Transform_Type{1} = 'Reference-Net';
Transform_Type{2} = 'Foveation-Texture-Net';
Transform_Type{3} = 'Uniform-Net';
Transform_Type{4} = 'Foveation-Blur-Net';

scotoma_flag = 0;
glaucoma_flag = 1;
left2right_flag = 0;
top2bottom_flag = 0;

for m=1:4
	for i=start_Distortion:end_Distortion
		i_str = num2str(i);
		% Level of Occlusion:
		for z=1:17
			z_str = num2str(z);
			% Occluder:
			d_area = round(16*(z-1)*256);
			d_eq_half = ceil(sqrt(d_area)/2); % Changed from floor to ceil to get better approximation power with Scotoma
			%%		
			lower_x = mid_x-d_eq_half+1;
			higher_x = mid_x+d_eq_half;
			lower_y = mid_y-d_eq_half+1;
			higher_y = mid_y+d_eq_half;
			%
			b_eq = sqrt(256^2-d_area);
			b_eq_half = ceil(b_eq/2);
			for j=1:length(img_class)
				j_str = num2str(j);
				for img_num = 4751:5000
					% Load Images:
					img_name_str = num2str(img_num);
					img_input_name = [img_name_str '.png'];
					img_main_name = ['../All_Testing_Images/' Transform_Type{m} '/' img_class{j} '/' img_input_name];
					%
					img_main = imread(img_main_name);
					% Get Area equivalent width [Original]
					image_scotoma = img_main;
					image_glaucoma = uint8(128*ones(256,256,3));
					image_left2right = img_main;
					image_top2bottom = img_main;
					% Occlude with Scotoma:
					image_scotoma(lower_x:higher_x,lower_y:higher_y,:) = 128;
					% Do Opposite Procedure: Start with Gray Image and Fill the Center for Glaucoma!
					image_glaucoma(mid_x-b_eq_half+1:mid_x+b_eq_half,mid_y-b_eq_half+1:mid_y+b_eq_half,:) = img_main(mid_x-b_eq_half+1:mid_x+b_eq_half,mid_y-b_eq_half+1:mid_y+b_eq_half,:);
					% Cover Left2Right:
					image_left2right(:,1:16*(z-1),:) = 128;
					% Cover Top2Bottom:
					image_top2bottom(1:16*(z-1),:,:) = 128;
					% Save New Image:
					img_name_proxy = [img_name_str '.png'];
					%
					% Save Scotoma Image:
					if scotoma_flag == 1
						image_scotoma_name = ['./Testing_Images/Scotoma/' Transform_Type{m} '/' img_class{j} '/' z_str '/' img_name_proxy];
						imwrite(image_scotoma,image_scotoma_name,'PNG');
					end
					% Save Glaucoma Image:
					if glaucoma_flag == 1
						image_glaucoma_name = ['./Testing_Images/Glaucoma/' Transform_Type{m} '/' img_class{j} '/' z_str '/' img_name_proxy];
						imwrite(image_glaucoma,image_glaucoma_name,'PNG');
					end
					% Save Left2Right Image:
					if left2right_flag == 1
						image_left2right_name = ['./Testing_Images/Left2Right/' Transform_Type{m} '/' img_class{j} '/' z_str '/' img_name_proxy];
						imwrite(image_left2right,image_left2right_name,'PNG');
					end
					% Save Top2Bottom Image:
					if top2bottom_flag == 1
						image_top2bottom_name = ['./Testing_Images/Top2Bottom/' Transform_Type{m} '/' img_class{j} '/' z_str '/' img_name_proxy];
						imwrite(image_top2bottom,image_top2bottom_name,'PNG');
					end
				end
				display(z)
				display(j)
			end
		end
	end
end
