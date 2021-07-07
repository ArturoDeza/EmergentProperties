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

preamble_Transform{1} = 'Baseline';
preamble_Transform{2} = '0.4';
preamble_Transform{3} = 'Matched_Blur';
preamble_Transform{4} = 'Ada_Gauss';

preamble_Suffix{1} = '_Reference.png';
preamble_Suffix{2} = '_metamer_s0.4_5.png';
preamble_Suffix{3} = '_Matched_Blur.png';
preamble_Suffix{4} = '_Ada_Gauss.png';


miniamble_Transform{1} = '';
miniamble_Transform{2} = '_Metamer';
miniamble_Transform{3} = '_Matched';
miniamble_Transform{4} = '_Ada_Gauss';

mid_x = 128;
mid_y = 128;

for m=1:4
	for i=start_Distortion:end_Distortion
		i_str = num2str(i);
		% Level of Occlusion:
		for z=1:17
			z_str = num2str(z);
			file_name = ['./Data_Loader/' Metamer_Type{m} '/' Distortion{i} '_' z_str '.txt'];
			fid = fopen(file_name,'w');
			% Occluder:
			d_area = round(16*(z-1)*256);
			d_eq_half = ceil(sqrt(d_area)/2); % Changed from floor to ceil to get better approximation power with Scotoma
			%%		
			lower_x = mid_x-d_eq_half+1;
			higher_x = mid_x+d_eq_half;
			lower_y = mid_y-d_eq_half+1;
			higher_y = mid_y+d_eq_half;
			%
			j2 = 1;
			for j=1:length(img_class)
				j_str = num2str(j);
				for img_num = 4751:5000
					% Check for Same Category -- if so skip to next one!
					if j2 == j
						j2 = j2 + 1;
					end
					% Check for Overflow
					if j2 == 21
						j2 = 1;
					end
					% Convert z2 to string:
					j2_str = num2str(j2);
					% Load Images:
					img_name_str = num2str(img_num);
					img_input_name = [img_name_str preamble_Suffix{m}];
					img_periphery_name = ['/cbcl/cbcl01/deza/Datasets/Datasets/NeuroFovea/' preamble_Transform{m} '/' img_class{j} '/' img_input_name];
					img_fovea_name = ['/cbcl/cbcl01/deza/Datasets/Datasets/NeuroFovea/' preamble_Transform{m} '/' img_class{j2} '/' img_input_name]
					%
					img_periphery = imread(img_periphery_name);
					img_fovea = imread(img_fovea_name);
					% Now compute occluded Area of Stimuli:
					% Get Area equivalent width [Original]
					%
					img_final = img_fovea;
					image_final(lower_x:higher_x,lower_y:higher_y,:) = image_periphery(lower_x:higher_x,lower_y:higher_y,:);
					% Save New Image:
					img_name_proxy = [img_name_str '.png'];
					image_final_name = ['./Testing_Images/' Metamer_Type{m} '/' img_class{j} '/' j_str '/' img_name_proxy];
					imwrite(image_final,img_final_name,'PNG');
					%
					% Add Other Scene Count!					
					j2 = j2 + 1;
				end
			end
			fclose(fid);
		end
	end
end
