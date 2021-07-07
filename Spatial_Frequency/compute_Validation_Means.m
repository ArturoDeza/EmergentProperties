function compute_Validation_Means(z_start,z_end)

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

preamble_Transform = '/Testing_Images/Testing_Color/';
preamble_Metamer_Transform = '/Testing_Images/Testing_Color_Metamer/';
preamble_Matched_Blur = '/Testing_Images/Testing_Color_Matched_Blur';
preamble_Ada_Gauss = '/Testing_Images/Testing_Color_Ada_Gauss';


Template_prefix{1} = 'Baseline';
Template_prefix{2} = '0.4';
Template_prefix{3} = 'Matched_Blur';
Template_prefix{4} = 'Ada_Gauss';

Template_suffix{1} = '_Reference.png';
Template_suffix{2} = '_metamer_s0.4_5.png';
Template_suffix{3} = '_Matched_Blur.png';
Template_suffix{4} = '_Ada_Gauss.png';


save_R_name{1} = './Avg_Color_Bias/mean_img_R_Reference-Net.mat';
save_G_name{1} = './Avg_Color_Bias/mean_img_G_Reference-Net.mat';
save_B_name{1} = './Avg_Color_Bias/mean_img_B_Reference-Net.mat';

save_R_name{2} = './Avg_Color_Bias/mean_img_R_Foveation-Texture-Net.mat';
save_G_name{2} = './Avg_Color_Bias/mean_img_G_Foveation-Texture-Net.mat';
save_B_name{2} = './Avg_Color_Bias/mean_img_B_Foveation-Texture-Net.mat';

save_R_name{3} = './Avg_Color_Bias/mean_img_R_Uniform-Net.mat';
save_G_name{3} = './Avg_Color_Bias/mean_img_G_Uniform-Net.mat';
save_B_name{3} = './Avg_Color_Bias/mean_img_B_Uniform-Net.mat';

save_R_name{4} = './Avg_Color_Bias/mean_img_R_Foveation-Blur-Net.mat';
save_G_name{4} = './Avg_Color_Bias/mean_img_G_Foveation-Blur-Net.mat';
save_B_name{4} = './Avg_Color_Bias/mean_img_B_Foveation-Blur-Net.mat';

for z=z_start:z_end
	% For each scene:
	for i=1:20
		for j=1:250
			% Img indxs:
			img_indx = j+4500;
			img_indx_str = num2str(img_indx);
			% For every image in testing set
			img = imread(['/cbcl/cbcl01/deza/Datasets/Datasets/NeuroFovea/' Template_prefix{z} '/' img_class{i} '/' img_indx_str Template_suffix{z}]);
			% Compute image averages:
			img_R(i,j) = mean2(img(:,:,1));
			img_G(i,j) = mean2(img(:,:,2));
			img_B(i,j) = mean2(img(:,:,3));
		end
		disp(i)
	end
	%
	mean_img_R = mean(img_R(:))/255.0;
	mean_img_G = mean(img_G(:))/255.0;
	mean_img_B = mean(img_B(:))/255.0;
	%
	save(save_R_name{z},'mean_img_R');
	save(save_G_name{z},'mean_img_G');
	save(save_B_name{z},'mean_img_B');
	%
end



