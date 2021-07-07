clc;close all;clear all;

addpath('./minf')

class_id{1} = 'aquarium';
class_id{2} = 'badlands';
class_id{3} = 'bedroom';
class_id{4} = 'bridge';
class_id{5} = 'campus';
class_id{6} = 'corridor';
class_id{7} = 'forest_path';
class_id{8} = 'highway';
class_id{9} = 'hospital';
class_id{10} = 'industrial_area';
class_id{11} = 'japanese_garden';
class_id{12} = 'kitchen';
class_id{13} = 'mansion';
class_id{14} = 'mountain';
class_id{15} = 'ocean';
class_id{16} = 'office';
class_id{17} = 'restaurant';
class_id{18} = 'skyscraper';
class_id{19} = 'train_interior';
class_id{20} = 'waterfall';

MSE = nan(20,500,4);
Mutual_Information = nan(20,500,4);
SSIM = nan(20,500,4);
%MS_SSIM = nan(20,500,3);

check_flag = 0;

for i=1:20
	i_str = num2str(i)
	for j=4501:5000
		j_str = num2str(j);
		%			
		% Reference-Net
		img_ref = imread(['../../../Datasets/Datasets/NeuroFovea/Baseline/' class_id{i} '/' j_str '_Reference.png']);
		% Foveation-Texture-Net
		img_Fov_Texture = imread(['../../../Datasets/Datasets/NeuroFovea/0.4/' class_id{i} '/' j_str '_metamer_s0.4_5.png']);
		% Uniform-Net
		img_Uniform = imread(['../../../Datasets/Datasets/NeuroFovea/Matched_Blur/' class_id{i} '/' j_str '_Matched_Blur.png']);
		% Foveation-Blur-Net
		img_Fov_Blur = imread(['../../../Datasets/Datasets/NeuroFovea/Ada_Gauss/' class_id{i} '/' j_str '_Ada_Gauss.png']);

		if check_flag == 0

			MSE(i,j-4500,1) = mean2((double(rgb2gray(img_ref))-double(rgb2gray(img_Fov_Texture))).^2);
			MSE(i,j-4500,2) = mean2((double(rgb2gray(img_ref))-double(rgb2gray(img_Uniform))).^2);
			MSE(i,j-4500,3) = mean2((double(rgb2gray(img_ref))-double(rgb2gray(img_Fov_Blur))).^2);
			MSE(i,j-4500,4) = mean2((double(rgb2gray(img_ref))-double(rgb2gray(img_ref))).^2);

			MI(i,j-4500,1) = mi(rgb2gray(img_ref),rgb2gray(img_Fov_Texture));
			MI(i,j-4500,2) = mi(rgb2gray(img_ref),rgb2gray(img_Uniform));
			MI(i,j-4500,3) = mi(rgb2gray(img_ref),rgb2gray(img_Fov_Blur));
			MI(i,j-4500,4) = mi(rgb2gray(img_ref),rgb2gray(img_ref));


			SSIM(i,j-4500,1) = ssim(double(rgb2gray(img_ref)),double(rgb2gray(img_Fov_Texture)));
			SSIM(i,j-4500,2) = ssim(double(rgb2gray(img_ref)),double(rgb2gray(img_Uniform)));
			SSIM(i,j-4500,3) = ssim(double(rgb2gray(img_ref)),double(rgb2gray(img_Fov_Blur)));
			SSIM(i,j-4500,4) = ssim(double(rgb2gray(img_ref)),double(rgb2gray(img_ref)));

			%% SSIM check:
			[score map] = ssim_index(img_ref,img_Fov_Texture);
			SSIM_check(i,j-4500,1) = score;
			[score map] = ssim_index(img_ref,img_Uniform);
			SSIM_check(i,j-4500,2) = score;
			[score map] = ssim_index(img_ref,img_Fov_Blur);
			SSIM_check(i,j-4500,3) = score;
			[score map] = ssim_index(img_ref,img_ref);
			SSIM_check(i,j-4500,4) = score;

			%% SSIM Gray check:
			[score map] = ssim_index(rgb2gray(img_ref),rgb2gray(img_Fov_Texture));
			SSIM_check_gray(i,j-4500,1) = score;
			[score map] = ssim_index(rgb2gray(img_ref),rgb2gray(img_Uniform));
			SSIM_check_gray(i,j-4500,2) = score;
			[score map] = ssim_index(rgb2gray(img_ref),rgb2gray(img_Fov_Blur));
			SSIM_check_gray(i,j-4500,3) = score;
			[score map] = ssim_index(rgb2gray(img_ref),rgb2gray(img_ref));
			SSIM_check_gray(i,j-4500,4) = score;

		end

		MSE_Color(i,j-4500,1) = mean2((double(img_ref(:))-double(img_Fov_Texture(:))).^2);
		MSE_Color(i,j-4500,2) = mean2((double(img_ref(:))-double(img_Uniform(:))).^2);
		MSE_Color(i,j-4500,3) = mean2((double(img_ref(:))-double(img_Fov_Blur(:))).^2);
		MSE_Color(i,j-4500,4) = mean2((double(img_ref(:))-double(img_ref(:))).^2);

		MI_Color(i,j-4500,1) = mi(img_ref,img_Fov_Texture);
		MI_Color(i,j-4500,2) = mi(img_ref,img_Uniform);
		MI_Color(i,j-4500,3) = mi(img_ref,img_Fov_Blur);
		MI_Color(i,j-4500,4) = mi(img_ref,img_ref);

	end
	disp(i);
end

save('MSE_Color.mat','MSE_Color');
save('MI_Color.mat','MI_Color');


if check_flag == 0
	save('SSIM_check_gray.mat','SSIM_check_gray');
	save('SSIM_check.mat','SSIM_check');
	save('MSE.mat','MSE');
	save('MI.mat','MI');
	save('SSIM.mat','SSIM');

	MSE_validation(:,:,1) = MSE(:,1:250,1);
	MSE_validation(:,:,2) = MSE(:,1:250,2);
	MSE_validation(:,:,3) = MSE(:,1:250,3);
	MSE_validation(:,:,4) = MSE(:,1:250,4);

	MSE_testing(:,:,1) = MSE(:,251:500,1);
	MSE_testing(:,:,2) = MSE(:,251:500,2);
	MSE_testing(:,:,3) = MSE(:,251:500,3);
	MSE_testing(:,:,4) = MSE(:,251:500,4);

	MI_validation(:,:,1) = MI(:,1:250,1);
	MI_validation(:,:,2) = MI(:,1:250,2);
	MI_validation(:,:,3) = MI(:,1:250,3);
	MI_validation(:,:,4) = MI(:,1:250,4);

	MI_testing(:,:,1) = MI(:,251:500,1);
	MI_testing(:,:,2) = MI(:,251:500,2);
	MI_testing(:,:,3) = MI(:,251:500,3);
	MI_testing(:,:,4) = MI(:,251:500,4);

	SSIM_validation(:,:,1) = SSIM(:,1:250,1);
	SSIM_validation(:,:,2) = SSIM(:,1:250,2);
	SSIM_validation(:,:,3) = SSIM(:,1:250,3);
	SSIM_validation(:,:,4) = SSIM(:,1:250,4);

	SSIM_testing(:,:,1) = SSIM(:,251:500,1);
	SSIM_testing(:,:,2) = SSIM(:,251:500,2);
	SSIM_testing(:,:,3) = SSIM(:,251:500,3);
	SSIM_testing(:,:,4) = SSIM(:,251:500,4);

end






