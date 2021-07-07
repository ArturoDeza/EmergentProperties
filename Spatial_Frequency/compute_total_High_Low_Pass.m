function recompute_total_High_Low_Pass(z_start,z_end)

suffix_loader{1} = '_Reference-Net.mat';
suffix_loader{2} = '_Foveation-Texture-Net.mat';
suffix_loader{3} = '_Uniform-Net.mat';
suffix_loader{4} = '_Foveation-Blur-Net.mat';

values_low = [1,3,5,7,9,15,40];
values_high = [3,1.5,1,0.7,0.55,0.45,0.4];

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

System_Name{1} = 'Reference-Net';
System_Name{2} = 'Foveation-Texture-Net';
System_Name{3} = 'Uniform-Net';
System_Name{4} = 'Foveation-Blur-Net';

high_pass_color_flag = 1;
high_pass_gray_flag = 1;
low_pass_color_flag = 1;
low_pass_gray_flag = 1;

for z=z_start:z_end
	z_str = num2str(z);
	%
	mean_R = load(['./Avg_Color_Bias/mean_img_R' suffix_loader{z}]);
	mean_G = load(['./Avg_Color_Bias/mean_img_G' suffix_loader{z}]);
	mean_B = load(['./Avg_Color_Bias/mean_img_B' suffix_loader{z}]);
	%
	mean_R = mean_R.mean_img_R;
	mean_G = mean_G.mean_img_G;
	mean_B = mean_B.mean_img_B;
	%
	for i=1:20
		%for j=1:250
		i_str = num2str(i);
		for j=1:250
			img_id = 4750+j;
			img_indx_str = num2str(img_id);
			%
			img = imread(['../All_Testing_Images/' System_Name{z} '/' img_class{i} '/' img_indx_str '.png']);
			%
			for k=0:7
				k_plus_str = num2str(k+1);
				if k==0
					if high_pass_color_flag == 1
						imwrite(img,['./Testing_Images/High_Pass/' System_Name{z} '/' img_class{i} '/' k_plus_str '/' img_indx_str '.png']);
					end
					if low_pass_color_flag == 1
						imwrite(img,['./Testing_Images/Low_Pass/' System_Name{z} '/' img_class{i} '/' k_plus_str '/' img_indx_str '.png']);
					end
					if high_pass_gray_flag == 1
						imwrite(rgb2gray(img),['./Testing_Images/High_Pass_Gray/' System_Name{z} '/' img_class{i} '/' k_plus_str '/' img_indx_str '.png']);
					end
					if low_pass_gray_flag == 1
						imwrite(rgb2gray(img),['./Testing_Images/Low_Pass_Gray/' System_Name{z} '/' img_class{i} '/' k_plus_str '/' img_indx_str '.png']);
					end
				else
					%
					img1(:,:,1) = imgaussfilt(double(img(:,:,1)),values_low(k));
					img1(:,:,2) = imgaussfilt(double(img(:,:,2)),values_low(k));
					img1(:,:,3) = imgaussfilt(double(img(:,:,3)),values_low(k));
					%
					img_low = img1;
					%
					low_pass_buff(:,:,1) = imgaussfilt(double(img(:,:,1)),values_high(k));
					low_pass_buff(:,:,2) = imgaussfilt(double(img(:,:,2)),values_high(k));
					low_pass_buff(:,:,3) = imgaussfilt(double(img(:,:,3)),values_high(k));
					%
					img_high = double(double(img)-double(low_pass_buff));
					%
					%
					img_high_mean_1 = mean2(img_high(:,:,1));
					img_high_mean_2 = mean2(img_high(:,:,2));
					img_high_mean_3 = mean2(img_high(:,:,3));
					%
					img_high(:,:,1) = img_high(:,:,1) - img_high_mean_1 + 255.0*mean_R;
					img_high(:,:,2) = img_high(:,:,2) - img_high_mean_2 + 255.0*mean_G;
					img_high(:,:,3) = img_high(:,:,3) - img_high_mean_3 + 255.0*mean_B;
					%
					img_low_255 = uint8(img_low);
					img_high_255 = uint8(img_high);
					%
					%
					image_High = img_high_255;
					%
					image_Low = img_low_255;
					%
					image_High_Gray = rgb2gray(img_high_255);
					%
					image_Low_Gray = rgb2gray(img_low_255);
					%
					if high_pass_color_flag == 1
						imwrite(image_High,['./Testing_Images/High_Pass/' System_Name{z} '/' img_class{i} '/' k_plus_str '/' img_indx_str '.png']);
					end
					if low_pass_color_flag == 1
						imwrite(image_Low,['./Testing_Images/Low_Pass/' System_Name{z} '/' img_class{i} '/' k_plus_str '/' img_indx_str '.png']);
					end
					if high_pass_gray_flag == 1
						imwrite(image_High_Gray,['./Testing_Images/High_Pass_Gray/' System_Name{z} '/' img_class{i} '/' k_plus_str '/' img_indx_str '.png']);
					end
					if low_pass_gray_flag == 1
						imwrite(image_Low_Gray,['./Testing_Images/Low_Pass_Gray/' System_Name{z} '/' img_class{i} '/' k_plus_str '/' img_indx_str '.png']);
					end
				end
			end
		end
	end
end
