clear all
close all
clc

run './eidors-v3.10-ng/eidors/startup.m'
%% ROI Calculations
%%%% Please visit "https://github.com/DiogoMPessoa/Dimensionless-Respiratory-Airflow-Estimation/blob/main/EIT%20reconstruction/SampleDataVizualization.m"
%%%% Also check "http://eidors3d.sourceforge.net/tutorial/netgen/extrusion/thoraxmdl.shtml"
size_image=32;
ROIS_masks=struct(); 
Total_image=1:size_image*size_image;
Out_lung=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,59,60,61,62,63,64,65,66,67,68,69,70,71,92,93,94,95,96,97,98,99,100,101,125,126,127,128,129,130,131,132,158,159,160,161,162,163,190,191,192,193,194,223,224,225,255,256,257,288,320,513,545,577,704,736,768,769,799,800,801,831,832,833,834,862,863,864,865,866,867,893,894,895,896,897,898,899,900,901,925,926,927,928,929,930,931,932,933,934,935,956,957,958,959,960,961,962,963,964,965,966,967,968,969,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024];
In_lung=setdiff(Total_image,Out_lung);

ROIS_masks.Out_lung=Out_lung;
ROIS_masks.In_lung=In_lung;
ROIS_masks.Global_ROI=1:size_image*size_image;


left_ROI=[size_image*(size_image/2)+1:(size_image*size_image)];
right_ROI=[1:size_image*(size_image/2)];
anterior_ROI=[];
for i=1:size_image
    anterior_ROI=[anterior_ROI,[1:(size_image/2)]+((i-1)*size_image)];
end
posterior_ROI=[];
for i=1:size_image
    posterior_ROI=[posterior_ROI,[(size_image/2)+1:size_image]+((i-1)*size_image)];
end
ROIS_masks.right_ROI=intersect(right_ROI,In_lung);
ROIS_masks.left_ROI=intersect(left_ROI,In_lung);
ROIS_masks.anterior_ROI=intersect(anterior_ROI,In_lung);
ROIS_masks.posterior_ROI=intersect(posterior_ROI,In_lung);

quadrant1_ROI=[];
for i=1:size_image/2
    quadrant1_ROI=[quadrant1_ROI,[1:(size_image/2)]+((i-1)*size_image)];
end
quadrant2_ROI=[];
for i=1:size_image/2
    quadrant2_ROI=[quadrant2_ROI,[(size_image/2)+1:size_image]+((i-1)*size_image)];
end
quadrant3_ROI=[];
for i=(size_image/2)+1:size_image
    quadrant3_ROI=[quadrant3_ROI,[1:(size_image/2)]+((i-1)*size_image)];
end
quadrant4_ROI=[];
for i=(size_image/2)+1:size_image
    quadrant4_ROI=[quadrant4_ROI,[(size_image/2)+1:size_image]+((i-1)*size_image)];
end
ROIS_masks.quadrant1_ROI=intersect(quadrant1_ROI,In_lung);
ROIS_masks.quadrant2_ROI=intersect(quadrant2_ROI,In_lung);
ROIS_masks.quadrant3_ROI=intersect(quadrant3_ROI,In_lung);
ROIS_masks.quadrant4_ROI=intersect(quadrant4_ROI,In_lung);
heights=[8];
for h=1:length(heights)
    height=heights(h);
    for r=1:(size_image/height)
        eval(strcat('horizontal',num2str(r),'_ROI_size',num2str(height),'=[];'))
        for i=(r-1)*height+1:r*height
            new_line=i:size_image:size_image*size_image;
            eval(strcat('horizontal',num2str(r),'_ROI_size',num2str(height),'=[horizontal',num2str(r),'_ROI_size',num2str(height),',new_line];'))
        end
                
        eval(strcat('ROIS_masks.horizontal',num2str(r),'_ROI_size',num2str(height),'=intersect(horizontal',num2str(r),'_ROI_size',num2str(height),',In_lung);'))%intersect with inner area of the EIDORS model
        eval(strcat('ROIS_masks.horizontal',num2str(r),'_ROI_size',num2str(height),'_right_half=intersect(horizontal',num2str(r),'_ROI_size',num2str(height),',ROIS_masks.right_ROI);'))%intersect with right lung of the EIDORS model
        eval(strcat('ROIS_masks.horizontal',num2str(r),'_ROI_size',num2str(height),'_left_half=intersect(horizontal',num2str(r),'_ROI_size',num2str(height),',ROIS_masks.left_ROI);'))%intersect with left lung of the EIDORS model
    end
end

widths=[8];
for w=1:length(widths)
    width=widths(w);
    for r=1:(size_image/width)
        eval(strcat('vertical',num2str(r),'_ROI_size',num2str(width),'=[];'))
        for i=(r-1)*width+1:r*width
            new_col=(i-1)*size_image+1:((i-1)*size_image+size_image);
            eval(strcat('vertical',num2str(r),'_ROI_size',num2str(width),'=[vertical',num2str(r),'_ROI_size',num2str(width),', new_col];'))
        end        
        eval(strcat('ROIS_masks.vertical',num2str(r),'_ROI_size',num2str(width),'=intersect(vertical',num2str(r),'_ROI_size',num2str(width),',In_lung);'))%intersect with inner area of the EIDORS model
        eval(strcat('ROIS_masks.vertical',num2str(r),'_ROI_size',num2str(width),'_anterior_half=intersect(vertical',num2str(r),'_ROI_size',num2str(width),',ROIS_masks.anterior_ROI);'))%intersect with right lung of the EIDORS model
        eval(strcat('ROIS_masks.vertical',num2str(r),'_ROI_size',num2str(width),'_posterior_half=intersect(vertical',num2str(r),'_ROI_size',num2str(width),',ROIS_masks.posterior_ROI);'))%intersect with left lung of the EIDORS model
          
    end
end

ROIS_masks_linearIndex=ROIS_masks;

ROIS_names=fieldnames(ROIS_masks);

ROIS_masks_coordinates=struct();
for n=1:length(ROIS_names)
    sz = [size_image size_image];
    eval(strcat('[rows,cols]=ind2sub(sz,ROIS_masks.',ROIS_names{n},');'));
    eval(strcat('ROIS_masks_coordinates.',ROIS_names{n},'=[transpose(rows),transpose(cols)];'));
end

%% 

for No=1:9
root_dir = '\dataset';
first = ["0",int2str(No)];
subject_id=join(first,"");
eit_path = fullfile(root_dir, subject_id, 'EIT');

filenames = {};

files = dir(fullfile(eit_path, '*'));

for jj = 1:length(files)

  filename = files(jj).name;

  if strfind(filename, '.eit') > 0
    filenames{end + 1} = fullfile(eit_path, filename);
  end
  for i = 1:length(filenames)

     a= char(filenames{i});
    [voltage_data, ~, ~] = eidors_readdata(a);

fmdl = mk_library_model('adult_male_16el_lungs');
[fmdl.stimulation, fmdl.meas_select] = mk_stim_patterns(16,1,'{ad}','{ad}',{'no_meas_current','rotate_meas'}, .005);%'goeiimf-eit' stimulation pattern
fmdl = mdl_normalize(fmdl,1);
  %img = mk_lung_image(fmdl);


  


    opt = mk_lung_image(fmdl, 'options');   % get default options
   corr = 20;
  opt.heart_center(2)        = opt.heart_center(2)      + corr;
    opt.left_lung_center(2)    = opt.left_lung_center(2)  + corr;
  opt.right_lung_center(2)   = opt.right_lung_center(2) + corr;
   opt.diaphragm_center(2)    = opt.diaphragm_center(2)  + corr;

img = mk_lung_image(fmdl, opt);
img.calc_colours.ref_level=0;

opt2.imgsz = [32 32];
opt2.distr = 3; 
opt2.Nsim = 1000; 
opt2.target_size = 0.03; 
opt2.target_offset = 0;
opt2.noise_figure = .5; 
opt2.square_pixels = 1;
imdl = mk_GREIT_model(img, 0.25, [], opt2);

imgall = inv_solve(imdl,mean(voltage_data(:,1:end),2),voltage_data); 



image_slices_all_conductivity = show_slices(imgall);
close all
images_slices_separated = -calc_slices(imgall);   
images_slices_separated(isnan(images_slices_separated(:)))= 0;




fid= fopen(a,'rb');
d= fread(fid,[1 1021],'uchar');
str= char(d(3:2:end));

expression = '<(\w+).*>.*</\1>';
[~,matches] = regexp(str,expression,'tokens','match');
metadata.Fs=str2double(cell2mat(regexp(matches{9},'\d+\.?\d*','Match')));
metadata.Frames=str2double(cell2mat(regexp(matches{10},'\d+\.?\d*','Match')));
metadata.Duration = str2double(cell2mat(regexp(matches{11},'\d+\.?\d*','Match')));
metadata.BeginDate= datetime(cell2mat(regexp(matches{8},'\d+\.?\d*','Match')),'InputFormat','yyyyMMddHHmmss');

fclose(fid);



%eitFile_metadata = eit_metadata(a);
if(size(voltage_data,2)~=metadata.Frames)
    metadata.Frames = size(voltage_data,2);
end

if ~exist('num_cols','var')
    num_cols = 40;
end
number_pixels=32;

num_frames=size(images_slices_separated,3);

num_lines=ceil(num_frames/num_cols);

allImages=NaN(num_lines*number_pixels,num_cols*number_pixels);
for f=1:num_frames
    frame=images_slices_separated(:,:,f);
    frame(ROIS_masks_linearIndex.Out_lung)=NaN;
    
    line=ceil(f/num_cols);
    col=rem(f,num_cols);
    if(col==0)
        col=num_cols;
    end
    
    idx_line=((line-1)*number_pixels)+1;
    idx_col=((col-1)*number_pixels)+1;
    
   allImages(idx_line:(idx_line+(number_pixels-1)),idx_col:(idx_col+(number_pixels-1)))=frame;
end

dataSize = size(allImages);
patchSize = [32 32];

newData = []; 

rows = 1:patchSize(1):dataSize(1) - patchSize(1) + 1;
cols = 1:patchSize(2):dataSize(2) - patchSize(2) + 1;

for i = 1:length(rows)
    for j = 1:length(cols)
        patch = allImages(rows(i):rows(i)+patchSize(1)-1, cols(j):cols(j)+patchSize(2)-1);
        if ~all(all(isnan(patch))) 
            newData = [newData; patch];
        end
    end
end

allImages = newData; 

     original_string = filename;
    reduced_string=original_string(1:end-3);

    filename_template = strcat(reduced_string, 'csv');

     writematrix(allImages, filename_template);

end
end
end



for No=10:78
root_dir = 'dataset';
first = [int2str(No)];
subject_id=join(first,"");
eit_path = fullfile(root_dir, subject_id, 'EIT');

filenames = {};

files = dir(fullfile(eit_path, '*'));

for jj = 1:length(files)
  filename = files(jj).name;
  
  if strfind(filename, '.eit') > 0
    filenames{end + 1} = fullfile(eit_path, filename);
  end
  for i = 1:length(filenames)



        a= char(filenames{i});
    [voltage_data, ~, ~] = eidors_readdata(a);

fmdl = mk_library_model('adult_male_16el_lungs');
[fmdl.stimulation, fmdl.meas_select] = mk_stim_patterns(16,1,'{ad}','{ad}',{'no_meas_current','rotate_meas'}, .005);%'goeiimf-eit' stimulation pattern
fmdl = mdl_normalize(fmdl,1);
  %img = mk_lung_image(fmdl);


  


    opt = mk_lung_image(fmdl, 'options');   % get default options
   corr = 20;
  opt.heart_center(2)        = opt.heart_center(2)      + corr;
    opt.left_lung_center(2)    = opt.left_lung_center(2)  + corr;
  opt.right_lung_center(2)   = opt.right_lung_center(2) + corr;
   opt.diaphragm_center(2)    = opt.diaphragm_center(2)  + corr;

img = mk_lung_image(fmdl, opt);
img.calc_colours.ref_level=0;

opt2.imgsz = [32 32];
opt2.distr = 3; 
opt2.Nsim = 1000; 
opt2.target_size = 0.03; 
opt2.target_offset = 0;
opt2.noise_figure = .5; 
opt2.square_pixels = 1;
imdl = mk_GREIT_model(img, 0.25, [], opt2);

imgall = inv_solve(imdl,mean(voltage_data(:,1:end),2),voltage_data); 



image_slices_all_conductivity = show_slices(imgall);
close all
images_slices_separated = -calc_slices(imgall);   
images_slices_separated(isnan(images_slices_separated(:)))= 0;

fid= fopen(a,'rb');
d= fread(fid,[1 1021],'uchar');
str= char(d(3:2:end));

expression = '<(\w+).*>.*</\1>';
[~,matches] = regexp(str,expression,'tokens','match');
metadata.Fs=str2double(cell2mat(regexp(matches{9},'\d+\.?\d*','Match')));
metadata.Frames=str2double(cell2mat(regexp(matches{10},'\d+\.?\d*','Match')));
metadata.Duration = str2double(cell2mat(regexp(matches{11},'\d+\.?\d*','Match')));
metadata.BeginDate= datetime(cell2mat(regexp(matches{8},'\d+\.?\d*','Match')),'InputFormat','yyyyMMddHHmmss');

fclose(fid);
if(size(voltage_data,2)~=metadata.Frames)
    metadata.Frames = size(voltage_data,2);
end

if ~exist('num_cols','var')
    num_cols = 40;
end
number_pixels=32;

num_frames=size(images_slices_separated,3);

num_lines=ceil(num_frames/num_cols);

allImages=NaN(num_lines*number_pixels,num_cols*number_pixels);
for f=1:num_frames
    frame=images_slices_separated(:,:,f);
    frame(ROIS_masks_linearIndex.Out_lung)=NaN;
    
    line=ceil(f/num_cols);
    col=rem(f,num_cols);
    if(col==0)
        col=num_cols;
    end
    
    idx_line=((line-1)*number_pixels)+1;
    idx_col=((col-1)*number_pixels)+1;
    
   allImages(idx_line:(idx_line+(number_pixels-1)),idx_col:(idx_col+(number_pixels-1)))=frame;
end

dataSize = size(allImages);
patchSize = [32 32];

newData = []; 

rows = 1:patchSize(1):dataSize(1) - patchSize(1) + 1;
cols = 1:patchSize(2):dataSize(2) - patchSize(2) + 1;

for i = 1:length(rows)
    for j = 1:length(cols)
        patch = allImages(rows(i):rows(i)+patchSize(1)-1, cols(j):cols(j)+patchSize(2)-1);
        if ~all(all(isnan(patch))) 
            newData = [newData; patch];
        end
    end
end

allImages = newData; 

     original_string = filename;
     reduced_string=original_string(1:end-3);

    filename_template = strcat(reduced_string, 'csv');

     writematrix(allImages, filename_template);

end
end
end


