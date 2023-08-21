%%
% Compute features for a set of video files from datasets
% 
close all; 
clear;

% add path
addpath(genpath('/mnt/storage/home/um20242/scratch/RAPIQUE-UGC/include/'));
% setenv('PATH', [getenv('PATH') '/Library/Frameworks/Python.framework/Versions/3.8/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Applications/VMware Fusion.app/Contents/Public:/Library/Apple/usr/bin:/Applications/Wireshark.app/Contents/MacOS']);

%%
% parameters
algo_name = 'RAPIQUE'; % algorithm name, eg, 'V-BLIINDS'
data_name = 'YOUTUBE_UGC_2160P';  % dataset name, eg, 'KONVID_1K'
write_file = true;  % if true, save features on-the-fly
log_level = 0;  % 1=verbose, 0=quite

if strcmp(data_name, 'YOUTUBE_UGC')
    root_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/';
    data_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/';
elseif strcmp(data_name, 'YOUTUBE_UGC_360P')
    root_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/';
    data_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/360P';
elseif strcmp(data_name, 'YOUTUBE_UGC_480P')
    root_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/';
    data_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/480P';
elseif strcmp(data_name, 'YOUTUBE_UGC_720P')
    root_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/';
    data_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/720P';
elseif strcmp(data_name, 'YOUTUBE_UGC_1080P')
    root_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/';
    data_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/1080P';
elseif strcmp(data_name, 'YOUTUBE_UGC_2160P')
    root_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/';
    data_path = '/mnt/storage/home/um20242/scratch/dataset/ugc-dataset/2160P';
elseif strcmp(data_name, 'KONVID_1K')
    root_path = '/mnt/storage/home/um20242/scratch/KoNViD_1k/';
    data_path = '/mnt/storage/home/um20242/scratch/KoNViD_1k/KoNViD_1k_videos';
end

%%
% create temp dir to store decoded videos
video_tmp = '/mnt/storage/home/um20242/scratch/RAPIQUE-UGC/tmp/';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '/mnt/storage/home/um20242/scratch/RAPIQUE-UGC/mos_files/';
filelist_csv = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(filelist_csv);
num_videos = size(filelist,1);
out_path = '/mnt/storage/home/um20242/scratch/RAPIQUE-UGC/feat_files/';
if ~exist(out_path, 'dir'), mkdir(out_path); end
out_mat_name = fullfile(out_path, [data_name,'_',algo_name,'_feats.mat']);
feats_mat = [];
feats_mat_frames = cell(num_videos, 1);
%===================================================

% init deep learning models
minside = 512.0;
net = resnet50;
net
layer = 'avg_pool';

%% extract features
% parfor i = 1:num_videos % for parallel speedup
for i = 1:num_videos
    progressbar(i/num_videos) % Update figure
    if strcmp(data_name, 'YOUTUBE_UGC')
        video_name = fullfile(data_path, filelist.category{i}, ...
            [num2str(filelist.resolution(i)),'P'],[filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
     elseif strcmp(data_name, 'YOUTUBE_UGC_360P')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_480P')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_720P')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_1080P')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_2160P')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_ALL')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'KONVID_1K')
        newStr = erase(filelist.flickr_id{i},'.mp4');
        video_name = fullfile(data_path, [newStr, '.mp4']);
        yuv_name = fullfile(video_tmp, [newStr, '.yuv']);
    end
    fprintf('\n\nComputing features for %d sequence: %s\n', i, video_name);

    % decode video and store in temp dir
    cmd = ['ffmpeg -loglevel error -y -i ', video_name, ...
        ' -pix_fmt yuv420p -vsync 0 ', yuv_name];
    system(cmd);  

    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));

    % calculate video features
    tStart = tic;
    feats_frames = calc_RAPIQUE_features(yuv_name, width, height, ...
        framerate, minside, net, layer, log_level);
    fprintf('\nOverall %f seconds elapsed...', toc(tStart));
    % 
    feats_mat(i,:) = nanmean(feats_frames);
    feats_mat_frames{i} = feats_frames;
    % clear cache
    delete(yuv_name)

    if write_file
        save(out_mat_name, 'feats_mat');
%         save(out_mat_name, 'feats_mat', 'feats_mat_frames');
    end
end




