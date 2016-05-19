% This file uses a FLIC trained model and applies it to a video sequence from Poses in the Wild
%
% Download the model:
%    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/

% Options

opt.visualise = false;		% Visualise predictions?
opt.useGPU = true; 			% Run on GPU
opt.dims = [256 256]; 		% Input dimensions (needs to match matlab.txt)
opt.numJoints = 7; 			% Number of joints
% Here are the joints (in order):
% 1) Nose
% 2) Right wrist
% 3) Left wrist
% 4) Right elbow
% 5) Left elbow
% 6) Right shoulder
% 7) Left shoulder
opt.layerName = 'conv5_fusion'; % Output layer name
opt.modelDefFile = '../../models/heatmap-flic-fusion/matlab.prototxt'; % Model definition
opt.modelFile = '../../models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel'; % Model weights

% Add caffe matlab into path
addpath('../')

% Image input directory
opt.inputDir = 'sample_images/';

% Create image file list
imInds = 1:29;
for ind = 1:numel(imInds); files{ind} = [num2str(imInds(ind)) '.png']; end

% Apply network
joints = applyNet(files, opt)
