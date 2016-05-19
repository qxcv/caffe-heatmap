% This file uses a FLIC trained model and applies it to a video sequence from
% Poses in the Wild. Unlike demo.m, it applies it to *all of* Poses in the Wild
% and outputs predictions in a format compatible with my evaluation code.
%
% Download the model:
%    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/

%% Options

opt.visualise = false;		% Visualise predictions?
opt.useGPU = true; 			% Run on GPU
opt.dim = 256;
opt.dims = [opt.dim, opt.dim]; 		% Input dimensions (needs to match matlab.txt)
opt.numJoints = 7; 			% Number of joints
opt.layerName = 'conv5_fusion'; % Output layer name
opt.modelDefFile = '../../models/heatmap-flic-fusion/matlab.prototxt'; % Model definition
opt.modelFile = '../../models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel'; % Model weights
opt.cache_dir = 'cache/piw/';
opt.result_dir = 'results/piw/';
opt.inputDir = ''; % this isn't used any more
% First joint here does not reflect CNN output well. I'm not evaluating on
% it, though, so it shouldn't matter.
opt.trans_spec = struct(...
    'indices', {...
        1,     ... Top of body somewhere    #1
        7,     ... Right wrist              #2
        4,     ... Left wrist               #3
        6,     ... Right elbow              #4
        3,     ... Left elbow               #5
        5,     ... Right shoulder           #6
        2,     ... Left shoulder            #7
    }, ...
    'weights', {1, 1, 1, 1, 1, 1, 1});

%% Dataset reading
if ~exist(opt.cache_dir, 'dir'), mkdir(opt.cache_dir), end
test_seqs = get_piw(opt.cache_dir, opt.trans_spec);
% 'files' is a cell array of images for applyNet. trans_rec is a data
% structure which can be used to recover poses (in original coordinates)
% from returned joints.
[files, trans_rec] = prepare_test_seqs(test_seqs, opt.dim, opt.cache_dir);

%% Forward prop
% Add caffe matlab into path
addpath('../');
% Apply network
joints = applyNet(files, opt);

%% Pose recovery
results = recover_seq_preds(joints, trans_rec);
fprintf('Got predictions for %i sequences (%i frames total)\n', ...
    length(results), sum(cellfun(@length, results)));
mkdir_p(opt.result_dir);
save(fullfile(opt.result_dir, 'flow-convnets-piw.mat'), 'results', 'test_seqs');
