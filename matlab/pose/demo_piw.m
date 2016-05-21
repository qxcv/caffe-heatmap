% This file uses a FLIC trained model and applies it to a video sequence from
% Poses in the Wild. Unlike demo.m, it applies it to *all of* Poses in the Wild
% and outputs predictions in a format compatible with my evaluation code.
%
% Download the model:
%    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/

%% Options

opt.visualise = false;		% Visualise predictions?
opt.headless_visualise = true;
opt.visualise_path = 'piw-vis/';
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
[files, trans_rec] = prepare_test_seqs(test_seqs, opt.dim, ...
    opt.cache_dir, 'fullHeight');

%% Forward prop
% Add caffe matlab into path
addpath('../');
% Apply network
joints = applyNet(files, opt);

%% Pose recovery
raw_results = recover_seq_preds(joints, trans_rec);
fprintf('Got predictions for %i sequences (%i frames total)\n', ...
    length(raw_results), sum(cellfun(@length, raw_results)));
% PIW skeleton is:
% 1, 8: Neck (more like chin), torso
% 2, 3, 4: Left shoulder, elbw, wrist
% 5, 6, 7: Right shoulder, elbow, wrist
% We need to translate back into that format
piw_transback = @(p) [nan nan; p([7 5 3 6 4 2], :); nan nan];
seq_transback = @(s) cellfun(piw_transback, s, 'UniformOutput', false);
results = cellfun(seq_transback, raw_results, 'UniformOutput', false);
mkdir_p(opt.result_dir);
save(fullfile(opt.result_dir, 'flow-convnets-piw.mat'), ...
    'results', 'raw_results', 'test_seqs');

if opt.headless_visualise
    figure('Visible', 'off');
    axes('Visible', 'off');
    mkdir_p(opt.visualise_path);
    for fp_idx=1:length(files)
        fn = files{fp_idx};
        imagesc(imread(fn));
        hold on;
        plotSkeleton(joints(:, :, fp_idx), [], []);
        hold off;
        result_path = fullfile(opt.visualise_path, sprintf('frame-%i.jpg', fp_idx));
        fprintf('Writing to %s\n', result_path);
        print(gcf, '-djpeg', result_path, '-r 150');
    end
end
