% This file uses a FLIC trained model and applies it to a video sequence from
% MPII Cooking Activities.
%
% Download the model:
%    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/

%% Options

opt.visualise = false;		% Visualise predictions?
opt.headless_visualise = true;
opt.visualise_path = 'mpii-vis/';
opt.useGPU = true; 			% Run on GPU
opt.dim = 256;
opt.dims = [opt.dim, opt.dim]; 		% Input dimensions (needs to match matlab.txt)
opt.numJoints = 7; 			% Number of joints
opt.layerName = 'conv5_fusion'; % Output layer name
opt.modelDefFile = '../../models/heatmap-flic-fusion/matlab.prototxt'; % Model definition
opt.modelFile = '../../models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel'; % Model weights
opt.cache_dir = 'cache/mpii/';
opt.result_dir = 'results/mpii/';
opt.inputDir = ''; % this isn't used any more

%% Dataset reading
if ~exist(opt.cache_dir, 'dir'), mkdir(opt.cache_dir), end
test_seqs = get_mpii_cooking(opt.cache_dir);
% 'files' is a cell array of images for applyNet. trans_rec is a data
% structure which can be used to recover poses (in original coordinates)
% from returned joints.
[files, trans_rec] = prepare_test_seqs(test_seqs, opt.dim, ...
    opt.cache_dir, 'seqHeight');

%% Forward prop
% Add caffe matlab into path
addpath('../');
% Apply network
joints = applyNet(files, opt);

%% Pose recovery
raw_results = recover_seq_preds(joints, trans_rec);
fprintf('Got predictions for %i sequences (%i frames total)\n', ...
    length(raw_results), sum(cellfun(@length, raw_results)));
% MPII skeleton is:
% torso upper point (1), torso lower point (2), right shoulder (3), left
% shoulder (4), right elbow (5), left elbow (6), right wrist (7), left
% wrist (8), right hand (9), left hand (10), head upper point (11), head
% lower point (12).
mpii_transback = @(p) [nan([2 2]); p([6 7 4 5 2 3], :); nan([4 2])];
seq_transback = @(s) cellfun(mpii_transback, s, 'UniformOutput', false);
results = cellfun(seq_transback, raw_results, 'UniformOutput', false);
mkdir_p(opt.result_dir);
save(fullfile(opt.result_dir, 'flow-convnets-mpii.mat'), ...
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
