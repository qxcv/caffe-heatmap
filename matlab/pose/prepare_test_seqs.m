function [files, trans_rec] = prepare_test_seqs(test_seqs, dim, cache_dir, vstrat)
%PREPARE_TEST_SEQS Prepare test image sequence for flowing convnet code
result_dir = fullfile(cache_dir, 'frames/');
mkdir_p(result_dir);

trans_rec = cell([1 length(test_seqs.seqs)]);
files = {};

skipped = 0;

for seq_idx=1:length(test_seqs.seqs)
    seq = test_seqs.seqs{seq_idx};
    empty_cells = cell([1 length(seq)]);
    this_trans =  struct('frame_path', empty_cells, ...
        'scale', empty_cells, 'xtrim', empty_cells, ...
        'ytrim', empty_cells, 'frame_num', empty_cells);
    
    % Find bbox for sequence
    joint_locs = cat(1, test_seqs.data(seq).joint_locs);
    seq_bbox = round(get_bbox(joint_locs));
    pad_amount = [64 64];
    seq_bbox(1:2) = seq_bbox(1:2) - pad_amount;
    seq_bbox(3:4) = seq_bbox(3:4) + 2 * pad_amount;
    
    for frame_idx=1:length(seq)
        frame_ident = seq(frame_idx);
        datum = test_seqs.data(frame_ident);
        bounds = round(get_bbox(datum.joint_locs));
        % Order should be "x, y, w, h"
        this_bbox = [bounds(1) - pad_amount(1), seq_bbox(2), ...
            bounds(3) + 2 * pad_amount(1), seq_bbox(4)];
        
        % vstrat specifies a strategy for vertical cropping; horizontal
        % cropping is always based on joint location and is unaffected by
        % vstrat.
        if strcmp(vstrat, 'fullHeight')
            % We crop so that the entire vertical range of the frame is
            % included
            this_bbox(2) = 1;
            this_bbox(4) = size(readim(datum), 1);
        else
            % Only crop so that maximum vertical extent of pose across
            % sequence (plus some padding) is included
            assert(strcmp(vstrat, 'seqHeight'), 'vstrat must be fullHeight or seqHeight');
        end
        
        % Where to write frame
        frame_num = length(files) + 1;
        frame_path = fullfile(result_dir, sprintf('frame-d%04i-[s%i-f%i].png', frame_ident, seq_idx, frame_idx));
        files{frame_num} = frame_path; %#ok<AGROW>
        
        % Back-translation data
        info.frame_path = frame_path;
        [im, info.scale, info.xtrim, info.ytrim] ...
            = transform_datum(datum, dim, this_bbox);
        if ~exist(frame_path, 'file')
            imwrite(im, frame_path);
        else
            skipped = skipped + 1;
        end
        info.frame_num = frame_num;
        this_trans(frame_idx) = info;
    end
    
    trans_rec{seq_idx} = this_trans;
end

fprintf('Skipped writing %i files\n', skipped);
end

function [im, scale, xtrim, ytrim] = transform_datum(datum, dim, bbox)
im = readim(datum);
assert(numel(bbox) == 4);
xtrim = bbox(1) - 1;
ytrim = bbox(2) - 1;

% Crop box
im = impcrop(im, bbox);

% Rescale so that image is no more than dims in either dimension (and
% exactly dims in at least one dimension)
w = size(im, 2); h = size(im, 1);
if w > h
    % Wide picture
    scaled_size = [round(h * 256 / w), 256];
else
    % Tall picture
    scaled_size = [256, round(w * 256 / h)];
end
scale = scaled_size / [h w];
im = imresize(im, scaled_size);
w = size(im, 2); h = size(im, 1);
assert(h <= dim && w <= dim);
assert(h == dim || w == dim);

% Pad out to dims with black
before_pad_x = floor((dim - w) / 2);
after_pad_x = (dim - w) - before_pad_x;
before_pad_y = floor((dim - h) / 2);
after_pad_y = (dim - h) - before_pad_y;
im = padarray(im, [before_pad_y, before_pad_x], 0, 'pre');
im = padarray(im, [after_pad_y, after_pad_x], 0, 'post');
xtrim = xtrim - before_pad_x;
ytrim = ytrim - before_pad_y;

w = size(im, 2); h = size(im, 1);
assert(h == dim && w == dim);
end
