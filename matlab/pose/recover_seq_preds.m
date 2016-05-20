function preds = recover_seq_preds(joints, trans_rec)
%RECOVER_SEQ_PREDS Get predictions in GT coordinates
num_seqs = length(trans_rec);
preds = cell([1 length(num_seqs)]);
for seq_idx=1:num_seqs
    seq_trans = trans_rec{seq_idx};
    num_frames = length(seq_trans);
    seq_preds = cell([1 num_frames]);
    for frame_idx=1:num_frames
        frame_trans = seq_trans(frame_idx);
        pred = joints(:, :, frame_trans.frame_num)';
        assert(ismatrix(pred) && size(pred, 2) == 2);
        scaled = (pred - 1) / frame_trans.scale + 1;
        unpadded = bsxfun(@plus, scaled, [frame_trans.xtrim, frame_trans.ytrim]);
        seq_preds{frame_idx} = unpadded;
    end
    preds{seq_idx} = seq_preds;
end
end
