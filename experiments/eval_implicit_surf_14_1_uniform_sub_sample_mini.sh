# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#eval_}

python full_eval.py \
    --indir 'datasets' \
    --outdir 'results' \
    --modeldir 'models' \
    --dataset 'test_original/testset.txt' 'test_noisefree/testset.txt' 'test_sparse/testset.txt' 'test_dense/testset.txt' 'test_extra_noisy/testset.txt' 'test_real_world/testset.txt' 'implicit_surf_14/testset.txt' 'implicit_surf_14_extra_noisy/testset.txt' 'implicit_surf_14_noisefree/testset.txt' 'thingi10k_scans_original/testset.txt' 'thingi10k_scans_noisefree/testset.txt' 'thingi10k_scans_sparse/testset.txt' 'thingi10k_scans_dense/testset.txt' 'thingi10k_scans_extra_noisy/testset.txt'\
    --models ${NAME} \
    --batchSize 5001 \
    --workers 19 \
    --cache_capacity 5 \
    --patch_features 'imp_surf_magnitude' 'imp_surf_sign' 'patch_pts_ids' 'p_index' \
    --query_grid_resolution 256 \
    --epsilon 3 \
    --certainty_threshold 13 \
    --sigma 5 \
