# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#eval_}

python -m pdb -c c full_eval.py \
    --indir '/scratch_space/point2surf/data'  \
    --outdir '../data/results' \
    --modeldir '../data/models' \
    --dataset 'test_original/testset.txt' 'test_noisefree/testset.txt' 'test_sparse/testset.txt' 'test_dense/testset.txt' 'test_extra_noisy/testset.txt' 'implicit_surf_14/testset.txt' \
    --models ${NAME} \
    --modelpostfix '_model_99.pth' \
    --batchSize 100 \
    --workers 7 \
    --cache_capacity 5 \
    --patch_features 'imp_surf_magnitude' 'imp_surf_sign' 'patch_pts_ids' 'p_index' \
    --query_grid_resolution 256 \
    --epsilon 3 \
    --certainty_threshold 13 \
    --sigma 5 \
