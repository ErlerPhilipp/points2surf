# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#eval_}

python full_eval.py \
    --indir 'datasets' \
    --outdir 'results' \
    --modeldir 'models' \
    --dataset 'implicit_surf_14/testset.txt' \
    --models ${NAME} \
    --modelpostfix '_model_49.pth' \
    --batchSize 1501 \
    --workers 19 \
    --cache_capacity 5 \
    --patch_features 'imp_surf_magnitude' 'imp_surf_sign' 'patch_pts_ids' 'p_index' \
    --query_grid_resolution 256 \
    --epsilon 3 \
    --certainty_threshold 13 \
    --sigma 5 \
