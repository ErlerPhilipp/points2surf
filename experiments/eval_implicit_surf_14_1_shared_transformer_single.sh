# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#eval_}

res1=$(date +%s.%N)
echo "Start time: $res1"

#python full_eval.py \
#    --indir '/data/datasets/own/implicit_surf_14' \
#    --outdir 'results' \
#    --modeldir 'models' \
#    --dataset 'single.txt' \
#    --models ${NAME} \
#    --modelpostfix '_model_99.pth' \
#    --batchSize 100 \
#    --workers 7 \
#    --cache_capacity 5 \
#    --patch_features 'imp_surf_magnitude' 'imp_surf_sign' 'patch_pts_ids' 'p_index' \
#    --query_grid_resolution 256 \
#    --epsilon 3 \
#    --certainty_threshold 13 \
#    --sigma 5 \

res2=$(date +%s.%N)
echo "Start time: $res1"
echo "Stop time:  $res2"
echo "Elapsed:    $(echo "$res2 - $res1"|bc )"