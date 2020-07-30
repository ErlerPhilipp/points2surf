# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#eval_}

python full_eval.py \
    --indir 'datasets' \
    --outdir 'results' \
    --modeldir 'models' \
    --dataset 'abc/testset.txt' 'abc_extra_noisy/testset.txt' 'abc_noisefree/testset.txt' 'real_world/testset.txt' 'famous_original/testset.txt' 'famous_noisefree/testset.txt' 'famous_sparse/testset.txt' 'famous_dense/testset.txt' 'famous_extra_noisy/testset.txt' 'thingi10k_scans_original/testset.txt' 'thingi10k_scans_noisefree/testset.txt' 'thingi10k_scans_sparse/testset.txt' 'thingi10k_scans_dense/testset.txt' 'thingi10k_scans_extra_noisy/testset.txt'\
    --models ${NAME} \
    --modelpostfix '_model_149.pth' \
    --batchSize 501 \
    --workers 7 \
    --cache_capacity 5 \
    --query_grid_resolution 256 \
    --epsilon 3 \
    --certainty_threshold 13 \
    --sigma 5 \
