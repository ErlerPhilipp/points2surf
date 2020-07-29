# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#train_}

python -m pdb -c c full_train.py \
    --name ${NAME}  \
    --desc ${NAME}  \
    --indir '/scratch_space/point2surf/data/implicit_surf_14'  \
    --outdir '../data/results'  \
    --logdir '../data/logs' \
    --trainset 'trainset.txt'  \
    --testset 'valset.txt'  \
    --nepoch 150  \
    --lr 0.01  \
    --debug 0  \
    --workers 7  \
    --batchSize 501  \
    --points_per_patch 300  \
    --patches_per_shape 1000  \
    --sub_sample_size 1000  \
    --cache_capacity 1000  \
    --patch_radius 0.0  \
    --single_transformer 0  \
    --shared_transformer \
    --patch_center 'mean'  \
    --training_order 'random_shape_consecutive'  \
    --outputs 'imp_surf_magnitude' 'imp_surf_sign' 'patch_pts_ids' 'p_index' \
