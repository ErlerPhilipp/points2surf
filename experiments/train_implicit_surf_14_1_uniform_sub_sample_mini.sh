# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#train_}

python full_train.py \
    --name ${NAME}  \
    --desc ${NAME}  \
    --indir 'datasets/implicit_surf_14'  \
    --outdir 'models'  \
    --logdir 'logs' \
    --trainset 'trainset.txt'  \
    --testset 'valset.txt'  \
    --nepoch 150  \
    --lr 0.01  \
    --debug 0  \
    --net_size 256 \
    --workers 18  \
    --batchSize 1501  \
    --points_per_patch 300  \
    --patches_per_shape 1000  \
    --sub_sample_size 1000  \
    --cache_capacity 50  \
    --patch_radius 0.0  \
    --single_transformer 0  \
    --uniform_subsample 1  \
    --patch_center 'mean'  \
    --training_order 'random_shape_consecutive'  \
    --outputs 'imp_surf_magnitude' 'imp_surf_sign' 'patch_pts_ids' 'p_index' \
