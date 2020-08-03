# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#train_}

python full_train.py \
    --name ${NAME}  \
    --desc ${NAME}  \
    --indir 'datasets/abc_train'  \
    --outdir 'models'  \
    --logdir 'logs' \
    --trainset 'trainset.txt'  \
    --testset 'valset.txt'  \
    --nepoch 250  \
    --lr 0.01  \
    --scheduler_steps 100 200  \
    --debug 0  \
    --workers 22  \
    --batchSize 1001  \
    --points_per_patch 300  \
    --patches_per_shape 1000  \
    --sub_sample_size 1000  \
    --cache_capacity 30  \
    --patch_radius 0.0  \
    --single_transformer 0  \
    --shared_transformer 0  \
    --uniform_subsample 1 \
    --use_point_stn 0  \
    --net_size 1024 \
    --patch_center 'mean'  \
    --training_order 'random_shape_consecutive'  \
    --outputs 'imp_surf_magnitude' 'imp_surf_sign' 'patch_pts_ids' 'p_index' \
