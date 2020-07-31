import os

from source import points_to_surf_train
from source import points_to_surf_eval
from source import sdf
from source.base import evaluation

# When you see this error:
# 'Expected more than 1 value per channel when training...' which is raised by the BatchNorm1d layer
# for multi-gpu, use a batch size that can't be divided by the number of GPUs
# for single-gpu, use a straight batch size
# see https://github.com/pytorch/pytorch/issues/2584
# see https://forums.fast.ai/t/understanding-code-error-expected-more-than-1-value-per-channel-when-training/9257/12


if __name__ == '__main__':

    model_name = 'vanilla'
    dataset = 'abc_minimal'
    base_dir = 'datasets'
    in_dir_train = os.path.join(base_dir, dataset)

    train_set = 'trainset.txt'
    val_set = 'valset.txt'
    test_set = 'testset.txt'

    # features = ['imp_surf', 'patch_pts_ids', 'p_index']  # l2-loss
    features = ['imp_surf_magnitude', 'imp_surf_sign', 'patch_pts_ids', 'p_index']  # l2-loss + BCE-loss

    # workers = 22  # for strong training machine
    workers = 7  # for typical PC

    # batch_size = 501  # ~7.5 GB memory on 4 2080 TI for 300 patch points + 1000 sub-sample points
    # batch_size = 3001  # ~10 GB memory on 4 2080 TI for 50 patch points + 200 sub-sample points
    batch_size = 100  # ~7 GB memory on 1 1070 for 300 patch points + 1000 sub-sample points

    # grid_resolution = 256  # quality like in the paper
    grid_resolution = 128  # quality for a short test
    rec_epsilon = 3
    certainty_threshold = 13
    sigma = 5

    fixed_radius = False
    patch_radius = 0.1 if fixed_radius else 0.0
    single_transformer = 0
    shared_transformer = 0

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_params = [
        '--name', model_name,
        '--desc', model_name,
        '--indir', in_dir_train,
        '--outdir', 'models',
        '--trainset', train_set,
        '--testset', val_set,
        '--net_size', str(1024),
        '--nepoch', str(10),
        '--lr', str(0.01),
        '--debug', str(0),
        '--workers', str(workers),
        '--batchSize', str(batch_size),
        '--points_per_patch', str(300),
        '--patches_per_shape', str(1000),
        '--sub_sample_size', str(1000),
        '--cache_capacity', str(10),
        '--patch_radius', str(patch_radius),
        '--single_transformer', str(single_transformer),
        '--shared_transformer', str(shared_transformer),
        '--patch_center', 'mean',
        '--training_order', 'random_shape_consecutive',
        '--use_point_stn', str(1),
        '--uniform_subsample', str(0),
        '--outputs',
    ]
    train_params += features

    # train model on GT data with multiple query points per patch
    train_opt = points_to_surf_train.parse_arguments(train_params)
    points_to_surf_train.points_to_surf_train(train_opt)

    valsets = ['abc_minimal', ]
    for valset in valsets:
        # validate model
        in_dir_val = os.path.join(base_dir, valset)
        out_dir_val = os.path.join('results', model_name, valset)
        res_dir_eval = os.path.join(out_dir_val, 'eval')
        eval_params = [
            '--indir', in_dir_val,
            '--outdir', out_dir_val,
            '--dataset', val_set,
            '--models', model_name,
            '--batchSize', str(batch_size),
            '--workers', str(workers),
            '--cache_capacity', str(5),
        ]
        eval_opt = points_to_surf_eval.parse_arguments(eval_params)
        points_to_surf_eval.points_to_surf_eval(eval_opt)
        evaluation.eval_predictions(
            os.path.join(res_dir_eval, 'eval'),
            os.path.join(in_dir_val, '05_query_dist'),
            os.path.join(res_dir_eval, 'rme_comp_res.csv'),
            unsigned=False)

    testsets = ['abc_minimal', ]
    for testset in testsets:
        out_dir = os.path.join('results', model_name, testset)
        res_dir_rec = os.path.join(out_dir, 'rec')

        # reconstruct SDFs from testset
        in_dir_test = os.path.join(base_dir, testset)
        print('Points2Surf is reconstructing {} into {}'.format(out_dir, res_dir_rec))
        recon_params = [
                       '--indir', in_dir_test,
                       '--outdir', out_dir,
                       '--dataset', test_set,
                       '--query_grid_resolution', str(grid_resolution),
                       '--reconstruction', str(True),
                       '--models', model_name,
                       '--batchSize', str(batch_size),
                       '--workers', str(workers),
                       '--cache_capacity', str(5),
                       '--epsilon', str(rec_epsilon),
                       ]
        recon_opt = points_to_surf_eval.parse_arguments(recon_params)
        points_to_surf_eval.points_to_surf_eval(recon_opt)

        # reconstruct meshes from predicted SDFs
        imp_surf_dist_ms_dir = os.path.join(res_dir_rec, 'dist_ms')
        query_pts_ms_dir = os.path.join(res_dir_rec, 'query_pts_ms')
        vol_out_dir = os.path.join(res_dir_rec, 'vol')
        mesh_out_dir = os.path.join(res_dir_rec, 'mesh')
        sdf.implicit_surface_to_mesh_directory(
            imp_surf_dist_ms_dir, query_pts_ms_dir,
            vol_out_dir, mesh_out_dir,
            grid_resolution, sigma, certainty_threshold,
            workers)

        # get Hausdorff distance for reconstructed meshes
        new_meshes_dir_abs = os.path.join(res_dir_rec, 'mesh')
        ref_meshes_dir_abs = os.path.join(in_dir_test, '03_meshes')
        csv_file = os.path.join(res_dir_rec, 'hausdorff_dist_pred_rec.csv')
        evaluation.mesh_comparison(
            new_meshes_dir_abs=new_meshes_dir_abs,
            ref_meshes_dir_abs=ref_meshes_dir_abs,
            num_processes=workers,
            report_name=csv_file,
            samples_per_model=10000,
            dataset_file_abs=os.path.join(in_dir_test, test_set))

    print('Points2Surf is finished!')
