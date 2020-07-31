import os
import time

from source import points_to_surf_eval
from source.base import evaluation
from source import sdf


# When you see this error:
# 'Expected more than 1 value per channel when training...' which is raised by the BatchNorm1d layer
# for multi-gpu, use a batch size that can't be divided by the number of GPUs
# for single-gpu, use a straight batch size
# see https://github.com/pytorch/pytorch/issues/2584
# see https://forums.fast.ai/t/understanding-code-error-expected-more-than-1-value-per-channel-when-training/9257/12


def full_eval(opt):

    indir_root = opt.indir
    outdir_root = os.path.join(opt.outdir, opt.models+os.path.splitext(opt.modelpostfix)[0])
    datasets = opt.dataset
    if not isinstance(datasets, list):
        datasets = [datasets]
    for dataset in datasets:
        print(f'Evaluating on dataset {dataset}')
        opt.indir = os.path.join(indir_root, os.path.dirname(dataset))
        opt.outdir = os.path.join(outdir_root, os.path.dirname(dataset))
        opt.dataset = os.path.basename(dataset)

        # evaluate
        if os.path.exists(os.path.join(opt.indir, '05_query_dist')):
            opt.reconstruction = False
            points_to_surf_eval.points_to_surf_eval(opt)

            res_dir_eval = os.path.join(opt.outdir, 'eval')

            evaluation.eval_predictions(
                os.path.join(res_dir_eval, 'eval'),
                os.path.join(opt.indir, '05_query_dist'),
                os.path.join(res_dir_eval, 'rme_comp_res.csv'),
                unsigned=False)

        # reconstruct
        start = time.time()
        opt.reconstruction = True
        points_to_surf_eval.points_to_surf_eval(opt)
        res_dir_rec = os.path.join(opt.outdir, 'rec')
        end = time.time()
        print('Inference of SDF took: {}'.format(end - start))

        start = time.time()
        imp_surf_dist_ms_dir = os.path.join(res_dir_rec, 'dist_ms')
        query_pts_ms_dir = os.path.join(res_dir_rec, 'query_pts_ms')
        vol_out_dir = os.path.join(res_dir_rec, 'vol')
        mesh_out_dir = os.path.join(res_dir_rec, 'mesh')
        sdf.implicit_surface_to_mesh_directory(
            imp_surf_dist_ms_dir, query_pts_ms_dir,
            vol_out_dir, mesh_out_dir,
            opt.query_grid_resolution,
            opt.sigma,
            opt.certainty_threshold,
            opt.workers)
        end = time.time()
        print('Sign propagation took: {}'.format(end - start))

        new_meshes_dir_abs = os.path.join(res_dir_rec, 'mesh')
        ref_meshes_dir_abs = os.path.join(opt.indir, '03_meshes')
        csv_file = os.path.join(res_dir_rec, 'hausdorff_dist_pred_rec.csv')
        evaluation.mesh_comparison(
            new_meshes_dir_abs=new_meshes_dir_abs,
            ref_meshes_dir_abs=ref_meshes_dir_abs,
            num_processes=opt.workers,
            report_name=csv_file,
            samples_per_model=10000,
            dataset_file_abs=os.path.join(opt.indir, opt.dataset))


if __name__ == '__main__':

    # evaluate model on GT data with multiple query points per patch
    full_eval(opt=points_to_surf_eval.parse_arguments())

    print('MeshNet is finished!')
