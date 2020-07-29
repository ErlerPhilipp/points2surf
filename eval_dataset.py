import numpy as np
import os
import shutil
import configparser
import time

from source.base import utils
from source.base import utils_mp
from source.base import evaluation
from source.base import file_utils
from source import sdf

debug = False


def clean_up_broken_inputs(base_dir, dataset_dir, final_out_dir, final_out_extension,
                           clean_up_dirs, broken_dir='broken'):

    final_out_dir_abs = os.path.join(base_dir, dataset_dir, final_out_dir)
    final_output_files = [f for f in os.listdir(final_out_dir_abs)
                          if os.path.isfile(os.path.join(final_out_dir_abs, f)) and
                          (final_out_extension is None or f[-len(final_out_extension):] == final_out_extension)]

    # move inputs and intermediate results that have no final output
    final_output_file_stems = set(tuple([f.split('.', 1)[0] for f in final_output_files]))
    final_output_file_stem_len = len(final_output_files[0].split('.', 1)[0])
    inconsistent_file_length = final_output_file_stem_len != len(final_output_files[-1].split('.', 1)[0])
    if inconsistent_file_length:
        print('WARNING: output files don\'t have consistent length. Clean-up broken inputs may do unwanted things.')
    for clean_up_dir in clean_up_dirs:
        dir_abs = os.path.join(base_dir, dataset_dir, clean_up_dir)
        if not os.path.isdir(dir_abs):
            continue
        dir_files = [f for f in os.listdir(dir_abs) if os.path.isfile(os.path.join(dir_abs, f))]
        if inconsistent_file_length:
            dir_file_stems = [f.split('.', 1)[0] for f in dir_files]
        else:
            dir_file_stems = [f[:final_output_file_stem_len] for f in dir_files]
        dir_file_stems_without_final_output = [f not in final_output_file_stems for f in dir_file_stems]
        dir_files_without_final_output = np.array(dir_files)[dir_file_stems_without_final_output]

        broken_dir_abs = os.path.join(base_dir, dataset_dir, broken_dir, clean_up_dir)
        broken_files = [os.path.join(broken_dir_abs, f) for f in dir_files_without_final_output]

        for fi, f in enumerate(dir_files_without_final_output):
            os.makedirs(broken_dir_abs, exist_ok=True)
            shutil.move(os.path.join(dir_abs, f), broken_files[fi])


def _reconstruct_gt(pts_file, p_ids_grid_file, query_dist_file, query_pts_file,
                    volume_out_file, mc_out_file,
                    grid_res, sigma, certainty_threshold):

    pts_ms = np.load(pts_file)
    p_ids_grid = np.load(p_ids_grid_file)
    query_dist_ms = np.load(query_dist_file)
    query_pts_ps = np.load(query_pts_file)

    p_pts_ms = pts_ms[p_ids_grid]
    query_pts_ms = utils.patch_space_to_model_space(query_pts_ps, p_pts_ms)

    sdf.implicit_surface_to_mesh(query_dist_ms=query_dist_ms, query_pts_ms=query_pts_ms,
                                 volume_out_file=volume_out_file, mc_out_file=mc_out_file,
                                 grid_res=grid_res, sigma=sigma, certainty_threshold=certainty_threshold)


def reconstruct_gt(base_dir, dataset_dir,
                   pts_dir, p_ids_grid_dir, query_dist_dir, query_pts_dir,
                   gt_reconstruction_dir,
                   grid_resolution, sigma, certainty_threshold, num_processes):

    pts_dir_abs = os.path.join(base_dir, dataset_dir, pts_dir)
    p_ids_grid_dir_abs = os.path.join(base_dir, dataset_dir, p_ids_grid_dir)
    query_dist_dir_abs = os.path.join(base_dir, dataset_dir, query_dist_dir)
    query_pts_dir_abs = os.path.join(base_dir, dataset_dir, query_pts_dir)
    recon_mesh_dir_abs = os.path.join(base_dir, dataset_dir, gt_reconstruction_dir)
    recon_vol_dir_abs = os.path.join(base_dir, dataset_dir, gt_reconstruction_dir, 'vol')

    os.makedirs(recon_mesh_dir_abs, exist_ok=True)
    os.makedirs(recon_vol_dir_abs, exist_ok=True)

    call_params = []
    dist_files = [f for f in os.listdir(query_dist_dir_abs)
                  if os.path.isfile(os.path.join(query_dist_dir_abs, f)) and f[-8:] == '.xyz.npy']
    for dist_file in dist_files:
        pts_file_in = os.path.join(pts_dir_abs, dist_file)
        p_ids_grid_file_in = os.path.join(p_ids_grid_dir_abs, dist_file)
        query_dist_file_in = os.path.join(query_dist_dir_abs, dist_file)
        query_pts_file_in = os.path.join(query_pts_dir_abs, dist_file)
        recon_vol_file_out = os.path.join(recon_vol_dir_abs, dist_file[:-4] + '.off')
        recon_mesh_file_out = os.path.join(recon_mesh_dir_abs, dist_file[:-8] + '.ply')
        if file_utils.call_necessary([pts_file_in, p_ids_grid_file_in, query_dist_file_in, query_pts_file_in],
                                     [recon_mesh_file_out, recon_vol_file_out]):
            call_params.append((pts_file_in, p_ids_grid_file_in, query_dist_file_in, query_pts_file_in,
                                recon_vol_file_out, recon_mesh_file_out, grid_resolution, sigma, certainty_threshold))

    utils_mp.start_process_pool(_reconstruct_gt, call_params, num_processes)


def apply_meshlab_filter(base_dir, dataset_dir, pts_dir, recon_mesh_dir, num_processes, filter_file, meshlabserver_bin):

    pts_dir_abs = os.path.join(base_dir, dataset_dir, pts_dir)
    recon_mesh_dir_abs = os.path.join(base_dir, dataset_dir, recon_mesh_dir)

    os.makedirs(recon_mesh_dir_abs, exist_ok=True)

    calls = []
    pts_files = [f for f in os.listdir(pts_dir_abs)
                 if os.path.isfile(os.path.join(pts_dir_abs, f)) and f[-4:] == '.xyz']
    for pts_file in pts_files:
        pts_file_abs = os.path.join(pts_dir_abs, pts_file)
        poisson_rec_mesh_abs = os.path.join(recon_mesh_dir_abs, pts_file[:-4] + '.ply')
        if file_utils.call_necessary(pts_file_abs, poisson_rec_mesh_abs):
            cmd_args = ' -i {} -o {} -s {}'.format(pts_file_abs, poisson_rec_mesh_abs, filter_file)
            calls.append((meshlabserver_bin + cmd_args,))

    utils_mp.start_process_pool(utils_mp.mp_worker, calls, num_processes)


def get_cgal_normals(base_dir, dataset_dir, pts_dir, normals_out_dir, num_processes, cgal_bin):

    pts_dir_abs = os.path.join(base_dir, dataset_dir, pts_dir)
    normals_out_dir_abs = os.path.join(base_dir, dataset_dir, normals_out_dir)

    os.makedirs(normals_out_dir_abs, exist_ok=True)

    calls = []
    pts_files = [f for f in os.listdir(pts_dir_abs)
                 if os.path.isfile(os.path.join(pts_dir_abs, f)) and f[-4:] == '.xyz']
    for pts_file in pts_files:
        pts_file_abs = os.path.join(pts_dir_abs, pts_file)
        pc_with_normals_abs = os.path.join(normals_out_dir_abs, pts_file[:-4] + '.xyz')
        if file_utils.call_necessary(pts_file_abs, pc_with_normals_abs):
            cmd_args = ' "{}" "{}"'.format(pts_file_abs, pc_with_normals_abs)
            calls.append((cgal_bin + cmd_args,))

    utils_mp.start_process_pool(utils_mp.mp_worker, calls, num_processes)


def read_config(config, config_file):
    if os.path.isfile(config_file):
        config.read(config_file)
    else:
        print("""
ERROR: No config file found. Create a 'settings.ini' in the dataset directory with these contents:
[general]
only_for_evaluation = 0
patch_size = 0.01
grid_resolution = 100 
num_query_points_per_patch = 10
num_scans_per_mesh_min = 5
num_scans_per_mesh_max = 30
scanner_noise_sigma = 0.01
        """)


def main(dataset_name: str):

    # meshlabserver = "C:\\Program Files\\VCG\\MeshLab\\meshlabserver.exe"
    meshlabserver = '/home/perler/repos/meshlab/src/distrib/meshlabserver'

    num_processes = 12
    # num_processes = 1
    # base_dir = '../../datasets'
    base_dir = '/data/datasets/own/'

    #dataset_dir = 'implicit_surf_8'
    #dataset_dir = 'implicit_surf_real_world'
    dataset_dir = dataset_name

    config_file = os.path.join(base_dir, dataset_dir, 'settings.ini')
    config = configparser.ConfigParser()
    read_config(config, config_file)
    print('Processing dataset: ' + config_file)

    do_clean = False
    filter_broken_inputs = True

    dirs_to_clean = \
        ['00_base_meshes',
         '01_base_meshes_ply',
         '02_meshes_cleaned',
         '03_meshes',
         '04_pts', '04_blensor_py',  # , '04_pcd' '04_pts_noisefree',
         '05_patch_dists', '05_patch_ids', '05_query_dist', '05_query_pts',
         '05_patch_ids_grid', '05_query_pts_grid', '05_query_dist_grid',
         '06_poisson_rec', '06_mc_gt_recon', '06_poisson_rec_gt_normals',
         '06_normals', '06_normals/pts', '06_dist_from_p_normals']
    dirs_to_clean_remainders = \
        ['00_base_meshes',
         '01_base_meshes_ply',
         '02_meshes_cleaned',
         '03_meshes',
         '04_pts', '04_blensor_py',  # '04_pcd', '04_pts_noisefree',
         '05_patch_dists', '05_patch_ids', '05_query_dist', '05_query_pts',
         '05_patch_ids_grid', '05_query_pts_grid', '05_query_dist_grid',
         '06_poisson_rec', '06_mc_gt_recon', '06_poisson_rec_gt_normals',
         '06_normals', '06_normals/pts', '06_dist_from_p_normals']

    # clean old dataset
    if do_clean:
        for dir in dirs_to_clean:
            shutil.rmtree(os.path.join(base_dir, dataset_dir, dir), True)

    if filter_broken_inputs:  # the user might have removed unwanted input meshes after some processing
        clean_up_broken_inputs(base_dir=base_dir, dataset_dir=dataset_dir,
                               final_out_dir='00_base_meshes', final_out_extension=None,
                               clean_up_dirs=dirs_to_clean_remainders, broken_dir='broken')

    start = time.time()
    print('### reconstruct poisson with pcpnet normals')
    dirs = (os.path.join(base_dir, dataset_dir, '04_pts_vis'),
            os.path.join(base_dir, dataset_dir, '06_normals_pcpnet'),)
    endings_per_dir = ('.xyz', '.normals', )
    file_utils.concat_txt_dirs(
       ref_dir=os.path.join(base_dir, dataset_dir, '06_normals_pcpnet'), ref_ending='.normals',
       dirs=dirs, endings_per_dir=endings_per_dir,
       out_dir=os.path.join(base_dir, dataset_dir, '07_pts_normals_pcpnet'), out_ending='.xyz')
    print('### poisson reconstruction from pcpnet normals')
    apply_meshlab_filter(base_dir=base_dir, dataset_dir=dataset_dir, pts_dir='07_pts_normals_pcpnet',
                         recon_mesh_dir='06_poisson_rec_pcpnet_normals', num_processes=num_processes,
                         filter_file='poisson.mlx', meshlabserver_bin=meshlabserver)
    end = time.time()
    print('SPSR with PCPNet normals took: {}'.format(end - start))
    print('### normal estimation and poisson reconstruction pcpnet - hausdorff distance')
    new_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '06_poisson_rec_pcpnet_normals')
    ref_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '03_meshes')
    csv_file = os.path.join(base_dir, dataset_dir, 'comp_poisson_rec_pcpnet_normals.csv')
    val_set_file_abs = os.path.join(base_dir, dataset_dir, 'valset.txt')
    evaluation.mesh_comparison(new_meshes_dir_abs=new_meshes_dir_abs, ref_meshes_dir_abs=ref_meshes_dir_abs,
                               num_processes=num_processes, report_name=csv_file,
                               samples_per_model=10000, dataset_file_abs=val_set_file_abs)

    print('### get ground truth normals for point cloud')
    utils.get_pts_normals(base_dir=base_dir, dataset_dir=dataset_dir,
                          dir_in_pointcloud='04_pts', dir_in_meshes='03_meshes',
                          dir_out_normals='06_normals', samples_per_model=100000, num_processes=num_processes)
    print('### poisson reconstruction from gt normals')
    apply_meshlab_filter(base_dir=base_dir, dataset_dir=dataset_dir, pts_dir='06_normals/pts',
                         recon_mesh_dir='06_poisson_rec_gt_normals', num_processes=num_processes,
                         filter_file='../../poisson.mlx', meshlabserver_bin=meshlabserver)
    print('### normal estimation and poisson reconstruction gt normals - hausdorff distance')
    new_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '06_poisson_rec_gt_normals')
    ref_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '03_meshes')
    csv_file = os.path.join(base_dir, dataset_dir, 'comp_poisson_rec_gt_normals.csv')
    val_set_file_abs = os.path.join(base_dir, dataset_dir, 'valset.txt')
    evaluation.mesh_comparison(new_meshes_dir_abs=new_meshes_dir_abs, ref_meshes_dir_abs=ref_meshes_dir_abs,
                               num_processes=num_processes, report_name=csv_file,
                               samples_per_model=10000, dataset_file_abs=val_set_file_abs)

    #print('### get CGAL normals for point cloud')
    #get_cgal_normals(base_dir=base_dir, dataset_dir=dataset_dir, pts_dir='04_pts',
    #                 normals_out_dir='06_pts_normals_cgal', num_processes=num_processes,
    #                 cgal_bin='D:/datasets/own/bin/estimate_normals.exe')
    #print('### poisson reconstruction from cgal normals')
    #apply_meshlab_filter(base_dir=base_dir, dataset_dir=dataset_dir, pts_dir='06_pts_normals_cgal',
    #                     recon_mesh_dir='06_poisson_rec_cgal_normals', num_processes=num_processes,
    #                     filter_file='poisson.mlx', meshlabserver_bin=meshlabserver)
    #print('### normal estimation and poisson reconstruction cgal - hausdorff distance')
    #new_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '06_poisson_rec_cgal_normals')
    #ref_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '03_meshes')
    #csv_file = os.path.join(base_dir, dataset_dir, 'comp_poisson_rec_cgal_normals.csv')
    #val_set_file_abs = os.path.join(base_dir, dataset_dir, 'valset.txt')
    #utils_eval.mesh_comparison(new_meshes_dir_abs=new_meshes_dir_abs, ref_meshes_dir_abs=ref_meshes_dir_abs,
    #                           num_processes=num_processes, report_name=csv_file,
    #                           samples_per_model=10000, dataset_file_abs=val_set_file_abs)

    #print('### distance from query point to patch along normal')
    #utils.get_dist_from_patch_planes(base_dir=base_dir, dataset_dir=dataset_dir,
    #                                 dir_in_pointcloud='04_pts', dir_in_normals='06_normals',
    #                                 dir_in_pids='05_patch_ids', dir_in_query='05_query_pts',
    #                                 dir_out_dists='06_dist_from_p_normals',
    #                                 num_query_points_per_patch=num_query_points_per_patch,
    #                                 num_processes=num_processes)

    #if not only_for_evaluation:
    #    print('### distance from query point to patch along normal - comparison')
    #    utils_eval.eval_predictions(pred_path=os.path.join(base_dir, dataset_dir, '06_dist_from_p_normals'),
    #                                gt_path=os.path.join(base_dir, dataset_dir, '05_query_dist'),
    #                                report_file=os.path.join(base_dir, dataset_dir, '06_rme_comp_p_normal_dists.csv'),
    #                                unsigned=True)

    #print('### normal estimation and poisson reconstruction')
    #apply_meshlab_filter(base_dir=base_dir, dataset_dir=dataset_dir, pts_dir='04_pts',
    #                     recon_mesh_dir='06_poisson_rec', num_processes=num_processes,
    #                     filter_file='normals_poisson.mlx', meshlabserver_bin=meshlabserver)
    #print('### normal estimation and poisson reconstruction - hausdorff distance')
    #new_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '06_poisson_rec')
    #ref_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '03_meshes')
    #csv_file = os.path.join(base_dir, dataset_dir, 'comp_poisson_rec_ml_normals.csv')
    #val_set_file_abs = os.path.join(base_dir, dataset_dir, 'valset.txt')
    #utils_eval.mesh_comparison(new_meshes_dir_abs=new_meshes_dir_abs, ref_meshes_dir_abs=ref_meshes_dir_abs,
    #                           num_processes=num_processes, report_name=csv_file,
    #                           samples_per_model=10000, dataset_file_abs=val_set_file_abs)

    #if not only_for_evaluation:
    #    print('### ground truth reconstruction with marching cubes')
    #    utils_impl_surf.get_query_dist(
    #        base_dir=base_dir, dataset_dir=dataset_dir, dir_in_pointcloud='04_pts',
    #        dir_in_mesh='03_meshes', dir_out_p_ids='05_patch_ids_grid',
    #        dir_out_patch_dists='05_patch_dists', dir_out_query_dist='05_query_dist_grid',
    #        dir_out_query_pts='05_query_pts_grid',
    #        grid_resolution=grid_resolution, r=patch_size,
    #        num_query_points_per_patch=1,
    #        num_processes=num_processes, signed_distance_batch_size=5000,
    #        skip_distances=only_for_evaluation)

    #    reconstruct_gt(base_dir=base_dir, dataset_dir=dataset_dir,
    #                   pts_dir='04_pts', p_ids_grid_dir='05_patch_ids_grid',
    #                   query_dist_dir='05_query_dist_grid', query_pts_dir='05_query_pts_grid',
    #                   gt_reconstruction_dir='06_mc_gt_recon',
    #                   grid_resolution=grid_resolution, sigma=1.0, certainty_threshold=6,
    #                   num_processes=num_processes)
    #    print('### ground truth reconstruction with marching cubes - hausdorff distance')
    #    new_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '06_mc_gt_recon')
    #    ref_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '03_meshes')
    #    csv_file = os.path.join(base_dir, dataset_dir, 'comp_mc_gt_rec.csv')
    #    val_set_file_abs = os.path.join(base_dir, dataset_dir, 'valset.txt')
    #    utils_eval.mesh_comparison(new_meshes_dir_abs=new_meshes_dir_abs, ref_meshes_dir_abs=ref_meshes_dir_abs,
    #                               num_processes=num_processes, report_name=csv_file,
    #                               samples_per_model=10000, dataset_file_abs=val_set_file_abs)


if __name__ == "__main__":
    # datasets = ['test_original', 'test_noisefree', 'test_dense', 'test_extra_noisy', 'test_sparse',
    #             'implicit_surf_14', 'implicit_surf_14_extra_noisy', 'implicit_surf_14_noisefree', 'test_real_world']
    # datasets = ['thingi10k_scans_original', 'thingi10k_scans_extra_noisy', 'thingi10k_scans_noisefree',
    #             'thingi10k_scans_sparse', 'thingi10k_scans_dense']
    datasets = ['implicit_surf_14']

    for d in datasets:
        main(d)
