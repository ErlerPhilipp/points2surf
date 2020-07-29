# This code evaluates a dataset with Screened Poisson Surface Reconstruction (Meshlab)
# To evaluate with PCPNet normals, you need to manually place the PCPNet outputs (*.normals) in
# '{dataset_dir}/06_normals_pcpnet/' before running this script.

import numpy as np
import os
import shutil
import time

from source.base import utils
from source.base import utils_mp
from source.base import evaluation
from source.base import file_utils


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
    meshlabserver = '~/repos/meshlab/src/distrib/meshlabserver'

    num_processes = 12
    # num_processes = 1

    base_dir = 'datasets'
    dataset_dir = dataset_name

    print('Processing dataset: ' + dataset_name)

    filter_broken_inputs = True

    dirs_to_clean = \
        ['00_base_meshes',
         '01_base_meshes_ply',
         '02_meshes_cleaned',
         '03_meshes',
         '04_pts', '04_blensor_py',
         '05_patch_dists', '05_patch_ids', '05_query_dist', '05_query_pts',
         '05_patch_ids_grid', '05_query_pts_grid', '05_query_dist_grid',
         '06_poisson_rec', '06_mc_gt_recon', '06_poisson_rec_gt_normals',
         '06_normals', '06_normals/pts', '06_dist_from_p_normals']

    if filter_broken_inputs:  # the user might have removed unwanted input meshes after some processing
        clean_up_broken_inputs(base_dir=base_dir, dataset_dir=dataset_dir,
                               final_out_dir='00_base_meshes', final_out_extension=None,
                               clean_up_dirs=dirs_to_clean, broken_dir='broken')

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

    # this works only when GT meshes are available
    print('### get ground truth normals for point cloud')
    utils.get_pts_normals(base_dir=base_dir, dataset_dir=dataset_dir,
                          dir_in_pointcloud='04_pts', dir_in_meshes='03_meshes',
                          dir_out_normals='06_normals', samples_per_model=100000, num_processes=num_processes)
    print('### poisson reconstruction from gt normals')
    apply_meshlab_filter(base_dir=base_dir, dataset_dir=dataset_dir, pts_dir='06_normals/pts',
                         recon_mesh_dir='06_poisson_rec_gt_normals', num_processes=num_processes,
                         filter_file='poisson.mlx', meshlabserver_bin=meshlabserver)
    print('### normal estimation and poisson reconstruction gt normals - hausdorff distance')
    new_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '06_poisson_rec_gt_normals')
    ref_meshes_dir_abs = os.path.join(base_dir, dataset_dir, '03_meshes')
    csv_file = os.path.join(base_dir, dataset_dir, 'comp_poisson_rec_gt_normals.csv')
    val_set_file_abs = os.path.join(base_dir, dataset_dir, 'valset.txt')
    evaluation.mesh_comparison(new_meshes_dir_abs=new_meshes_dir_abs, ref_meshes_dir_abs=ref_meshes_dir_abs,
                               num_processes=num_processes, report_name=csv_file,
                               samples_per_model=10000, dataset_file_abs=val_set_file_abs)

    # normal estimation with Meshlab is pretty inaccurate
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


if __name__ == "__main__":
    datasets = [
        'abc', 'abc_extra_noisy', 'abc_noisefree',
        'famous_original', 'famous_noisefree', 'famous_dense', 'famous_extra_noisy', 'famous_sparse',
        'thingi10k_scans_original', 'thingi10k_scans_dense', 'thingi10k_scans_sparse',
        'thingi10k_scans_extra_noisy', 'thingi10k_scans_noisefree',
        'real_world'  # reconstruction with GT normals will fail at real_world
    ]

    for d in datasets:
        main(d)
