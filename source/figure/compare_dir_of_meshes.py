# This messy code calculates the Chamfer distance for DeepSDF and AtlasNet outputs

from source.base import evaluation
from source.base import utils_mp

import trimesh
import numpy as np

import os


def _to_unit_cube(file_in, in_file_ref, file_out):

    mesh = trimesh.load(file_in)
    mesh_ref = trimesh.load(in_file_ref)
    ref_bb = mesh_ref.bounds
    #mesh_ref = np.load(in_file_ref)  # for real-world, load point cloud
    #ref_bb = [np.min(mesh_ref, axis=0), np.max(mesh_ref, axis=0)]


    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0 / np.max(np.abs(mesh.bounds[1] - mesh.bounds[0]))  # atlasnet does some strange scaling and translation
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)

    # the datasets were not really scaled to unitcube (first scaled to unit cube and then randomly rotated)
    # revert this error here
    ref_scale = np.max(np.abs(ref_bb[1] - ref_bb[0]))
    ref_scale_trafo = trimesh.transformations.scale_matrix(factor=ref_scale)
    mesh.apply_transform(ref_scale_trafo)

    # revert the wrong offset
    translation = (ref_bb[0] + ref_bb[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=translation)
    mesh.apply_transform(translation)

    mesh.export(file_out)


def revert_atlasnet_transform(in_dir_abs, out_dir_abs, ref_meshes_dir_abs, num_processes=1):

    os.makedirs(out_dir_abs, exist_ok=True)

    call_params = []

    mesh_files = [f for f in os.listdir(in_dir_abs)
                  if os.path.isfile(os.path.join(in_dir_abs, f))]
    for fi, f in enumerate(mesh_files):
        in_file_abs = os.path.join(in_dir_abs, f)
        in_file_ref = os.path.join(ref_meshes_dir_abs, f[:-12] + '.ply')
        #in_file_ref = os.path.join(ref_meshes_dir_abs, f[:-12] + '.xyz.npy')  # for real-world
        out_file_abs = os.path.join(out_dir_abs, f[:-4] + '.ply')

        #if utils_files.call_necessary(in_file_abs, out_file_abs):
        call_params += [(in_file_abs, in_file_ref, out_file_abs)]

    utils_mp.start_process_pool(_to_unit_cube, call_params, num_processes)


if __name__ == "__main__":
    #datasets = ['test_original', 'test_noisefree', 'test_dense', 'test_extra_noisy', 'test_sparse',
    #            'implicit_surf_14', 'implicit_surf_14_extra_noisy', 'implicit_surf_14_noisefree', 'test_real_world']
    datasets = ['thingi10k_scans_original', 'thingi10k_scans_extra_noisy', 'thingi10k_scans_noisefree',
                'thingi10k_scans_sparse', 'thingi10k_scans_dense']
    num_processes = 14
    #num_processes = 1

    for dataset in datasets:
#        # AtlasNet train imp_surf_14, val imp_surf_14
#        original_meshes = '/data/datasets/own/' + dataset + '/03_meshes'
#        original_pc = '/data/datasets/own/' + dataset + '/04_pts'
#        val_set_file_abs = '/data/datasets/own/' + dataset + '/testset.txt'
#        reconstructed_meshes_raw = '/home/perler/repos/AtlasNet/log/atlasnet_ae_imp_surf_14_25_squares/' + dataset + '/'
#        reconstructed_meshes = '/home/perler/repos/AtlasNet/log/atlasnet_ae_imp_surf_14_25_squares/' + dataset + '/ply/'
#        report_file = '/home/perler/repos/AtlasNet/log/atlasnet_ae_imp_surf_14_25_squares/' + dataset + '.csv'
#
#        # for AtlasNet
#        revert_atlasnet_transform(
#            in_dir_abs=reconstructed_meshes_raw,
#            out_dir_abs=reconstructed_meshes,
#            ref_meshes_dir_abs=original_meshes,
#            #ref_meshes_dir_abs=original_pc,  # for real-world, no mesh exists
#            num_processes=num_processes)
#
#        utils_eval.mesh_comparison(
#            new_meshes_dir_abs=reconstructed_meshes, ref_meshes_dir_abs=original_meshes,
#            num_processes=num_processes, report_name=report_file,
#            samples_per_model=10000, dataset_file_abs=val_set_file_abs)

        # DeepSDF train imp_surf_14, val imp_surf_14
        original_meshes = '/data/datasets/own/' + dataset + '/03_meshes'
        val_set_file_abs = '/data/datasets/own/' + dataset + '/testset.txt'
        reconstructed_meshes = '/home/perler/repos/DeepSDF/examples/' + dataset + '/Reconstructions/1000/Meshes/' + dataset + '/03_meshes/'
        report_file = '/home/perler/repos/DeepSDF/examples/' + dataset + '/Reconstructions/1000/Meshes/' + dataset + '/imp_surf_14_comp.csv'

        print('### chamfer distance: {}'.format(dataset))
        evaluation.mesh_comparison(new_meshes_dir_abs=reconstructed_meshes, ref_meshes_dir_abs=original_meshes,
                                   num_processes=num_processes, report_name=report_file,
                                   samples_per_model=10000, dataset_file_abs=val_set_file_abs)


