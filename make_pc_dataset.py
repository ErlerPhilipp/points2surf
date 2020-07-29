import numpy as np
import os
import random

import trimesh
import trimesh.proximity
import trimesh.path
import trimesh.repair
import trimesh.sample

from source.base import utils_mp
from source.base import file_utils
from source.base import point_cloud


def _to_unit_cube(mesh: trimesh.Trimesh):

    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0/bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)

    return mesh


def _convert_point_cloud(in_pc, out_pc_xyz, out_pc_npy, target_num_points=150000):

    mesh = None
    try:
        mesh = trimesh.load(in_pc)
    except AttributeError as e:
        print(e)
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)
    except NameError as e:
        print(e)

    if mesh is not None:
        mesh = _to_unit_cube(mesh)
        points = mesh.vertices
        points = points[:, :3]  # remove normals
        points = points.astype(np.float32)

        # get sub-sample
        if target_num_points is not None and target_num_points > 0 and target_num_points < points.shape[0]:
            sub_sample_ids = np.random.choice(np.arange(points.shape[0]), target_num_points, replace=False)
            points = points[sub_sample_ids]

        file_utils.make_dir_for_file(out_pc_npy)
        file_utils.make_dir_for_file(out_pc_xyz)
        np.save(out_pc_npy, points)
        point_cloud.write_xyz(out_pc_xyz, points)


def convert_point_clouds(in_dir_abs, out_dir_abs, out_dir_npy_abs, target_file_type: str,
                         target_num_points=150000, num_processes=8):
    """
    Convert a mesh file to another file type.
    :param in_dir_abs:
    :param out_dir_abs:
    :param out_dir_npy_abs:
    :param target_file_type: ending of wanted mesh file, e.g. '.ply'
    :return:
    """

    os.makedirs(out_dir_abs, exist_ok=True)

    mesh_files = []
    for root, dirs, files in os.walk(in_dir_abs, topdown=True):
        for name in files:
            mesh_files.append(os.path.join(root, name))

    allowed_mesh_types = ['.off', '.ply', '.obj', '.stl']
    mesh_files = list(filter(lambda f: (f[-4:] in allowed_mesh_types), mesh_files))

    calls = []
    for fi, f in enumerate(mesh_files):
        file_base_name = os.path.basename(f)
        file_out = os.path.join(out_dir_abs, file_base_name[:-4] + target_file_type)
        file_out_npy = os.path.join(out_dir_npy_abs, file_base_name[:-4] + target_file_type + '.npy')
        if file_utils.call_necessary(f, [file_out, file_out_npy]):
            calls.append((f, file_out, file_out_npy, target_num_points))

    utils_mp.start_process_pool(_convert_point_cloud, calls, num_processes)


def make_dataset_splits(base_dir, dataset_dir, final_out_dir, seed=42, only_test_set=False, testset_ratio=0.1):

    rnd = random.Random(seed)

    # write files for train / test / eval set
    final_out_dir_abs = os.path.join(base_dir, dataset_dir, final_out_dir)
    final_output_files = [f for f in os.listdir(final_out_dir_abs)
                          if os.path.isfile(os.path.join(final_out_dir_abs, f)) and f[-4:] == '.npy']
    files_dataset = [f[:-8] for f in final_output_files]

    if len(files_dataset) == 0:
        raise ValueError('Dataset is empty! {}'.format(final_out_dir_abs))

    if only_test_set:
        files_test = files_dataset
    else:
        files_test = rnd.sample(files_dataset, max(3, min(int(testset_ratio * len(files_dataset)), 100)))  # 3..50, ~10%
    files_train = list(set(files_dataset).difference(set(files_test)))

    files_test.sort()
    files_train.sort()

    file_train_set = os.path.join(base_dir, dataset_dir, 'trainset.txt')
    file_test_set = os.path.join(base_dir, dataset_dir, 'testset.txt')
    file_val_set = os.path.join(base_dir, dataset_dir, 'valset.txt')

    file_utils.make_dir_for_file(file_test_set)
    nl = '\n'
    file_test_set_str = nl.join(files_test)
    file_train_set_str = nl.join(files_train)
    with open(file_test_set, "w") as text_file:
        text_file.write(file_test_set_str)
    if not only_test_set:
        with open(file_train_set, "w") as text_file:
            text_file.write(file_train_set_str)
    with open(file_val_set, "w") as text_file:
        text_file.write(file_test_set_str)  # validate the test set by default


def main(dataset_name: str):

    #base_dir = '/data/datasets/own/'
    base_dir = '../../datasets'
    # base_dir = '/home/dlmain/perler/meshnet/datasets/'
    #num_processes = 1
    num_processes = 14  # 16 processes need up to 64 GB RAM for the signed distances

    # dataset_dir = 'implicit_surf_13'
    # dataset_dir = 'implicit_surf_0'
    dataset_dir = dataset_name

    print('Processing dataset: ' + os.path.join(base_dir, dataset_dir))

    # no signed distances needed, therefore less strict requirements for input meshes
    only_for_evaluation = True

    print('### convert base meshes to ply')
    convert_point_clouds(in_dir_abs=os.path.join(base_dir, dataset_dir, '00_base_meshes'),
                         out_dir_abs=os.path.join(base_dir, dataset_dir, '04_pts_vis'),
                         out_dir_npy_abs=os.path.join(base_dir, dataset_dir, '04_pts'),
                         target_file_type='.xyz', target_num_points=50000, num_processes=num_processes)

    make_dataset_splits(base_dir=base_dir, dataset_dir=dataset_dir,
                        final_out_dir='04_pts' if only_for_evaluation else'05_query_pts',
                        seed=42, only_test_set=only_for_evaluation, testset_ratio=0.1)


if __name__ == "__main__":
    #datasets = ['test_original', 'test_noisefree', 'test_dense', 'test_extra_noisy', 'test_sparse', 'implicit_surf_14']
    datasets = ['test_real_world']

    for d in datasets:
        main(d)
