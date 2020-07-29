# This rather messy code converts our datasets into a format that DeepSDF can use.
# To use it on your own data you may need to change some hard-coded paths and constants in the main.
# Also, you need meshlabserver for some mesh cleaning.

import os
import trimesh
import trimesh.repair
import numpy as np

from source import sdf
from source.base import utils_mp
from source.base import file_utils


def _convert_pc(in_pc, out_pc):

    pc = None
    try:
        pc = np.load(in_pc)
        pc = pc.astype(np.float)
    except AttributeError as e:
        print(e)
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)
    except NameError as e:
        print(e)

    if pc is not None:
        try:
            # just useless faces because separated vertices would be removed
            faces = np.zeros((pc.shape[0], 3), dtype=np.int32)
            faces[:, 1] = 1
            faces[:, 2] = np.arange(pc.shape[0])
            pc_ply = trimesh.Trimesh(vertices=pc, faces=faces)
            file_utils.make_dir_for_file(out_pc)
            pc_ply.export(out_pc)
        except ValueError as e:
            print(e)


def convert_pcs(in_dir_abs, out_dir_abs, file_set, num_processes):
    """
    Convert a mesh file to another file type.
    :param in_dir_abs:
    :param out_dir_abs:
    :return:
    """

    os.makedirs(out_dir_abs, exist_ok=True)

    mesh_files = []
    for root, dirs, files in os.walk(in_dir_abs, topdown=True):
        for name in files:
            mesh_files.append(os.path.join(root, name))

    allowed_mesh_types = ['.npy']
    mesh_files = list(filter(lambda f: (f[-4:] in allowed_mesh_types), mesh_files))

    files = open(file_set).readlines()
    files = set([f.replace('\n', '') for f in files])
    mesh_files = list(filter(lambda f: (os.path.basename(f)[:-8] in files), mesh_files))

    calls = []
    for fi, f in enumerate(mesh_files):
        file_base_name = os.path.basename(f)
        file_out = os.path.join(out_dir_abs, file_base_name[:-8] + '.ply')
        if file_utils.call_necessary(f, file_out):
            calls.append((f, file_out))

    utils_mp.start_process_pool(_convert_pc, calls, num_processes)


def _convert_sdf(file_in_query_pts, file_in_sdf, out_pc):

    query_pts = np.load(file_in_query_pts)
    query_dist_ms = np.load(file_in_sdf)

    # DeepSDF only accepts float, not double
    query_pts = query_pts.astype(dtype=np.float32)
    query_dist_ms = query_dist_ms.astype(dtype=np.float32)

    pos = query_dist_ms > 0.0
    neg = query_dist_ms < 0.0
    pc_pos = query_pts[pos]
    sdf_pos = query_dist_ms[pos]
    pc_neg = query_pts[neg]
    sdf_neg = query_dist_ms[neg]

    pc_sdf_pos = np.zeros((pc_pos.shape[0], 4), dtype=np.float32)
    pc_sdf_pos[:, 0:3] = pc_pos
    pc_sdf_pos[:, 3] = sdf_pos

    pc_sdf_neg = np.zeros((pc_neg.shape[0], 4), dtype=np.float32)
    pc_sdf_neg[:, 0:3] = pc_neg
    pc_sdf_neg[:, 3] = sdf_neg

    np.savez(out_pc, pos=pc_sdf_pos, neg=pc_sdf_neg)


def _make_sdf_samples_from_pc(file_in_pts, file_in_normal, file_in_mesh, out_pc):

    # deviation from the surface in both directions, as described in the paper, section 6.3
    eta = 0.01  # actual value is never mentioned, so we just assume something small

    pts = np.load(file_in_pts)
    pts = pts.astype(dtype=np.float32)  # DeepSDF only accepts float, not double

    normals = np.loadtxt(file_in_normal)
    normals_length = np.linalg.norm(normals, axis=1)
    normals_length_repeated = np.repeat(np.expand_dims(normals_length, axis=1), 3, axis=1)
    normals_normalized = normals / normals_length_repeated

    # get sdf samples near the surface
    query_pts_pos = pts + eta * normals_normalized
    query_pts_neg = pts - eta * normals_normalized
    signed_dist_pos = np.full((query_pts_pos.shape[0],), -eta)
    signed_dist_neg = np.full((query_pts_neg.shape[0],), eta)

    # combine close points and signed distance to 4D vectors
    close_sdf_samples_pos = np.zeros((query_pts_pos.shape[0], 4), dtype=np.float32)
    close_sdf_samples_pos[:, 0:3] = query_pts_pos
    close_sdf_samples_pos[:, 3] = signed_dist_pos
    close_sdf_samples_neg = np.zeros((query_pts_neg.shape[0], 4), dtype=np.float32)
    close_sdf_samples_neg[:, 0:3] = query_pts_neg
    close_sdf_samples_neg[:, 3] = signed_dist_neg

    # get sdf samples in the unit cube (can be far away from the surface)
    far_samples_ratio = 0.2
    num_far_samples = int((query_pts_pos.shape[0] + query_pts_neg.shape[0]) * far_samples_ratio)
    random_samples_unit_cube = np.random.rand(num_far_samples, 3) - 0.5
    mesh = trimesh.load(file_in_mesh)
    random_samples_unit_cube_sd = sdf.get_signed_distance(
        in_mesh=mesh,
        query_pts_ms=random_samples_unit_cube,
        signed_distance_batch_size=1000
    )

    # split far sdf samples in pos and neg (inside and outside)
    random_samples_unit_cube_sd_ids_pos = random_samples_unit_cube_sd > 0.0
    random_samples_unit_cube_sd_ids_neg = random_samples_unit_cube_sd < 0.0
    random_samples_unit_cube_pos = random_samples_unit_cube[random_samples_unit_cube_sd_ids_pos]
    random_samples_unit_cube_sd_pos = random_samples_unit_cube_sd[random_samples_unit_cube_sd_ids_pos]
    random_samples_unit_cube_neg = random_samples_unit_cube[random_samples_unit_cube_sd_ids_neg]
    random_samples_unit_cube_sd_neg = random_samples_unit_cube_sd[random_samples_unit_cube_sd_ids_neg]

    # combine far points and signed distance to 4D vectors
    far_sdf_samples_pos = np.zeros((random_samples_unit_cube_pos.shape[0], 4), dtype=np.float32)
    far_sdf_samples_pos[:, 0:3] = random_samples_unit_cube_pos
    far_sdf_samples_pos[:, 3] = random_samples_unit_cube_sd_pos
    far_sdf_samples_neg = np.zeros((random_samples_unit_cube_neg.shape[0], 4), dtype=np.float32)
    far_sdf_samples_neg[:, 0:3] = random_samples_unit_cube_neg
    far_sdf_samples_neg[:, 3] = random_samples_unit_cube_sd_neg

    np.savez(out_pc, pos=close_sdf_samples_neg, neg=close_sdf_samples_pos,
             pos_far=far_sdf_samples_pos, neg_far=far_sdf_samples_neg)

    # debug: save visualization
    file_out_query_vis = out_pc + '.ply'
    query_pts_ms = np.concatenate((query_pts_pos, query_pts_neg, random_samples_unit_cube_pos, random_samples_unit_cube_neg))
    query_dist_ms = np.concatenate((signed_dist_pos, signed_dist_neg, random_samples_unit_cube_sd_pos, random_samples_unit_cube_sd_neg))
    sdf.visualize_query_points(query_pts_ms, query_dist_ms, file_out_query_vis)
    print('wrote vis to {}'.format(file_out_query_vis))


def convert_sdfs(in_dir_query_pts, in_dir_query_sdf, out_dir_sdf,
                 file_set, num_processes):

    if not os.path.isfile(file_set):
        print('WARNING: dataset is missing a set file: {}'.format(file_set))
        return

    os.makedirs(out_dir_sdf, exist_ok=True)

    query_pts_files = []
    for root, dirs, files in os.walk(in_dir_query_pts, topdown=True):
        for name in files:
            query_pts_files.append(os.path.join(root, name))

    query_pts_files = list(filter(lambda f: (f[-4:] in ['.npy']), query_pts_files))

    files = open(file_set).readlines()
    files = set([f.replace('\n', '') for f in files])
    query_pts_files = list(filter(lambda f: (os.path.basename(f)[:-8] in files), query_pts_files))

    calls = []
    for fi, query_pts_file in enumerate(query_pts_files):
        file_base_name = os.path.basename(query_pts_file)
        file_out_pc = os.path.join(out_dir_sdf, file_base_name[:-8] + '.npz')
        file_in_sdf = os.path.join(in_dir_query_sdf, file_base_name[:-8] + '.ply.npy')
        calls.append((query_pts_file, file_in_sdf, file_out_pc))

    utils_mp.start_process_pool(_convert_sdf, calls, num_processes)


def make_sdf_samples(in_dir_pts, in_dir_normals, in_dir_meshes, out_dir_sdf,
                     file_set, num_processes):

    if not os.path.isfile(file_set):
        print('WARNING: dataset is missing a set file: {}'.format(file_set))
        return

    os.makedirs(out_dir_sdf, exist_ok=True)

    pts_files = []
    for root, dirs, files in os.walk(in_dir_pts, topdown=True):
        for name in files:
            pts_files.append(os.path.join(root, name))

    pts_files = list(filter(lambda f: (f[-4:] in ['.npy']), pts_files))

    files = open(file_set).readlines()
    files = set([f.replace('\n', '') for f in files])
    pts_files = list(filter(lambda f: (os.path.basename(f)[:-8] in files), pts_files))

    calls = []
    for fi, query_pts_file in enumerate(pts_files):
        file_base_name = os.path.basename(query_pts_file)
        file_out_pc = os.path.join(out_dir_sdf, file_base_name[:-8] + '.npz')
        file_in_normal = os.path.join(in_dir_normals, file_base_name[:-8] + '.normals')
        file_in_mesh = os.path.join(in_dir_meshes, file_base_name[:-8] + '.ply')
        calls.append((query_pts_file, file_in_normal, file_in_mesh, file_out_pc))

    utils_mp.start_process_pool(_make_sdf_samples_from_pc, calls, num_processes)


def create_example(train_set, test_set, out_dir_examples, dataset):

    out_dir_example = os.path.join(out_dir_examples, dataset)
    out_file_specs_json = os.path.join(out_dir_example, 'specs.json')

    out_dir_splits = os.path.join(out_dir_examples, 'splits')
    out_file_train_set_json = os.path.join(out_dir_splits, '{}_train.json'.format(dataset))
    out_file_test_set_json = os.path.join(out_dir_splits, '{}_test.json'.format(dataset))

    os.makedirs(out_dir_example, exist_ok=True)

    specs_json_template = '''
{{
  "Description" : [ "converted from {origin}." ],
  "DataSource" : "data/",
  "TrainSplit" : "examples/splits/{dataset_name_train}_train.json",
  "TestSplit" : "examples/splits/{dataset_name_test}_test.json",
  "NetworkArch" : "deep_sdf_decoder",
  "NetworkSpecs" : {{
    "dims" : [ 512, 512, 512, 512, 512, 512, 512, 512 ],
    "dropout" : [0, 1, 2, 3, 4, 5, 6, 7],
    "dropout_prob" : 0.2,
    "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
    "latent_in" : [4],
    "xyz_in_all" : false,
    "use_tanh" : false,
    "latent_dropout" : false,
    "weight_norm" : true
    }},
  "CodeLength" : 256,
  "NumEpochs" : 1001,
  "SnapshotFrequency" : 100,
  "AdditionalSnapshots" : [ 100, 200, 500 ],
  "LearningRateSchedule" : [
    {{
      "Type" : "Step",
      "Initial" : 0.0005,
      "Interval" : 500,
      "Factor" : 0.5
    }},
    {{
      "Type" : "Step",
      "Initial" : 0.001,
      "Interval" : 500,
      "Factor" : 0.5
    }}],
  "SamplesPerScene" : 16384,
  "ScenesPerBatch" : 64,
  "DataLoaderThreads" : 16,
  "ClampingDistance" : 0.1,
  "CodeRegularization" : true,
  "CodeRegularizationLambda" : 1e-4
}}
    '''
    # default settings except the code_bound parameter is missing because it caused crashs
    this_specs_json = specs_json_template.format(
        origin=train_set, dataset_name_train=dataset, dataset_name_test=dataset)

    with open(out_file_specs_json, "w") as text_file:
        text_file.write(this_specs_json)

    def get_set_file_string(set_file):
        with open(set_file, "r") as set_file:
            set_files = set_file.readlines()
        json_file_list = ['\t\t\t"{f}",'.format(f=f.replace('\n', '')) for f in set_files]
        json_file_list[-1] = json_file_list[-1][:-1]  # remove trailing comma of last entry
        return '\n'.join(json_file_list)

    files_set_header = '''
{{
  "{dataset}": {{
    "03_meshes": [
'''.format(dataset=dataset)

    files_set_footer = '''
    ]
  }
}
'''  # no double curly braces because no format()

    if os.path.isfile(train_set):
        out_file_train_set_json_contents = files_set_header + get_set_file_string(train_set) + files_set_footer
        with open(out_file_train_set_json, "w") as text_file:
            text_file.write(out_file_train_set_json_contents)

    if os.path.isfile(test_set):
        out_file_test_set_json_contents = files_set_header + get_set_file_string(test_set) + files_set_footer
        with open(out_file_test_set_json, "w") as text_file:
            text_file.write(out_file_test_set_json_contents)


def apply_meshlab_filter(base_dir, dataset_dir, in_dir, out_dir, num_processes, filter_file, meshlabserver_bin):

    in_dir_abs = os.path.join(base_dir, dataset_dir, in_dir)
    out_mesh_dir_abs = os.path.join(base_dir, dataset_dir, out_dir)

    os.makedirs(out_mesh_dir_abs, exist_ok=True)

    calls = []
    pts_files = [f for f in os.listdir(in_dir_abs)
                 if os.path.isfile(os.path.join(in_dir_abs, f))]
    for pts_file in pts_files:
        pts_file_abs = os.path.join(in_dir_abs, pts_file)
        poisson_rec_mesh_abs = os.path.join(out_mesh_dir_abs, pts_file)
        if file_utils.call_necessary(pts_file_abs, poisson_rec_mesh_abs):
            cmd_args = ' -i {} -o {} -s {} --verbose'.format(pts_file_abs, poisson_rec_mesh_abs, filter_file)
            calls.append((meshlabserver_bin + cmd_args,))

    utils_mp.start_process_pool(utils_mp.mp_worker, calls, num_processes)


def main():

    num_processes = 12
    #num_processes = 1

    base_dir = 'datasets'
    meshlabserver = '~/repos/meshlab/src/distrib/meshlabserver'
    hole_filling_mesh_simp_script = 'hole_filling_mesh_simp.mlx'
    out_dir_examples = '~/repos/DeepSDF/examples/'

    datasets = [
        'abc', 'abc_extra_noisy', 'abc_noisefree',
        'famous_original', 'famous_noisefree', 'famous_dense', 'famous_extra_noisy', 'famous_sparse',
        'thingi10k_scans_original', 'thingi10k_scans_dense', 'thingi10k_scans_sparse',
        'thingi10k_scans_extra_noisy', 'thingi10k_scans_noisefree',
        'real_world'
    ]

    for dataset in datasets:
        print('Processing {}'.format(dataset))

        dir_mesh = '03_meshes'
        dir_mesh_repaired = '05_meshes_repaired'
        dir_mesh_repaired_abs = os.path.join(base_dir, dataset, dir_mesh_repaired)
        dir_pc = os.path.join(base_dir, dataset, '04_pts')
        dir_query_pts = os.path.join(base_dir, dataset, '05_query_pts')
        dir_sdf_abs = os.path.join(base_dir, dataset, '05_query_dist')
        dir_normals_abs = os.path.join(base_dir, dataset, '06_normals_pcpnet')
        test_set = os.path.join(base_dir, dataset, 'testset.txt')
        train_set = os.path.join(base_dir, dataset, 'trainset.txt')
        out_dir_pc_abs = '/home/perler/repos/DeepSDF/data/SurfaceSamples/' + dataset + '/03_meshes/'
        out_dir_sdf_abs = '/home/perler/repos/DeepSDF/data/SdfSamples/' + dataset + '/03_meshes/'

        # during DeepSDF reconstruction, we use GT signed distances for the SDF samples far from the surface
        # we fill possible holes with Meshlab
        apply_meshlab_filter(base_dir=base_dir, dataset_dir=dataset, in_dir=dir_mesh,
                             out_dir=dir_mesh_repaired, num_processes=num_processes,
                             filter_file=hole_filling_mesh_simp_script, meshlabserver_bin=meshlabserver)

        # GT samples for evaluation (Chamfer distance)
        # this should be directly on the surface (no noise) or it will result in wrong Chamfer distances
        convert_pcs(dir_pc, out_dir_pc_abs, test_set, num_processes)

        # for training (take query points + sdf from our dataset)
        convert_sdfs(
            in_dir_query_pts=dir_query_pts,
            in_dir_query_sdf=dir_sdf_abs,
            out_dir_sdf=out_dir_sdf_abs,
            file_set=train_set,
            num_processes=num_processes)

        # for reconstruction (DeepSDF needs SDF samples, not plain points!, take point clouds + new sdf)
        make_sdf_samples(
            in_dir_pts=dir_pc,
            in_dir_normals=dir_normals_abs,
            in_dir_meshes=dir_mesh_repaired_abs,
            out_dir_sdf=out_dir_sdf_abs,
            file_set=test_set,
            num_processes=num_processes)

        # make examples
        create_example(train_set=train_set, test_set=test_set, out_dir_examples=out_dir_examples, dataset=dataset)


if __name__ == "__main__":
    main()
