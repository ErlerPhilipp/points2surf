import numpy as np
import os
import shutil
import random
import configparser

import trimesh
import trimesh.proximity
import trimesh.path
import trimesh.repair
import trimesh.sample
import trimesh.transformations as trafo

from source.base import utils
from source.base import utils_mp
from source.base import file_utils
from source import sdf
from source.base import point_cloud


def _convert_mesh(in_mesh, out_mesh):

    mesh = None
    try:
        mesh = trimesh.load(in_mesh)
    except AttributeError as e:
        print(e)
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)
    except NameError as e:
        print(e)

    if mesh is not None:
        try:
            mesh.export(out_mesh)
        except ValueError as e:
            print(e)


def convert_meshes(in_dir_abs, out_dir_abs, target_file_type: str, num_processes=8):
    """
    Convert a mesh file to another file type.
    :param in_dir_abs:
    :param out_dir_abs:
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
        if file_utils.call_necessary(f, file_out):
            calls.append((f, file_out))

    utils_mp.start_process_pool(_convert_mesh, calls, num_processes)


def _normalize_mesh(file_in, file_out):

    mesh = trimesh.load(file_in)
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

    mesh.export(file_out)


def normalize_meshes(base_dir, in_dir, out_dir, dataset_dir, num_processes=1):
    """
    Translate meshes to origin and scale to unit cube.
    :param base_dir:
    :param in_dir:
    :param filter_dir:
    :param out_dir:
    :param dataset_dir:
    :param num_processes:
    :return:
    """

    in_dir_abs = os.path.join(base_dir, dataset_dir, in_dir)
    out_dir_abs = os.path.join(base_dir, dataset_dir, out_dir)

    os.makedirs(out_dir_abs, exist_ok=True)

    call_params = []

    mesh_files = [f for f in os.listdir(in_dir_abs)
                 if os.path.isfile(os.path.join(in_dir_abs, f))]
    for fi, f in enumerate(mesh_files):
        in_file_abs = os.path.join(in_dir_abs, f)
        out_file_abs = os.path.join(out_dir_abs, f)

        if not file_utils.call_necessary(in_file_abs, out_file_abs):
            continue

        call_params += [(in_file_abs, out_file_abs)]

    utils_mp.start_process_pool(_normalize_mesh, call_params, num_processes)
    
    
def _blensor_vs_to_ws(pts_vs, obj_location, scanner_rotation_inv):

    def _revert_offset(pts_data: np.ndarray, inv_offset: np.ndarray):
        pts_reverted = pts_data
        # don't just check the header because missing rays may be added with NaNs
        if pts_reverted.shape[0] > 0:
            pts_offset_correction = np.broadcast_to(inv_offset, pts_reverted.shape)
            pts_reverted += pts_offset_correction

        return pts_reverted

    # undo coordinate system changes
    pts_ws = utils.right_handed_to_left_handed(pts_vs)

    # move back from camera distance, always along x axis
    _revert_offset(pts_ws, -obj_location)

    # get and apply inverse rotation matrix of camera
    pts_ws = trafo.transform_points(pts_ws, scanner_rotation_inv, translate=False)

    return pts_ws
    

def _pcd_files_to_pts(pcd_files, mesh_file, pts_file_raw_npy, pts_file_npy, pts_file, obj_locations, obj_rotations, hits_per_scan_file, min_pts_size=0, debug=False):
    """
    Convert pcd blensor results to xyz or directly to npy files. Merge front and back scans.
    Stores the number of hits per scan position.
    Moving the object instead of the camera because the point cloud is in some very weird space that behaves
    crazy when the camera moves. A full day wasted on this shit!
    :param pcd_files:
    :param pts_file_npy:
    :param pts_file:
    :param obj_locations:
    :param obj_rotations:
    :param min_pts_size:
    :param debug:
    :return:
    """

    import gzip
    from source.base.point_cloud import get_closest_distance_batched

    # https://www.blensor.org/numpy_import.html
    # timestamp
    # yaw, pitch
    # distance,distance_noise   # distance != 0.0 --> hit
    # x,y,z
    # x_noise,y_noise,z_noise
    # object_id
    # 255*color[0]
    # 255*color[1]
    # 255*color[2]
    # idx
    def extract_hits_from_blensor_numpy(arr_raw):
        hits = arr_raw[arr_raw[:, 3] != 0.0]
        # noisy_xyz = hits[:, [8, 9, 10]]
        return hits
    
    pts_raw_to_cat = []
    pts_ws_to_cat = []
    pts_ws_noisefree_to_cat = []
    hits_per_scan = []
    for fi, f in enumerate(pcd_files):
        try:
            if f.endswith('.numpy') or f.endswith('.numpy.gz'):
                numpy_file = gzip.GzipFile(f, "r") if f.endswith('.numpy.gz') else f
                samples_raw = np.loadtxt(numpy_file, dtype=np.float32)
                samples_hits_vs = extract_hits_from_blensor_numpy(samples_raw)
                pts_raw_to_cat += [samples_raw]
            elif f.endswith('.pcd'):
                samples_hits_vs, header_info = point_cloud.load_pcd(file_in=f)
            else:
                raise ValueError('Input file {} has an unknown format!'.format(f))
        except EOFError as er:
            print('Error processing {}: {}'.format(f, er))
            continue
        
        hits_per_scan.append(samples_hits_vs.shape[0])
        obj_location = np.array(obj_locations[fi])
        scanner_rotation_inv = trafo.quaternion_matrix(trafo.quaternion_conjugate(obj_rotations[fi]))
        
        pts_ws = _blensor_vs_to_ws(samples_hits_vs[:, 8:11], obj_location, scanner_rotation_inv)
        pts_noisefree_ws = _blensor_vs_to_ws(samples_hits_vs[:, 5:8], obj_location, scanner_rotation_inv)
        
        if pts_ws.shape[0] > 0:
            pts_ws_to_cat += [pts_ws]
        
        if pts_noisefree_ws.shape[0] > 0:
            pts_ws_noisefree_to_cat += [pts_noisefree_ws]

        # debug outputs to check the rotations... the pointcloud MUST align exactly with the mesh
        if debug:
            point_cloud.write_ply(file_path=os.path.join('debug', 'test_{}.ply'.format(str(fi))), points=pts_ws)

    pts_data_raw = np.concatenate(tuple(pts_raw_to_cat), axis=0)
    np.savez_compressed(pts_file_raw_npy, pts_data_raw)

    if len(pts_ws_to_cat) == 0:  # no hits
        print('No scanner hits for object {} in {} scans'.format(os.path.basename(mesh_file), len(pcd_files)))
        return
    
    # load corresponding mesh and get face ids closest to noise-free points
    # assign face normals to noisy points
    mesh: trimesh.Trimesh = trimesh.load(mesh_file)
    pts_noisefree_merged = np.concatenate(pts_ws_noisefree_to_cat, axis=0)
    pts_closest, dist, face_ids = get_closest_distance_batched(pts_noisefree_merged, mesh, batch_size=1000, workers=1)
    pts_normals = mesh.face_normals[face_ids]
        
    pts_merged = np.concatenate(pts_ws_to_cat, axis=0)
    pts_merged = np.concatenate((pts_merged, pts_normals), axis=1)  # add normals to noisy points
    np.save(pts_file_npy, pts_merged)

    if pts_merged.shape[0] > min_pts_size:
        point_cloud.write_ply(file_path=pts_file, points=pts_merged[:, :3], normals=pts_merged[:, 3:])

    np.savez_compressed(hits_per_scan_file, hits_per_scan=np.array(hits_per_scan, dtype=np.int32))


def sample_blensor(base_dir, dataset_dir, blensor_bin, dir_in, dir_out_raw, 
                   dir_out, dir_out_vis, dir_out_pcd, dir_out_blensor_scripts, dir_out_locations, dir_out_rotations,
                   num_scans_per_mesh_min, num_scans_per_mesh_max, num_processes, min_pts_size=0,
                   scanner_noise_sigma_min=0.0, scanner_noise_sigma_max=0.05):
    """
    Call Blender to use a Blensor script to sample a point cloud from a mesh
    :param base_dir:
    :param dataset_dir:
    :param dir_in:
    :param dir_out:
    :param dir_blensor_scripts:
    :param num_scans_per_mesh_min: default: 5
    :param num_scans_per_mesh_max: default: 100
    :param scanner_noise_sigma_min: default: 0.0004, rather a lot: 0.01
    :param scanner_noise_sigma_max: default: 0.0004, rather a lot: 0.01
    :return:
    """

    # test blensor scripts with: .\blender -P 00990000_6216c8dabde0a997e09b0f42_trimesh_000.py

    blender_path = os.path.join(base_dir, blensor_bin)
    dir_abs_in = os.path.join(base_dir, dataset_dir, dir_in)
    dir_abs_out = os.path.join(base_dir, dataset_dir, dir_out)
    dir_abs_out_raw = os.path.join(base_dir, dataset_dir, dir_out_raw)
    dir_abs_out_vis = os.path.join(base_dir, dataset_dir, dir_out_vis)
    dir_abs_out_pcd = os.path.join(base_dir, dataset_dir, dir_out_pcd)
    dir_abs_out_blensor = os.path.join(base_dir, dataset_dir, dir_out_blensor_scripts)
    dir_abs_out_locations = os.path.join(base_dir, dataset_dir, dir_out_locations)
    dir_abs_out_rotations = os.path.join(base_dir, dataset_dir, dir_out_rotations)

    os.makedirs(dir_abs_out, exist_ok=True)
    os.makedirs(dir_abs_out_raw, exist_ok=True)
    os.makedirs(dir_abs_out_vis, exist_ok=True)
    os.makedirs(dir_abs_out_pcd, exist_ok=True)
    os.makedirs(dir_abs_out_blensor, exist_ok=True)
    os.makedirs(dir_abs_out_locations, exist_ok=True)
    os.makedirs(dir_abs_out_rotations, exist_ok=True)
    dir_abs_out_hits_per_scan = os.path.join(base_dir, dataset_dir, '04_hits_per_scan')
    os.makedirs(dir_abs_out_hits_per_scan, exist_ok=True)

    with open('blensor_script_template.py', 'r') as file:
        blensor_script_template = file.read()

    blender_blensor_calls = []
    pcd_base_files = []
    pcd_noisy_files = []
    obj_locations = []
    obj_rotations = []

    obj_files = [f for f in os.listdir(dir_abs_in)
                 if os.path.isfile(os.path.join(dir_abs_in, f)) and f[-4:] == '.ply']
    for fi, file in enumerate(obj_files):

        # gather all file names involved in the blensor scanning
        obj_file = os.path.join(dir_abs_in, file)
        blensor_script_file = os.path.join(dir_abs_out_blensor, file[:-4] + '.py')

        new_pcd_base_files = []
        new_pcd_noisy_files = []
        new_obj_locations = []
        new_obj_rotations = []
        rnd = np.random.RandomState(file_utils.filename_to_hash(obj_file))
        num_scans = rnd.randint(num_scans_per_mesh_min, num_scans_per_mesh_max + 1)
        noise_sigma = rnd.rand() * (scanner_noise_sigma_max - scanner_noise_sigma_min) + scanner_noise_sigma_min
        for num_scan in range(num_scans):
            pcd_base_file = os.path.join(
                dir_abs_out_pcd, file[:-4] + '_{num}.numpy.gz'.format(num=str(num_scan).zfill(5)))
            pcd_noisy_file = pcd_base_file[:-9] + '00000.numpy.gz'

            obj_location = (rnd.rand(3) * 2.0 - 1.0)
            obj_location_rand_factors = np.array([0.1, 1.0, 0.1])
            obj_location *= obj_location_rand_factors
            obj_location[1] += 4.0  # offset in cam view dir
            obj_rotation = trafo.random_quaternion(rnd.rand(3))

            # extend lists of pcd output files
            new_pcd_base_files.append(pcd_base_file)
            new_pcd_noisy_files.append(pcd_noisy_file)
            new_obj_locations.append(obj_location.tolist())
            new_obj_rotations.append(obj_rotation.tolist())

        # Save locations and rotations as npz files
        # get arrays with arr = np.load(r'...\00010015_c909a395340949eeb1a90b25_trimesh_001.npz')['locations/rotations']
        locations_npz_file = os.path.join(dir_abs_out_locations, file[:-4] + '.npz')
        rotations_npz_file = os.path.join(dir_abs_out_rotations, file[:-4] + '.npz')
        np.savez_compressed(locations_npz_file, locations=np.array(new_obj_locations))
        np.savez_compressed(rotations_npz_file, rotations=np.array(new_obj_rotations))

        new_scan_sigmas = [noise_sigma] * num_scans

        pcd_base_files.append(new_pcd_base_files)
        pcd_noisy_files.append(new_pcd_noisy_files)
        obj_locations.append(new_obj_locations)
        obj_rotations.append(new_obj_rotations)

        # prepare blensor calls if necessary
        output_files = [os.path.join(dir_abs_out_pcd, os.path.basename(f)) for f in new_pcd_noisy_files]
        output_files += [blensor_script_file, locations_npz_file, rotations_npz_file]
        if file_utils.call_necessary(obj_file, output_files):
            blensor_script = blensor_script_template.format(
                file_loc=obj_file,
                obj_locations=str(new_obj_locations),
                obj_rotations=str(new_obj_rotations),
                evd_files=str(new_pcd_base_files),
                scan_sigmas=str(new_scan_sigmas),
            )
            blensor_script = blensor_script.replace('\\', '/')  # '\' would require escape sequence

            with open(blensor_script_file, "w") as text_file:
                text_file.write(blensor_script)

            # start blender with python script (-P) and close without prompt (-b)
            blender_blensor_call = '{} -P {} -b'.format(blender_path, blensor_script_file)
            blender_blensor_calls.append((blender_blensor_call,))

    utils_mp.start_process_pool(utils_mp.mp_worker, blender_blensor_calls, num_processes)

    def get_pcd_origin_file(pcd_file):
        origin_file = os.path.basename(pcd_file)[:-9] + '.ply'
        origin_file = origin_file.replace('00000.ply', '.ply')
        origin_file = origin_file.replace('_noisy.ply', '.ply')
        origin_file = origin_file.replace('_00000.ply', '.ply')
        return origin_file

    print('### convert pcd to pts')
    call_params = []
    for fi, files in enumerate(pcd_noisy_files):
        mesh_file = os.path.join(dir_abs_in, obj_files[fi])
        pcd_files_abs = [os.path.join(dir_abs_out_pcd, os.path.basename(f)) for f in files]
        pcd_origin = get_pcd_origin_file(files[0])
        xyz_file = os.path.join(dir_abs_out_vis, pcd_origin)
        xyz_raw_npy_file = os.path.join(dir_abs_out_raw, pcd_origin + '.npz')
        xyz_npy_file = os.path.join(dir_abs_out, pcd_origin + '.npy')
        hits_per_scan_file = os.path.join(dir_abs_out_hits_per_scan, pcd_origin + '.npz')

        if file_utils.call_necessary(pcd_files_abs + [mesh_file], [xyz_npy_file, xyz_file, xyz_raw_npy_file, hits_per_scan_file]):
            call_params += [(pcd_files_abs, mesh_file, xyz_raw_npy_file, xyz_npy_file, xyz_file, obj_locations[fi], obj_rotations[fi], hits_per_scan_file, min_pts_size)]

    utils_mp.start_process_pool(_pcd_files_to_pts, call_params, num_processes)


def _clean_mesh(file_in, file_out, num_max_faces=None, enforce_solid=True):
    mesh = trimesh.load(file_in)

    mesh.process()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    if not mesh.is_watertight:
        mesh.fill_holes()
        trimesh.repair.fill_holes(mesh)

    if enforce_solid and not mesh.is_watertight:
        return

    if not mesh.is_winding_consistent:
        trimesh.repair.fix_inversion(mesh, multibody=True)
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_winding(mesh)

    if enforce_solid and not mesh.is_winding_consistent:
        return

    if enforce_solid and not mesh.is_volume:  # watertight, consistent winding, outward facing normals
        return

    # large meshes might cause out-of-memory errors in signed distance calculation
    if num_max_faces is None:
        mesh.export(file_out)
    elif len(mesh.faces) < num_max_faces:
        mesh.export(file_out)


def clean_meshes(base_dir, dataset_dir, dir_in_meshes, dir_out, num_processes, num_max_faces=None, enforce_solid=True):
    """
    Try to repair meshes or filter broken ones. Enforce that meshes are solids to calculate signed distances.
    :param base_dir:
    :param dataset_dir:
    :param dir_in_meshes:
    :param dir_out:
    :param num_processes:
    :param num_max_faces:
    :param enforce_solid:
    :return:
    """

    dir_in_abs = os.path.join(base_dir, dataset_dir, dir_in_meshes)
    dir_out_abs = os.path.join(base_dir, dataset_dir, dir_out)

    os.makedirs(dir_out_abs, exist_ok=True)

    calls = []
    mesh_files = [f for f in os.listdir(dir_in_abs)
                  if os.path.isfile(os.path.join(dir_in_abs, f))]
    files_in_abs = [os.path.join(dir_in_abs, f) for f in mesh_files]
    files_out_abs = [os.path.join(dir_out_abs, f) for f in mesh_files]
    for fi, f in enumerate(mesh_files):
        # skip if result already exists and is newer than the input
        if file_utils.call_necessary(files_in_abs[fi], files_out_abs[fi]):
            calls.append((files_in_abs[fi], files_out_abs[fi], num_max_faces, enforce_solid))

    utils_mp.start_process_pool(_clean_mesh, calls, num_processes)


def _get_and_save_query_pts(
        file_in_mesh: str, file_out_query_pts: str, file_out_query_dist: str, file_out_query_vis: str,
        num_query_pts: int, patch_radius: float,
        far_query_pts_ratio=0.1, signed_distance_batch_size=1000, debug=False):

    import trimesh

    # random state for file name
    rng = np.random.RandomState(file_utils.filename_to_hash(file_in_mesh))

    in_mesh = trimesh.load(file_in_mesh)

    # get query pts
    query_pts_ms = sdf.get_query_pts_for_mesh(
        in_mesh, num_query_pts, patch_radius, far_query_pts_ratio, rng)
    np.save(file_out_query_pts, query_pts_ms.astype(np.float32))

    # get signed distance
    query_dist_ms = sdf.get_signed_distance(
        in_mesh, query_pts_ms, signed_distance_batch_size)
    # fix NaNs, Infs and truncate
    nan_ids = np.isnan(query_dist_ms)
    inf_ids = np.isinf(query_dist_ms)
    query_dist_ms[nan_ids] = 0.0
    query_dist_ms[inf_ids] = 1.0
    query_dist_ms[query_dist_ms < -1.0] = -1.0
    query_dist_ms[query_dist_ms > 1.0] = 1.0
    np.save(file_out_query_dist, query_dist_ms.astype(np.float32))

    if debug and file_out_query_vis is not None:
        # save visualization
        sdf.visualize_query_points(query_pts_ms, query_dist_ms, file_out_query_vis)


def get_query_pts_dist_ms(
        base_dir, dataset_dir, dir_in_mesh,
        dir_out_query_pts_ms,
        dir_out_query_dist_ms,
        dir_out_query_vis,
        patch_radius,
        num_query_pts=2000,
        far_query_pts_ratio=0.1,
        signed_distance_batch_size=1000,
        num_processes=8,
        debug=False):
    """
    Get query points and their GT signed distances in model space.
    :param base_dir:
    :param dataset_dir:
    :param dir_in_mesh:
    :param dir_out_query_pts_ms:
    :param dir_out_query_dist_ms:
    :param dir_out_query_vis:
    :param patch_radius:
    :param num_query_pts:
    :param far_query_pts_ratio:
    :param signed_distance_batch_size:
    :param num_processes:
    :param debug:
    :return:
    """

    import os.path
    from source.base import file_utils

    dir_in_mesh_abs = os.path.join(base_dir, dataset_dir, dir_in_mesh)
    dir_out_query_pts_abs = os.path.join(base_dir, dataset_dir, dir_out_query_pts_ms)
    dir_out_query_dist_abs = os.path.join(base_dir, dataset_dir, dir_out_query_dist_ms)
    dir_out_query_vis_abs = os.path.join(base_dir, dataset_dir, dir_out_query_vis)

    os.makedirs(dir_out_query_pts_abs, exist_ok=True)
    os.makedirs(dir_out_query_dist_abs, exist_ok=True)
    if debug:
        os.makedirs(dir_out_query_vis_abs, exist_ok=True)

    # get query points
    print('### get query points')
    call_params = []
    files_mesh = [f for f in os.listdir(dir_in_mesh_abs)
                  if os.path.isfile(os.path.join(dir_in_mesh_abs, f)) and f[-4:] == '.ply']
    for fi, f in enumerate(files_mesh):
        file_in_mesh = os.path.join(dir_in_mesh_abs, f)
        file_out_query_pts = os.path.join(dir_out_query_pts_abs, f + '.npy')
        file_out_query_dist = os.path.join(dir_out_query_dist_abs, f + '.npy')
        file_out_query_vis = os.path.join(dir_out_query_vis_abs, f + '.ply')

        if file_utils.call_necessary(file_in_mesh, [file_out_query_pts, file_out_query_dist]):
            call_params.append((file_in_mesh, file_out_query_pts, file_out_query_dist, file_out_query_vis,
                                num_query_pts, patch_radius, far_query_pts_ratio,
                                signed_distance_batch_size, debug))

    utils_mp.start_process_pool(_get_and_save_query_pts, call_params, num_processes)


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


def clean_up_broken_inputs(base_dir, dataset_dir, final_out_dir, final_out_extension,
                           clean_up_dirs, broken_dir='broken'):
    """
    Assume that the file stem (excluding path and everything after the first '.') is a unique identifier in
    multiple directories.
    """

    final_out_dir_abs = os.path.join(base_dir, dataset_dir, final_out_dir)
    final_output_files = [f for f in os.listdir(final_out_dir_abs)
                          if os.path.isfile(os.path.join(final_out_dir_abs, f)) and
                          (final_out_extension is None or f[-len(final_out_extension):] == final_out_extension)]

    if len(final_output_files) == 0:
        print('Warning: Output dir "{}" is empty'.format(final_out_dir_abs))
        return

    # move inputs and intermediate results that have no final output
    final_output_file_stems = set(tuple([f.split('.', 1)[0] for f in final_output_files]))
    # final_output_file_stem_lengths = [len(f.split('.', 1)[0]) for f in final_output_files]
    # num_final_output_file_stem_lengths = len(set(final_output_file_stem_lengths))
    # inconsistent_file_length = num_final_output_file_stem_lengths > 1
    # if inconsistent_file_length:
    #     print('WARNING: output files don\'t have consistent length. Clean-up broken inputs may do unwanted things.')
    for clean_up_dir in clean_up_dirs:
        dir_abs = os.path.join(base_dir, dataset_dir, clean_up_dir)
        if not os.path.isdir(dir_abs):
            continue
        dir_files = [f for f in os.listdir(dir_abs) if os.path.isfile(os.path.join(dir_abs, f))]
        dir_file_stems = [f.split('.', 1)[0] for f in dir_files]
        dir_file_stems_without_final_output = [f not in final_output_file_stems for f in dir_file_stems]
        dir_files_without_final_output = np.array(dir_files)[dir_file_stems_without_final_output]

        broken_dir_abs = os.path.join(base_dir, dataset_dir, broken_dir, clean_up_dir)
        broken_files = [os.path.join(broken_dir_abs, f) for f in dir_files_without_final_output]

        for fi, f in enumerate(dir_files_without_final_output):
            os.makedirs(broken_dir_abs, exist_ok=True)
            shutil.move(os.path.join(dir_abs, f), broken_files[fi])


def write_dataset_csv(base_dir, dataset_dir, pts_dir, final_out_dir):

    # get pts_file, dist_file, num_points
    dist_dir_abs = os.path.join(base_dir, dataset_dir, final_out_dir)
    pts_dir_abs = os.path.join(base_dir, dataset_dir, pts_dir)
    pts_files = [f for f in os.listdir(pts_dir_abs)
                if os.path.isfile(os.path.join(pts_dir_abs, f)) and f[-4:] == '.xyz']
    pts_file_stems = set([f.split('.', 1)[0] for f in pts_files])
    dist_files = [f for f in os.listdir(dist_dir_abs)
                 if os.path.isfile(os.path.join(dist_dir_abs, f)) and f[-13:] == '.xyz.dist.npz']

    csv_data = []
    for dist_file in dist_files:
        dist_file_stem = dist_file.split('.', 1)[0]
        dist_file_abs = os.path.join(dist_dir_abs, dist_file)
        if dist_file_stem in pts_file_stems:
            pts_file = dist_file_stem + '.xyz'
            num_points = file_utils.load_npz(dist_file_abs).shape[0]
            csv_data.append((pts_file, dist_file, str(num_points)))

    csv_file = os.path.join(base_dir, dataset_dir, 'dataset_stats.csv')
    file_utils.make_dir_for_file(csv_file)
    nl = '\n'
    csv_lines = [','.join(item) for item in csv_data]
    csv_lines_str = nl.join(csv_lines)
    with open(csv_file, "w") as text_file:
        text_file.write(csv_lines_str)


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
    """
    This is meant to test the reconstruction from GT signed distances.
    Requires dense query points and SDs near the surface.
    :param base_dir:
    :param dataset_dir:
    :param pts_dir:
    :param p_ids_grid_dir:
    :param query_dist_dir:
    :param query_pts_dir:
    :param gt_reconstruction_dir:
    :param grid_resolution:
    :param sigma:
    :param certainty_threshold:
    :param num_processes:
    :return:
    """

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


def read_config(config, config_file):
    if os.path.isfile(config_file):
        config.read(config_file)
    else:
        print("""
ERROR: No config file found. Create a 'settings.ini' in the dataset directory with these contents:
[general]
only_for_evaluation = 0
grid_resolution = 256
epsilon = 3
num_scans_per_mesh_min = 5
num_scans_per_mesh_max = 30
scanner_noise_sigma = 0.01
        """)


def make_dataset(dataset_name: str, blensor_bin: str, base_dir: str, num_processes=7,
                 seed=42, num_query_points_per_shape=2000):
    """
    Make dataset from meshes.
    :param dataset_name:
    :param blensor_bin:
    :param base_dir:
    :param num_processes: 16 processes need up to 64 GB RAM for the signed distances
    :param seed: only used for the dataset splits
    :param num_query_points_per_shape: should be greater than the training parameter 'patches_per_shape'
    :return:
    """

    dataset_dir = dataset_name

    config_file = os.path.join(base_dir, dataset_dir, 'settings.ini')
    config = configparser.ConfigParser()
    read_config(config, config_file)
    print('Processing dataset: ' + config_file)

    # no signed distances needed, therefore less strict requirements for input meshes
    only_for_evaluation = bool(int(config['general']['only_for_evaluation']))  # default: false
    grid_resolution = int(config['general']['grid_resolution'])  # default: 128  e.g. for marching cubes reconstruction
    epsilon = int(config['general']['epsilon'])  # default: 3
    num_scans_per_mesh_min = int(config['general']['num_scans_per_mesh_min'])  # default: 5
    num_scans_per_mesh_max = int(config['general']['num_scans_per_mesh_max'])  # default: 30
    scanner_noise_sigma_min = float(config['general']['scanner_noise_sigma_min'])  # default: 0.0004, rather a lot: 0.01
    scanner_noise_sigma_max = float(config['general']['scanner_noise_sigma_max'])  # default: 0.0004, rather a lot: 0.01

    patch_radius = point_cloud.get_patch_radius(grid_resolution, epsilon)

    # the user might have removed unwanted input meshes after some processing
    # this moves (intermediate) outputs that don't have inputs anymore to the 'broken' dir
    filter_broken_inputs = True

    dirs_to_clean = \
        ['00_base_meshes',
         '01_base_meshes_ply',
         '02_meshes_cleaned',
         '03_meshes',
         '04_pts', '04_pts_raw', '04_pts_vis', '04_blensor_py', '04_locations', '04_rotations',  
         # not '04_pcd' since it has multiple files per input file, the association is not implemented
         '05_patch_dists', '05_patch_ids', '05_query_dist', '05_query_pts',
         '05_patch_ids_grid', '05_query_pts_grid', '05_query_dist_grid',
         '06_poisson_rec', '06_mc_gt_recon', '06_poisson_rec_gt_normals',
         '06_normals', '06_normals/pts', '06_dist_from_p_normals']

    if filter_broken_inputs:
        clean_up_broken_inputs(base_dir=base_dir, dataset_dir=dataset_dir,
                               final_out_dir='00_base_meshes', final_out_extension=None,
                               clean_up_dirs=dirs_to_clean, broken_dir='broken')

    print('### convert base meshes to ply')
    convert_meshes(in_dir_abs=os.path.join(base_dir, dataset_dir, '00_base_meshes'),
                   out_dir_abs=os.path.join(base_dir, dataset_dir, '01_base_meshes_ply'),
                   target_file_type='.ply', num_processes=num_processes)

    if filter_broken_inputs:
        clean_up_broken_inputs(base_dir=base_dir, dataset_dir=dataset_dir,
                               final_out_dir='01_base_meshes_ply', final_out_extension='.ply',
                               clean_up_dirs=dirs_to_clean, broken_dir='broken')

    print('### clean mesh')
    clean_meshes(base_dir=base_dir, dataset_dir=dataset_dir,
                 dir_in_meshes='01_base_meshes_ply', dir_out='02_meshes_cleaned', num_processes=num_processes,
                 num_max_faces=None if only_for_evaluation else 50000,
                 enforce_solid=False if only_for_evaluation else True)  # wrong results for 5 famous objects?

    if filter_broken_inputs:
        clean_up_broken_inputs(base_dir=base_dir, dataset_dir=dataset_dir,
                               final_out_dir='02_meshes_cleaned', final_out_extension='.ply',
                               clean_up_dirs=dirs_to_clean, broken_dir='broken')

    print('### scale and translate mesh')
    normalize_meshes(base_dir=base_dir, in_dir='02_meshes_cleaned', out_dir='03_meshes', dataset_dir=dataset_dir,
                     num_processes=num_processes)

    print('### sample with Blensor')
    sample_blensor(base_dir=base_dir, dataset_dir=dataset_dir, blensor_bin=blensor_bin,
                   dir_in='03_meshes', dir_out_raw='04_pts_raw', dir_out='04_pts', dir_out_vis='04_pts_vis', dir_out_pcd='04_pcd',
                   dir_out_blensor_scripts='04_blensor_py', dir_out_locations='04_locations', dir_out_rotations='04_rotations',
                   num_scans_per_mesh_min=num_scans_per_mesh_min, num_scans_per_mesh_max=num_scans_per_mesh_max,
                   num_processes=num_processes,
                   min_pts_size=0 if only_for_evaluation else 100,
                   scanner_noise_sigma_min=scanner_noise_sigma_min, scanner_noise_sigma_max=scanner_noise_sigma_max)

    if filter_broken_inputs:
        clean_up_broken_inputs(base_dir=base_dir, dataset_dir=dataset_dir,
                               final_out_dir='04_pts', final_out_extension='.xyz.npy',
                               clean_up_dirs=dirs_to_clean, broken_dir='broken')

    if not only_for_evaluation:
        print('### make query points, calculate signed distances')
        dir_mesh = '03_meshes'
        dir_query_dist = '05_query_dist'
        dir_query_pts_ms = '05_query_pts'
        dir_out_query_vis = '05_query_vis'  # None to disable
        far_query_pts_ratio = 0.5  # 0.1  # not too little or the network fails at the inside classification
        get_query_pts_dist_ms(
            base_dir=base_dir, dataset_dir=dataset_dir, dir_in_mesh=dir_mesh,
            dir_out_query_pts_ms=dir_query_pts_ms,
            dir_out_query_dist_ms=dir_query_dist,
            dir_out_query_vis=dir_out_query_vis,
            patch_radius=patch_radius,
            num_query_pts=num_query_points_per_shape,
            far_query_pts_ratio=far_query_pts_ratio,
            signed_distance_batch_size=500,
            num_processes=num_processes,
            debug=True)

        print('### statistics and clean up')
        if filter_broken_inputs:
            clean_up_broken_inputs(base_dir=base_dir, dataset_dir=dataset_dir,
                                   final_out_dir='05_query_pts' if only_for_evaluation else '05_query_dist',
                                   final_out_extension='.npy',
                                   clean_up_dirs=dirs_to_clean, broken_dir='broken')

    make_dataset_splits(base_dir=base_dir, dataset_dir=dataset_dir,
                        final_out_dir='04_pts' if only_for_evaluation else'05_query_pts',
                        seed=seed, only_test_set=only_for_evaluation, testset_ratio=0.1)


def _test_dataset(file_name = '00010011_83bf511191ef439b90a1b26c_trimesh_001'):
    """
    Assemble data from one point cloud for verification.
    """
    import os
    import numpy as np
    from trimesh import transformations as trafo
    from source.base import file_utils, point_cloud

    base_dir = r'E:\datasets\p2s_pps\paropt'
    dataset_name = 'abc_train_normals'
    dataset_dir = os.path.join(base_dir, dataset_name)
    output_dir = os.path.join(dataset_dir, '_debug')
    output_file = os.path.join(output_dir, file_name + '.ply')

    # pcd_file = os.path.join(dataset_dir, '04_pcd', file_name + '.numpy.gz')
    # mesh_file = os.path.join(dataset_dir, '03_meshes', file_name + '.ply')
    # pts_file_raw_npy = os.path.join(dataset_dir, '04_pts_raw', file_name + '.npy')
    # pts_file_npy = os.path.join(dataset_dir, '04_pts', file_name + '.npy')
    pts_file = os.path.join(dataset_dir, '04_pts_vis', file_name + '.ply')
    hits_per_scan_file = os.path.join(dataset_dir, '04_hits_per_scan', file_name + '.ply.npz')
    rotations_file = os.path.join(dataset_dir, '04_rotations', file_name + '.npz')
    locations_file = os.path.join(dataset_dir, '04_locations', file_name + '.npz')

    # get points and normals, special case see https://github.com/mikedh/trimesh/issues/1192
    pointcloud = trimesh.load_mesh(pts_file)
    pointcloud = pointcloud.metadata['_ply_raw']['vertex']['data']
    pts = np.stack((pointcloud['x'], pointcloud['y'], pointcloud['z']), axis=-1)
    normals = np.stack((pointcloud['nx'], pointcloud['ny'], pointcloud['nz']), axis=-1)

    # get hits per scan contains the number of points per scan
    hits_per_scan = np.load(hits_per_scan_file)['hits_per_scan']
    
    # assemble scanner matrices from object rotations and locations in view space of those scans
    obj_rot_vs = np.load(rotations_file)['rotations']
    obj_loc_vs = np.load(locations_file)['locations']
    scanner_loc_ws: list[np.ndarray] = []
    for i in range(len(obj_rot_vs)):
        obj_rot_mat = trafo.quaternion_matrix(obj_rot_vs[i])
        scanner_loc_ws.append(_blensor_vs_to_ws(np.zeros((1, 3)), obj_loc_vs[i], obj_rot_mat))

    # calculate the direction to the scanner for each point
    points = pts.copy()
    colors = normals.copy()
    dir_to_scanner = []
    pts_hit_start_id = 0
    for i in range(len(scanner_loc_ws)):
        hits_this_scan = hits_per_scan[i]
        pts_per_scan = points[pts_hit_start_id:pts_hit_start_id + hits_this_scan]
        pts_hit_start_id += hits_this_scan
        num_points = hits_this_scan
        scanner_locations = scanner_loc_ws[i]
        scanner_locations_duplicate = np.tile(scanner_locations, (num_points, 1))
        dir_to_this_scanner = scanner_locations_duplicate - pts_per_scan
        dir_to_this_scanner = dir_to_this_scanner / np.linalg.norm(dir_to_this_scanner, axis=1, keepdims=True)  # normalize
        dir_to_scanner.append(dir_to_this_scanner)
    dir_to_scanner = np.concatenate(dir_to_scanner)

    # write output file
    file_utils.make_dir_for_file(output_file)
    point_cloud.write_ply(file_path=output_file, points=points, normals=dir_to_scanner, colors=colors)
    

if __name__ == "__main__":
    
    # test the camera positions for a few files
    # file_names = [
    #     '00010006_7e4956ae07e24f6584127385_trimesh_000',
    #     '00010009_d97409455fa543b3a224250f_trimesh_000',
    #     '00010011_83bf511191ef439b90a1b26c_trimesh_001',
    #     '00010015_c909a395340949eeb1a90b25_trimesh_001',
    #     '00010016_f9191e596ce343e1b16f56f8_trimesh_000',
    #     '00010045_75f31cb4dff84986aadc622b_trimesh_006',
    # ]
    # for file_name in file_names:
    #     _test_dataset(file_name=file_name)
    # # _test_dataset()
    # exit(0)
    
    blensor_bin = "bin/Blensor-x64.AppImage"
    base_dir = 'datasets'
    # blensor_bin = r'E:\binaries\blensor_win\blender.exe'
    # base_dir = r'E:\datasets\p2s_pps\normals'
    num_processes = 7
    datasets = [
        # 'abc_train_normals', 
        'abc', 
        'abc_extra_noisy', 'abc_noisefree',
        'famous_original', 'famous_noisefree', 'famous_dense', 'famous_extra_noisy', 'famous_sparse',
        'thingi10k_scans_original', 'thingi10k_scans_dense', 'thingi10k_scans_sparse',
        'thingi10k_scans_extra_noisy', 'thingi10k_scans_noisefree'
    ]

    for d in datasets:
        make_dataset(dataset_name=d, blensor_bin=blensor_bin, base_dir=base_dir, num_processes=num_processes)
