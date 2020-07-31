import time

import numpy as np
import trimesh

from source.base import utils


def make_sample_points_for_3d_grid_unit_cube(grid_resolution):
    # convert from unit cube to voxel center cells (half a cell is missing at min and max)
    voxel_size = 1.0 / grid_resolution
    linspace_unit_cube = np.linspace(-1.0, 1.0 - voxel_size, grid_resolution, dtype=np.float32)
    x, y, z = np.meshgrid(linspace_unit_cube, linspace_unit_cube, linspace_unit_cube)
    flat_expand = lambda x: np.expand_dims(x.flatten(), axis=1)
    query_pts = np.concatenate((flat_expand(y), flat_expand(x), flat_expand(z)), axis=1)
    query_pts += voxel_size * 0.5
    return query_pts


def get_voxel_centers_grid(pts, grid_resolution, k, distance_threshold_ms=None, num_processes=1):

    import sys
    import scipy.spatial as spatial

    # kd-tree for point cloud
    leaf_size = 100
    sys.setrecursionlimit(int(max(1000, round(pts.shape[0] / leaf_size))))
    kdtree = spatial.cKDTree(pts, leaf_size)

    # get grid voxel centers
    grid_pts_ms = make_sample_points_for_3d_grid_unit_cube(grid_resolution)

    # filter out voxel centers that are far away from the point cloud
    # try to keep at least one voxel inside the surface and one outside
    if distance_threshold_ms is None:
        distance_threshold_ms = 1.0 / grid_resolution * 4.0  # larger to prevent holes in the volume
    grid_pts_neighbor_ids = kdtree.query_ball_point(grid_pts_ms, distance_threshold_ms, n_jobs=num_processes)
    grid_pts_has_close_neighbor = np.array([bool(len(ids_list)) for ids_list in grid_pts_neighbor_ids])
    grid_pts_near_surf_ms = grid_pts_ms[grid_pts_has_close_neighbor]

    # get patch points
    patch_pts_dists, patch_pts_ids = kdtree.query(grid_pts_near_surf_ms, k, n_jobs=num_processes)
    return grid_pts_near_surf_ms, patch_pts_dists, patch_pts_ids


def get_voxel_centers_grid_smaller_pc(pts, grid_resolution, distance_threshold_vs=10):
    """
    This is more efficient when the point cloud is considerable smaller than the grid. n_p << grid_res^3
    :param pts:
    :param grid_resolution:
    :param distance_threshold_vs:
    :return:
    """
    from scipy.ndimage.filters import convolve

    # get grid voxel centers
    pts_vs = model_space_to_volume_space(pts, grid_resolution)
    query_pts_vol = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.float32)
    query_pts_vol[pts_vs[:, 0], pts_vs[:, 1], pts_vs[:, 2]] = 1.0

    # expand query points from the known necessary points
    kernel = np.ones((distance_threshold_vs, distance_threshold_vs, distance_threshold_vs), dtype=np.float32)
    grid_pts_near_surf_vol = convolve(query_pts_vol, kernel, mode='nearest')

    # convert volume to points in model space
    grid_pts_near_surf_vs = np.nonzero(grid_pts_near_surf_vol[:-1, :-1, :-1])
    grid_pts_near_surf_vs = np.stack(grid_pts_near_surf_vs, axis=1)
    grid_pts_near_surf_ms = volume_space_to_model_space(grid_pts_near_surf_vs, grid_resolution)

    return grid_pts_near_surf_ms.astype(np.float32)


def model_space_to_volume_space(pts_ms, vol_res):
    pts_pos_octant = (pts_ms + 1.0) / 2.0
    return np.floor(pts_pos_octant * vol_res).astype(np.int)


def volume_space_to_model_space(pts_vs, vol_res):
    return ((pts_vs + 0.5) / vol_res) * 2.0 - 1.0


def add_samples_to_volume(vol, pos_ms, val):
    """
    add samples, average multiple values per voxel
    :param vol:
    :param pos_ms:
    :param val:
    :return:
    """

    # get distance between samples and their corresponding voxel centers
    pos_vs = model_space_to_volume_space(pos_ms, vol.shape[0])
    # grid_cell_centers_ms = volume_space_to_model_space(pos_vs, vol.shape[0])
    grid_cell_centers_ms = pos_ms
    dist_pos_cell_center = utils.cartesian_dist(pos_ms, grid_cell_centers_ms)

    # cluster by voxel
    unique_grid_pos, unique_counts = np.unique(pos_vs, return_counts=True, axis=0)
    values_per_voxel = np.split(val, np.cumsum(unique_counts)[:-1])
    dist_pos_cell_center_per_voxel = np.split(dist_pos_cell_center, np.cumsum(unique_counts)[:-1])
    coordinates_per_voxel = np.split(pos_vs, np.cumsum(unique_counts)[:-1])
    coordinates_per_voxel_first = np.array([c[0] for c in coordinates_per_voxel])

    # get sample closest to voxel center
    dist_pos_cell_center_per_voxel_arg_min = [np.argmin(voxel_data) for voxel_data in dist_pos_cell_center_per_voxel]
    # values_per_voxel_mean = np.array([v.mean() for v in values_per_voxel])  # TODO: weight by distance to voxel center
    values_per_voxel_closest = np.array([v[dist_pos_cell_center_per_voxel_arg_min[vi]]
                                        for vi, v in enumerate(values_per_voxel)])
    vol[coordinates_per_voxel_first[:, 0], coordinates_per_voxel_first[:, 1], coordinates_per_voxel_first[:, 2]] = \
        values_per_voxel_closest  # values_per_voxel_mean
    return vol


def propagate_sign(vol, sigma=5, certainty_threshold=13):
    """
    iterative propagation of SDF signs from 'seed' voxels to get a dense, truncated volume
    :param vol:
    :param certainty_threshold: int in (0..5^3]
    :param sigma: neighborhood of propagation (kernel size)
    :return:
    """

    from scipy.ndimage.filters import convolve

    # # remove disconnected voxels as pre-processing
    # # e.g. a single outside voxel surrounded by inside voxels
    # sigma_disc = 3
    # kernel_neighbor_sum = np.ones((sigma_disc, sigma_disc, sigma_disc), dtype=np.float32)
    # kernel_center = int(sigma_disc / 2)
    # kernel_neighbor_sum[kernel_center, kernel_center, kernel_center] = 0.0
    # neighbor_sum = convolve(np.sign(vol), kernel_neighbor_sum, mode='nearest')
    # num_neighbors = sigma_disc**3.0 - 1.0
    # neighbors_have_same_sign = np.abs(neighbor_sum) == int(num_neighbors)
    # disconnected_voxels = np.logical_and(neighbors_have_same_sign, np.sign(neighbor_sum) != np.sign(vol))
    # vol[disconnected_voxels] = neighbor_sum[disconnected_voxels] / num_neighbors

    # smoothing as pre-processing helps with the reconstruction noise
    # quality should be ok as long as the surface stays at the same place
    # over-smoothing reduces the overall quality
    # sigma_smooth = 3
    # kernel_neighbor_sum = np.ones((sigma_smooth, sigma_smooth, sigma_smooth), dtype=np.float32)
    # vol = convolve(vol, kernel_neighbor_sum, mode='nearest') / sigma_smooth**3

    vol_sign_propagated = np.sign(vol)
    unknown_initially = vol_sign_propagated == 0
    kernel = np.ones((sigma, sigma, sigma), dtype=np.float32)

    # assume borders to be outside, reduce number of iterations
    vol[+0, :, :] = -1.0
    vol[-1, :, :] = -1.0
    vol[:, +0, :] = -1.0
    vol[:, -1, :] = -1.0
    vol[:, :, +0] = -1.0
    vol[:, :, -1] = -1.0

    while True:
        unknown_before = vol_sign_propagated == 0
        if unknown_before.sum() == 0:
            break

        # sum sign of neighboring voxels
        vol_sign_propagated_new = convolve(vol_sign_propagated, kernel, mode='nearest')

        # update only when the sign is certain
        vol_sign_propagated_new_unsure = np.abs(vol_sign_propagated_new) < certainty_threshold
        vol_sign_propagated_new[vol_sign_propagated_new_unsure] = 0.0

        # map again to [-1.0, 0.0, +1.0]
        vol_sign_propagated_new = np.sign(vol_sign_propagated_new)

        # stop when no changes happen
        unknown_after = vol_sign_propagated_new == 0
        if unknown_after.sum() >= unknown_before.sum():
            break  # no changes -> some values might be caught in a tie
        vol_sign_propagated[unknown_initially] = vol_sign_propagated_new[unknown_initially]  # add new values

    vol[vol == 0] = vol_sign_propagated[vol == 0]
    return vol


def implicit_surface_to_mesh(query_dist_ms, query_pts_ms,
                             volume_out_file, mc_out_file, grid_res, sigma, certainty_threshold=26):

    from source.base import mesh_io
    from source.base import file_utils

    if query_dist_ms.max() == 0.0 and query_dist_ms.min() == 0.0:
        print('WARNING: implicit surface for {} contains only zeros'.format(volume_out_file))
        return

    # add known values and propagate their signs to unknown values
    volume = np.zeros((grid_res, grid_res, grid_res))
    volume = add_samples_to_volume(volume, query_pts_ms, query_dist_ms)

    start = time.time()
    volume = propagate_sign(volume, sigma, certainty_threshold)
    end = time.time()
    print('Sign propagation took: {}'.format(end - start))

    # clamp to -1..+1
    volume[volume < -1.0] = -1.0
    volume[volume > 1.0] = 1.0

    # green = inside; red = outside
    query_dist_ms_norm = query_dist_ms / np.max(np.abs(query_dist_ms))
    query_pts_color = np.zeros((query_dist_ms_norm.shape[0], 3))
    query_pts_color[query_dist_ms_norm < 0.0, 0] = np.abs(query_dist_ms_norm[query_dist_ms_norm < 0.0]) + 1.0 / 2.0
    query_pts_color[query_dist_ms_norm > 0.0, 1] = query_dist_ms_norm[query_dist_ms_norm > 0.0] + 1.0 / 2.0
    mesh_io.write_off(volume_out_file, query_pts_ms, np.array([]), colors_vertex=query_pts_color)

    if volume.min() < 0.0 and volume.max() > 0.0:
        # reconstruct mesh from volume using marching cubes
        from skimage import measure
        start = time.time()
        v, f, normals, values = measure.marching_cubes_lewiner(volume, 0)
        end = time.time()
        print('Marching Cubes Lewiner took: {}'.format(end - start))

        if v.size == 0 and f.size == 0:
            print('Warning: marching cubes gives no result!')
        else:
            import trimesh
            import trimesh.repair
            v = (((v + 0.5) / float(grid_res)) - 0.5) * 2.0
            mesh = trimesh.Trimesh(vertices=v, faces=f)
            trimesh.repair.fix_inversion(mesh)
            file_utils.make_dir_for_file(mc_out_file)
            mesh.export(mc_out_file)
    else:
        print('Warning: volume for marching cubes contains no 0-level set!')


def implicit_surface_to_mesh_file(query_dist_ms_file, query_pts_ms_file,
                                  volume_out_file, mc_out_file, grid_res, sigma, certainty_threshold):
    query_dist_ms = np.load(query_dist_ms_file)
    query_pts_ms = np.load(query_pts_ms_file)
    implicit_surface_to_mesh(query_dist_ms, query_pts_ms,
                             volume_out_file, mc_out_file, grid_res, sigma, certainty_threshold)


def implicit_surface_to_mesh_directory(imp_surf_dist_ms_dir, query_pts_ms_dir,
                                       vol_out_dir, mesh_out_dir,
                                       grid_res, sigma, certainty_threshold, num_processes=1):

    import os
    from source.base import file_utils
    from source.base import utils_mp

    os.makedirs(vol_out_dir, exist_ok=True)
    os.makedirs(mesh_out_dir, exist_ok=True)

    calls = []
    dist_ps_files = [f for f in os.listdir(imp_surf_dist_ms_dir)
                     if os.path.isfile(os.path.join(imp_surf_dist_ms_dir, f)) and f[-8:] == '.xyz.npy']
    files_dist_ms_in_abs = [os.path.join(imp_surf_dist_ms_dir, f) for f in dist_ps_files]
    files_query_pts_ms_in_abs = [os.path.join(query_pts_ms_dir, f) for f in dist_ps_files]
    files_vol_out_abs = [os.path.join(vol_out_dir, f[:-8] + '.off') for f in dist_ps_files]
    files_mesh_out_abs = [os.path.join(mesh_out_dir, f[:-8] + '.ply') for f in dist_ps_files]
    for fi, f in enumerate(dist_ps_files):
        # skip if result already exists and is newer than the input
        if file_utils.call_necessary([files_dist_ms_in_abs[fi], files_query_pts_ms_in_abs[fi]],
                                     [files_vol_out_abs[fi], files_mesh_out_abs[fi]]):
            calls.append((files_dist_ms_in_abs[fi], files_query_pts_ms_in_abs[fi],
                          files_vol_out_abs[fi], files_mesh_out_abs[fi], grid_res, sigma, certainty_threshold))

    utils_mp.start_process_pool(implicit_surface_to_mesh_file, calls, num_processes)


def visualize_query_points(query_pts_ms, query_dist_ms, file_out_off):

    import trimesh

    # grey-scale distance
    query_dist_abs_ms = np.abs(query_dist_ms)
    query_dist_abs_normalized_ms = query_dist_abs_ms / query_dist_abs_ms.max()

    # red-green: inside-outside
    query_dist_col = np.zeros((query_dist_ms.shape[0], 3))
    pos_dist = query_dist_ms < 0.0
    neg_dist = query_dist_ms > 0.0
    query_dist_col[pos_dist, 0] = 0.5 + 0.5 * query_dist_abs_normalized_ms[pos_dist]
    query_dist_col[neg_dist, 1] = 0.5 + 0.5 * query_dist_abs_normalized_ms[neg_dist]

    mesh = trimesh.Trimesh(vertices=query_pts_ms, vertex_colors=query_dist_col)
    mesh.export(file_out_off)


def get_query_pts_for_mesh(in_mesh: trimesh.Trimesh, num_query_pts: int, patch_radius: float,
                           far_query_pts_ratio=0.1, rng=np.random.RandomState()):
    # assume mesh to be centered around the origin
    import trimesh.proximity

    def _get_points_near_surface(mesh: trimesh.Trimesh):
        samples, face_id = mesh.sample(num_query_pts_close, return_index=True)
        offset_factor = (rng.random(size=(num_query_pts_close,)) - 0.5) * 2.0 * patch_radius
        sample_normals = mesh.face_normals[face_id]
        sample_normals_len = np.sqrt(np.linalg.norm(sample_normals, axis=1))
        sample_normals_len_broadcast = np.broadcast_to(np.expand_dims(sample_normals_len, axis=1), sample_normals.shape)
        sample_normals_normalized = sample_normals / sample_normals_len_broadcast
        offset_factor_broadcast = np.broadcast_to(np.expand_dims(offset_factor, axis=1), sample_normals.shape)
        noisy_samples = samples + offset_factor_broadcast * sample_normals_normalized
        return noisy_samples

    num_query_pts_far = int(num_query_pts * far_query_pts_ratio)
    num_query_pts_close = num_query_pts - num_query_pts_far

    in_mesh.fix_normals()
    query_pts_close = _get_points_near_surface(in_mesh)

    # add e.g. 10% samples that may be far from the surface
    query_pts_far = (rng.random(size=(num_query_pts_far, 3))) - 0.5

    query_pts = np.concatenate((query_pts_far, query_pts_close), axis=0)

    return query_pts


def get_signed_distance(in_mesh: trimesh.Trimesh, query_pts_ms,
                        signed_distance_batch_size=1000):

    import trimesh.proximity

    # process batches because trimesh's signed_distance very inefficient on memory
    # 3k queries on a mesh with 27k vertices and 55k faces takes around 8 GB of RAM
    dists_ms = np.zeros((query_pts_ms.shape[0],))
    pts_ids = np.arange(query_pts_ms.shape[0])
    pts_ids_split = np.array_split(pts_ids, max(1, int(query_pts_ms.shape[0] / signed_distance_batch_size)))
    for pts_ids_batch in pts_ids_split:
        dists_ms[pts_ids_batch] = trimesh.proximity.signed_distance(in_mesh, query_pts_ms[pts_ids_batch])

    nan_ids = np.isnan(dists_ms)
    inf_ids = np.isinf(dists_ms)
    num_nans = nan_ids.sum()
    num_infs = inf_ids.sum()

    if num_nans > 0 or num_infs > 0:
        print('Error: Encountered {} NaN and {} Inf values in signed distance of {}.'.format(
            num_nans, num_infs, query_pts_ms))
        # # debug output of NaNs
        # # repeat and log error
        # trimesh.util.attach_to_log()
        # dists_ms = prox.signed_distance(mesh, pts_query)
        # dists_ms[nan_ids] = 0.0
        # dists_ms[inf_ids] = 1.0
        # replaced_ids = np.nonzero(np.logical_or(nan_ids, inf_ids))
        # print('Error: Replacing {} with zeros and ones.'.format(replaced_ids))

    return dists_ms
