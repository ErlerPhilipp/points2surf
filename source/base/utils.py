import numpy as np
import os

from source.base import utils_mp
from source.base import file_utils


def cartesian_dist(vec_x: np.array, vec_y: np.array, axis=1) -> np.ndarray:
    dist = np.linalg.norm(vec_x - vec_y, axis=axis)
    return dist


def batch_quat_to_rotmat(q, out=None):
    """
    quaternion a + bi + cj + dk should be given in the form [a,b,c,d]
    :param q:
    :param out:
    :return:
    """

    import torch

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2 / torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out


def is_matrix_symmetric(matrix):
    return (matrix != matrix.transpose()).count_nonzero() == 0


def right_handed_to_left_handed(pts: np.ndarray):
    pts_res = np.zeros_like(pts)
    if pts.shape[0] > 0:
        pts_res[:, 0] = pts[:, 0]
        pts_res[:, 1] = -pts[:, 2]
        pts_res[:, 2] = pts[:, 1]
    return pts_res


def get_patch_radii(pts_patch: np.array, query_pts: np.array):
    if pts_patch.shape == query_pts.shape:
        patch_radius = np.linalg.norm(pts_patch - query_pts, axis=0)
    else:
        dist = cartesian_dist(np.repeat(np.expand_dims(query_pts, axis=0), pts_patch.shape[0], axis=0),
                              pts_patch, axis=1)
        patch_radius = np.max(dist, axis=0)
    return patch_radius


def model_space_to_patch_space_single_point(
        pts_to_convert_ms: np.array, pts_patch_center_ms: np.array, patch_radius_ms):

    pts_patch_space = pts_to_convert_ms - pts_patch_center_ms
    pts_patch_space = pts_patch_space / patch_radius_ms
    return pts_patch_space


def model_space_to_patch_space(
        pts_to_convert_ms: np.array, pts_patch_center_ms: np.array, patch_radius_ms: float):

    pts_patch_center_ms_repeated = \
        np.repeat(np.expand_dims(pts_patch_center_ms, axis=0), pts_to_convert_ms.shape[-2], axis=-2)
    pts_patch_space = pts_to_convert_ms - pts_patch_center_ms_repeated
    pts_patch_space = pts_patch_space / patch_radius_ms

    return pts_patch_space


def patch_space_to_model_space_single_point(
        pts_to_convert_ps: np.array, pts_patch_center_ms: np.array, patch_radius_ms):

    pts_model_space = pts_to_convert_ps * \
                      np.repeat(np.expand_dims(patch_radius_ms, axis=0), pts_to_convert_ps.shape[0], axis=0)
    pts_model_space = pts_model_space + pts_patch_center_ms
    return pts_model_space


def patch_space_to_model_space(
        pts_to_convert_ps: np.array, pts_patch_center_ms: np.array, patch_radius_ms):

    pts_model_space = pts_to_convert_ps * \
                      np.repeat(np.expand_dims(patch_radius_ms, axis=1), pts_to_convert_ps.shape[1], axis=1)
    pts_model_space = pts_model_space + pts_patch_center_ms
    return pts_model_space


def _get_pts_normals_single_file(pts_file_in, mesh_file_in,
                                 normals_file_out, pts_normals_file_out,
                                 samples_per_model=10000):

    import trimesh.sample
    import sys
    import scipy.spatial as spatial
    from source.base import point_cloud

    # sample points on the surface and take face normal
    pts = np.load(pts_file_in)
    mesh = trimesh.load(mesh_file_in)
    samples, face_ids = trimesh.sample.sample_surface(mesh, samples_per_model)
    mesh.fix_normals()

    # get the normal of the closest sample for each point in the point cloud
    # otherwise KDTree construction may run out of recursions
    leaf_size = 100
    sys.setrecursionlimit(int(max(1000, round(samples.shape[0] / leaf_size))))
    kdtree = spatial.cKDTree(samples, leaf_size)
    pts_dists, sample_ids = kdtree.query(x=pts, k=1)
    face_ids_for_pts = face_ids[sample_ids]
    pts_normals = mesh.face_normals[face_ids_for_pts]

    np.save(normals_file_out, pts_normals)
    point_cloud.write_xyz(pts_normals_file_out, pts, normals=pts_normals)


def get_pts_normals(base_dir, dataset_dir, dir_in_pointcloud,
                    dir_in_meshes, dir_out_normals, samples_per_model=10000, num_processes=1):

    dir_in_pts_abs = os.path.join(base_dir, dataset_dir, dir_in_pointcloud)
    dir_in_meshes_abs = os.path.join(base_dir, dataset_dir, dir_in_meshes)
    dir_out_normals_abs = os.path.join(base_dir, dataset_dir, dir_out_normals)
    dir_out_pts_normals_abs = os.path.join(base_dir, dataset_dir, dir_out_normals, 'pts')

    os.makedirs(dir_out_normals_abs, exist_ok=True)
    os.makedirs(dir_out_pts_normals_abs, exist_ok=True)

    pts_files = [f for f in os.listdir(dir_in_pts_abs)
                 if os.path.isfile(os.path.join(dir_in_pts_abs, f)) and f[-4:] == '.npy']
    files_in_pts_abs = [os.path.join(dir_in_pts_abs, f) for f in pts_files]
    files_in_meshes_abs = [os.path.join(dir_in_meshes_abs, f[:-8] + '.ply') for f in pts_files]
    files_out_normals_abs = [os.path.join(dir_out_normals_abs, f) for f in pts_files]
    files_out_pts_normals_abs = [os.path.join(dir_out_pts_normals_abs, f[:-8] + '.xyz') for f in pts_files]

    calls = []
    for fi, f in enumerate(pts_files):
        # skip if result already exists and is newer than the input
        if file_utils.call_necessary([files_in_pts_abs[fi], files_in_meshes_abs[fi]],
                                     [files_out_normals_abs[fi], files_out_pts_normals_abs[fi]]):
            calls.append((files_in_pts_abs[fi], files_in_meshes_abs[fi],
                          files_out_normals_abs[fi], files_out_pts_normals_abs[fi],
                          samples_per_model))

    utils_mp.start_process_pool(_get_pts_normals_single_file, calls, num_processes)


def _get_dist_from_patch_planes_single_file(file_in_pts_abs, file_in_normals_abs,
                                            file_in_pids_abs, file_in_query_abs,
                                            file_out_dists_abs, num_query_points_per_patch):

    from trimesh.points import point_plane_distance

    pts = np.load(file_in_pts_abs)
    normals = np.load(file_in_normals_abs)
    pids = np.load(file_in_pids_abs)
    query = np.load(file_in_query_abs)

    patch_pts = pts[pids]
    patch_normals = normals[pids]
    patch_center_normal = patch_normals[:, 0]
    patch_centers = np.mean(patch_pts, axis=1)

    dists = np.zeros(query.shape[0])
    for pi in range(pids.shape[0]):
        query_points_id_start = pi * num_query_points_per_patch
        query_points_id_end = (pi + 1) * num_query_points_per_patch
        patch_dists = point_plane_distance(
            points=query[query_points_id_start:query_points_id_end],
            plane_normal=patch_center_normal[pi],
            plane_origin=patch_centers[pi])
        patch_dists[np.isnan(patch_dists)] = 0.0
        dists[query_points_id_start:query_points_id_end] = patch_dists
    np.save(file_out_dists_abs, dists)


def get_point_cloud_sub_sample(sub_sample_size, pts_ms, query_point_ms, uniform=False):
    # take random subsample from point cloud
    if pts_ms.shape[0] >= sub_sample_size:
        # np.random.seed(42)  # test if the random subset causes the irregularities
        def dist_prob():  # probability decreasing with distance from query point
            query_pts = np.broadcast_to(query_point_ms, pts_ms.shape)
            dist = cartesian_dist(query_pts, pts_ms)
            dist_normalized = dist / np.max(dist)
            prob = 1.0 - 1.5 * dist_normalized  # linear falloff
            # prob = 1.0 - 2.0 * np.sin(dist_normalized * np.pi / 2.0)  # faster falloff
            prob_clipped = np.clip(prob, 0.05, 1.0)  # ensure that the probability is (eps..1.0)
            prob_normalized = prob_clipped / np.sum(prob_clipped)
            return prob_normalized

        if uniform:
            # basically choice
            # with replacement for better performance, shouldn't hurt with large point clouds
            sub_sample_ids = np.random.randint(low=0, high=pts_ms.shape[0], size=sub_sample_size)
        else:
            prob = dist_prob()
            sub_sample_ids = np.random.choice(pts_ms.shape[0], size=sub_sample_size, replace=False, p=prob)
        pts_sub_sample_ms = pts_ms[sub_sample_ids, :]
    # if not enough take shuffled point cloud and fill with zeros
    else:
        pts_shuffled = pts_ms[:, :3]
        np.random.shuffle(pts_shuffled)
        zeros_padding = np.zeros((sub_sample_size - pts_ms.shape[0], 3), dtype=np.float32)
        pts_sub_sample_ms = np.concatenate((pts_shuffled, zeros_padding), axis=0)
    return pts_sub_sample_ms
