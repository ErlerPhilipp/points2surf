import numpy as np
import os

from source.base import utils_mp
from source.base import file_utils


def calc_accuracy(num_true, num_predictions):
    if num_predictions == 0:
        return float('NaN')
    else:
        return num_true / num_predictions


def calc_precision(num_true_pos, num_false_pos):
    if isinstance(num_true_pos, (int, float)) and isinstance(num_false_pos, (int, float)) and \
            num_true_pos + num_false_pos == 0:
        return float('NaN')
    else:
        return num_true_pos / (num_true_pos + num_false_pos)


def calc_recall(num_true_pos, num_false_neg):
    if isinstance(num_true_pos, (int, float)) and isinstance(num_false_neg, (int, float)) and \
            num_true_pos + num_false_neg == 0:
        return float('NaN')
    else:
        return num_true_pos / (num_true_pos + num_false_neg)


def calc_f1(precision, recall):
    if isinstance(precision, (int, float)) and isinstance(recall, (int, float)) and \
            precision + recall == 0:
        return float('NaN')
    else:
        return 2.0 * (precision * recall) / (precision + recall)


def compare_predictions_binary_tensors(ground_truth, predicted, prediction_name):
    """

    :param ground_truth:
    :param predicted:
    :param prediction_name:
    :return: res_dict, prec_per_patch
    """

    import torch

    if ground_truth.shape != predicted.shape:
        raise ValueError('The ground truth matrix and the predicted matrix have different sizes!')

    if not isinstance(ground_truth, torch.Tensor) and not isinstance(predicted, torch.Tensor):
        raise ValueError('Both matrices must be dense of type torch.tensor!')

    ground_truth_int = (ground_truth > 0.0).to(dtype=torch.int32)
    predicted_int = (predicted > 0.0).to(dtype=torch.int32)
    res_dict = dict()
    res_dict['comp_name'] = prediction_name

    res_dict["predictions"] = float(torch.numel(ground_truth_int))
    res_dict["pred_gt"] = float(torch.numel(ground_truth_int))
    res_dict["positives"] = float(torch.nonzero(predicted_int).shape[0])
    res_dict["pos_gt"] = float(torch.nonzero(ground_truth_int).shape[0])
    res_dict["true_neg"] = res_dict["predictions"] - float(torch.nonzero(predicted_int + ground_truth_int).shape[0])
    res_dict["negatives"] = res_dict["predictions"] - res_dict["positives"]
    res_dict["neg_gt"] = res_dict["pred_gt"] - res_dict["pos_gt"]
    true_pos = ((predicted_int + ground_truth_int) == 2).sum().to(dtype=torch.float32)
    res_dict["true_pos"] = float(true_pos.sum())
    res_dict["true"] = res_dict["true_pos"] + res_dict["true_neg"]
    false_pos = ((predicted_int * 2 + ground_truth_int) == 2).sum().to(dtype=torch.float32)
    res_dict["false_pos"] = float(false_pos.sum())
    false_neg = ((predicted_int + 2 * ground_truth_int) == 2).sum().to(dtype=torch.float32)
    res_dict["false_neg"] = float(false_neg.sum())
    res_dict["false"] = res_dict["false_pos"] + res_dict["false_neg"]
    res_dict["accuracy"] = calc_accuracy(res_dict["true"], res_dict["predictions"])
    res_dict["precision"] = calc_precision(res_dict["true_pos"], res_dict["false_pos"])
    res_dict["recall"] = calc_recall(res_dict["true_pos"], res_dict["false_neg"])
    res_dict["f1_score"] = calc_f1(res_dict["precision"], res_dict["recall"])

    return res_dict


def eval_predictions(pred_path, gt_path, report_file=None, unsigned=False):
    files = [f for f in os.listdir(pred_path) if os.path.isfile(os.path.join(pred_path, f)) and f[-4:] == '.npy']

    results = []
    for f in files:
        gt_off_path = os.path.join(gt_path, f[:-8] + '.ply.npy')
        rec_off_path = os.path.join(pred_path, f)

        mat_gt = np.load(gt_off_path)
        mat_rec = np.load(rec_off_path)

        if unsigned:
            mat_gt = np.abs(mat_gt)
            mat_rec = np.abs(mat_rec)

        gt_or_pred_nz = ((mat_rec != 0.0) + (mat_gt != 0.0)) > 0
        l2 = (mat_rec - mat_gt)
        l2_sq = l2 * l2
        mse = l2_sq[gt_or_pred_nz].mean()

        mat_gt_mean = mat_gt.mean()
        mat_rec_mean = mat_rec.mean()
        mat_gt_var = (mat_gt * mat_gt).mean() - mat_gt_mean * mat_gt_mean
        mat_rec_var = (mat_rec * mat_rec).mean() - mat_rec_mean * mat_rec_mean

        res_dict = {
            'file': f,
            'mse': mse,
            'mean_gt': mat_gt_mean,
            'mean_pred': mat_rec_mean,
            'var_gt': mat_gt_var,
            'var_pred': mat_rec_var,
        }
        results.append(res_dict)

    print('compare_prediction: {} vs {}\n'.format(gt_path, pred_path))
    lines = print_list_of_dicts(results, ['file', 'mse', 'mean_gt', 'mean_pred', 'var_gt', 'var_pred'], mode='csv')

    if report_file is not None:
        file_utils.make_dir_for_file(report_file)
        with open(report_file, 'w') as the_file:
            for l in lines:
                the_file.write(l + '\n')


def print_list_of_dicts(comp_res, keys_to_print=None, mode='latex'):

    if len(comp_res) == 0:
        return 'WARNING: comp_res is empty'

    if keys_to_print is None or len(keys_to_print) == 0:
        keys_to_print = comp_res[0].keys()

    def get_separator(i, length):
        if mode == 'latex':
            if i < length - 1:
                return ' & '
            else:
                return ' \\\\'
        elif mode == 'csv':
            return ','
            
    # key per line, mesh per column
    #for key in keys_to_print:
    #    line = key + ' && '
    #    for i, d in enumerate(comp_res):
    #        if isinstance(d[key], str):
    #            line += d[key] + get_separator(i, len(keys_to_print))
    #        else:
    #            line += '{0:.3f}'.format(d[key]) + get_separator(i, len(keys_to_print))
    #    print(line)
    
    # mesh per line, key per column
    lines = []
    # contents
    for d in comp_res:
        line = ''
        for i, key in enumerate(keys_to_print):
            if isinstance(d[key], str):
                line += d[key][:10].replace('_', ' ').rjust(max(10, len(key))) + get_separator(i, len(keys_to_print))
            else:
                line += '{0:.5f}'.format(d[key]).rjust(max(10, len(key))) + get_separator(i, len(keys_to_print))
        lines.append(line)
    
    lines.sort()
    
    # header
    line = ''
    for i, key in enumerate(keys_to_print):
        line += key.replace('_', ' ').rjust(10) + get_separator(i, len(keys_to_print))
    lines.insert(0, line)

    for l in lines:
        print(l)

    return lines


def visualize_patch(patch_pts_ps, patch_pts_ms, query_point_ps, pts_sub_sample_ms, query_point_ms,
                    file_path='debug/patch.ply'):

    import source.base.point_cloud as point_cloud

    def filter_padding(patch_pts, query_point):
        query_point_repeated = np.repeat(np.expand_dims(np.array(query_point), axis=0), patch_pts.shape[0], axis=0)
        same_points = patch_pts == query_point_repeated
        non_padding_point_ids = np.sum(same_points, axis=1) != 3
        return patch_pts[non_padding_point_ids]

    patch_pts_ps = filter_padding(patch_pts_ps, query_point_ps)
    if patch_pts_ms is not None:
        patch_pts_ms = filter_padding(patch_pts_ms, query_point_ms)

    query_point_ps = np.expand_dims(query_point_ps, axis=0) \
        if len(query_point_ps.shape) < 2 else query_point_ps
    query_point_ms = np.expand_dims(query_point_ms, axis=0) \
        if len(query_point_ms.shape) < 2 else query_point_ms

    pts = np.concatenate((patch_pts_ps, query_point_ps, pts_sub_sample_ms, query_point_ms), axis=0)
    if patch_pts_ms is not None:
        pts = np.concatenate((pts, patch_pts_ms), axis=0)

    def repeat_color_for_points(color, points):
        return np.repeat(np.expand_dims(np.array(color), axis=0), points.shape[0], axis=0)

    colors_patch_pts_ps = repeat_color_for_points([0.0, 0.0, 1.0], patch_pts_ps)
    colors_query_point_ps = repeat_color_for_points([1.0, 1.0, 0.0], query_point_ps)
    colors_pts_sub_sample_ms = repeat_color_for_points([0.0, 1.0, 0.0], pts_sub_sample_ms)
    colors_query_point_ms = repeat_color_for_points([1.0, 0.0, 1.0], query_point_ms)
    colors = np.concatenate((colors_patch_pts_ps, colors_query_point_ps, colors_pts_sub_sample_ms,
                             colors_query_point_ms), axis=0)
    if patch_pts_ms is not None:
        colors_patch_pts_ms = repeat_color_for_points([1.0, 0.0, 0.0], patch_pts_ms)
        colors = np.concatenate((colors, colors_patch_pts_ms), axis=0)

    point_cloud.write_ply(file_path=file_path, points=pts, colors=colors)


def _chamfer_distance_single_file(file_in, file_ref, samples_per_model, num_processes=1):
    # http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf

    import trimesh
    import trimesh.sample
    import sys
    import scipy.spatial as spatial

    def sample_mesh(mesh_file, num_samples):
        try:
            mesh = trimesh.load(mesh_file)
        except:
            return np.zeros((0, 3))
        samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
        return samples

    new_mesh_samples = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples = sample_mesh(file_ref, samples_per_model)

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, -1.0

    leaf_size = 100
    sys.setrecursionlimit(int(max(1000, round(new_mesh_samples.shape[0] / leaf_size))))
    kdtree_new_mesh_samples = spatial.cKDTree(new_mesh_samples, leaf_size)
    kdtree_ref_mesh_samples = spatial.cKDTree(ref_mesh_samples, leaf_size)

    ref_new_dist, corr_new_ids = kdtree_new_mesh_samples.query(ref_mesh_samples, 1, n_jobs=num_processes)
    new_ref_dist, corr_ref_ids = kdtree_ref_mesh_samples.query(new_mesh_samples, 1, n_jobs=num_processes)

    ref_new_dist_sum = np.sum(ref_new_dist)
    new_ref_dist_sum = np.sum(new_ref_dist)
    chamfer_dist = ref_new_dist_sum + new_ref_dist_sum

    return file_in, file_ref, chamfer_dist


def _hausdorff_distance_directed_single_file(file_in, file_ref, samples_per_model):
    import scipy.spatial as spatial
    import trimesh
    import trimesh.sample

    def sample_mesh(mesh_file, num_samples):
        try:
            mesh = trimesh.load(mesh_file)
        except:
            return np.zeros((0, 3))
        samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
        return samples

    new_mesh_samples = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples = sample_mesh(file_ref, samples_per_model)

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, -1.0

    dist, _, _ = spatial.distance.directed_hausdorff(new_mesh_samples, ref_mesh_samples)
    return file_in, file_ref, dist


def _hausdorff_distance_single_file(file_in, file_ref, samples_per_model):
    import scipy.spatial as spatial
    import trimesh
    import trimesh.sample

    def sample_mesh(mesh_file, num_samples):
        try:
            mesh = trimesh.load(mesh_file)
        except:
            return np.zeros((0, 3))
        samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
        return samples

    new_mesh_samples = sample_mesh(file_in, samples_per_model)
    ref_mesh_samples = sample_mesh(file_ref, samples_per_model)

    if new_mesh_samples.shape[0] == 0 or ref_mesh_samples.shape[0] == 0:
        return file_in, file_ref, -1.0, -1.0, -1.0

    dist_new_ref, _, _ = spatial.distance.directed_hausdorff(new_mesh_samples, ref_mesh_samples)
    dist_ref_new, _, _ = spatial.distance.directed_hausdorff(ref_mesh_samples, new_mesh_samples)
    dist = max(dist_new_ref, dist_ref_new)
    return file_in, file_ref, dist_new_ref, dist_ref_new, dist


def mesh_comparison(new_meshes_dir_abs, ref_meshes_dir_abs,
                    num_processes, report_name, samples_per_model=10000, dataset_file_abs=None):
    if not os.path.isdir(new_meshes_dir_abs):
        print('Warning: dir to check doesn\'t exist'.format(new_meshes_dir_abs))
        return

    new_mesh_files = [f for f in os.listdir(new_meshes_dir_abs)
                      if os.path.isfile(os.path.join(new_meshes_dir_abs, f))]
    ref_mesh_files = [f for f in os.listdir(ref_meshes_dir_abs)
                      if os.path.isfile(os.path.join(ref_meshes_dir_abs, f))]

    if dataset_file_abs is None:
        mesh_files_to_compare_set = set(ref_mesh_files)  # set for efficient search
    else:
        if not os.path.isfile(dataset_file_abs):
            raise ValueError('File does not exist: {}'.format(dataset_file_abs))
        with open(dataset_file_abs) as f:
            mesh_files_to_compare_set = f.readlines()
            mesh_files_to_compare_set = [f.replace('\n', '') + '.ply' for f in mesh_files_to_compare_set]
            mesh_files_to_compare_set = [f.split('.')[0] for f in mesh_files_to_compare_set]
            mesh_files_to_compare_set = set(mesh_files_to_compare_set)

    # # skip if everything is unchanged
    # new_mesh_files_abs = [os.path.join(new_meshes_dir_abs, f) for f in new_mesh_files]
    # ref_mesh_files_abs = [os.path.join(ref_meshes_dir_abs, f) for f in ref_mesh_files]
    # if not utils_files.call_necessary(new_mesh_files_abs + ref_mesh_files_abs, report_name):
    #     return

    def ref_mesh_for_new_mesh(new_mesh_file: str, all_ref_meshes: list) -> list:
        stem_new_mesh_file = new_mesh_file.split('.')[0]
        ref_files = list(set([f for f in all_ref_meshes if f.split('.')[0] == stem_new_mesh_file]))
        return ref_files

    call_params = []
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file.split('.')[0] in mesh_files_to_compare_set:
            new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
            ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
            if len(ref_mesh_files_matching) > 0:
                ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model))
    if len(call_params) == 0:
        raise ValueError('Results are empty!')
    results_hausdorff = utils_mp.start_process_pool(_hausdorff_distance_single_file, call_params, num_processes)
    results = [(r[0], r[1], str(r[2]), str(r[3]), str(r[4])) for r in results_hausdorff]

    call_params = []
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file.split('.')[0] in mesh_files_to_compare_set:
            new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
            ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
            if len(ref_mesh_files_matching) > 0:
                ref_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                call_params.append((new_mesh_file_abs, ref_mesh_file_abs, samples_per_model))
    results_chamfer = utils_mp.start_process_pool(_chamfer_distance_single_file, call_params, num_processes)
    results = [r + (str(results_chamfer[ri][2]),) for ri, r in enumerate(results)]

    # no reference but reconstruction
    for fi, new_mesh_file in enumerate(new_mesh_files):
        if new_mesh_file.split('.')[0] not in mesh_files_to_compare_set:
            if dataset_file_abs is None:
                new_mesh_file_abs = os.path.join(new_meshes_dir_abs, new_mesh_file)
                ref_mesh_files_matching = ref_mesh_for_new_mesh(new_mesh_file, ref_mesh_files)
                if len(ref_mesh_files_matching) > 0:
                    reference_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_mesh_files_matching[0])
                    results.append((new_mesh_file_abs, reference_mesh_file_abs, str(-2), str(-2), str(-2), str(-2)))
        else:
            mesh_files_to_compare_set.remove(new_mesh_file.split('.')[0])

    # no reconstruction but reference
    for ref_without_new_mesh in mesh_files_to_compare_set:
        new_mesh_file_abs = os.path.join(new_meshes_dir_abs, ref_without_new_mesh)
        reference_mesh_file_abs = os.path.join(ref_meshes_dir_abs, ref_without_new_mesh)
        results.append((new_mesh_file_abs, reference_mesh_file_abs, str(-1), str(-1), str(-1), str(-1)))

    # sort by file name
    results = sorted(results, key=lambda x: x[0])

    file_utils.make_dir_for_file(report_name)
    csv_lines = ['in mesh,ref mesh,Hausdorff dist new-ref,Hausdorff dist ref-new,Hausdorff dist,'
                 'Chamfer dist(-1: no input; -2: no reference)']
    csv_lines += [','.join(item) for item in results]
    #csv_lines += ['=AVERAGE(E2:E41)']
    csv_lines_str = '\n'.join(csv_lines)
    with open(report_name, "w") as text_file:
        text_file.write(csv_lines_str)
