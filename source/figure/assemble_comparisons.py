import os
from enum import Enum
import numpy as np
import csv
import string

from source.base import file_utils


class Fields(Enum):
    in_mesh = 0
    ref_mesh = 1
    hausdorff_in_ref = 2
    hausdorff_ref_in = 3
    hausdorff_sym = 4
    chamfer = 5


def get_column_mean_formula(col, row_start, row_end):
    # TODO: works only until column 26 -> ... div 260, div 26
    # e.g. =AVERAGE(F2:F56)
    formula_template = '=ROUND(AVERAGE({col_start}{row_start}:{col_end}{row_end}),2)'
    formula = formula_template.format(
        col_start=string.ascii_uppercase[col],
        row_start=row_start,
        col_end=string.ascii_uppercase[col],
        row_end=row_end,
    )
    return formula


def arg_sort(seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    # by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__)


if __name__ == '__main__':
    result_files = {
        # 'poisson_gt':                   'D:\\datasets\\own\\implicit_surf_6\\comp_poisson_rec_gt_normals.csv',
        # 'poisson_meshlab':              'D:\\datasets\\own\\implicit_surf_6\\comp_poisson_rec.csv',
        # 'meshnet_gt_100':               'D:\\datasets\\own\\implicit_surf_6\\comp_mc_gt_rec.csv',
        # 'meshnet_single_output_prop_neg_borders_256':    'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_6_bigger_backup\\mc\\hausdorff_dist_pred_rec.csv',
        # 'meshnet_split_outputs_sign_prop_conf_6_256':    'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_6_two_outputs\\implicit_surf_6_sign_prop_conf_6\\rec\\hausdorff_dist_pred_rec.csv',
        # 'meshnet_split_outputs_sign_prop_neg_borders_256':    'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_6_two_outputs\\implicit_surf_6_sign_prop_neg_borders\\rec\\hausdorff_dist_pred_rec.csv',
        # 'meshnet_split_outputs_better_subsampling':    'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_6_two_outputs_better_subsampling\\implicit_surf_6\\rec\\hausdorff_dist_pred_rec.csv',

        # older dataset
        #'poisson_gt': 'D:\\datasets\\own\\implicit_surf_7\\comp_poisson_rec_gt_normals.csv',
        #'poisson_meshlab': 'D:\\datasets\\own\\implicit_surf_7\\comp_poisson_rec.csv',
        #'meshnet_gt_100': 'D:\\datasets\\own\\implicit_surf_7\\comp_mc_gt_rec.csv',
        #'meshnet_rand_rot_more_complex_7_conf6':    'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_7_rand_rot_more_complex\\implicit_surf_7_conf6\\rec\\hausdorff_dist_pred_rec.csv',
        #'meshnet_rand_rot_more_complex_7_conf26':    'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_7_rand_rot_more_complex\\implicit_surf_7_conf26\\rec\\hausdorff_dist_pred_rec.csv',

        # # best, new dataset with rotation as data augmentation
        # 'poisson_gt': 'D:\\datasets\\own\\implicit_surf_7_1\\comp_poisson_rec_gt_normals.csv',
        # 'poisson_meshlab': 'D:\\datasets\\own\\implicit_surf_7_1\\comp_poisson_rec.csv',
        # #'meshnet_gt_100': 'D:\\datasets\\own\\implicit_surf_7_1\\comp_mc_gt_rec.csv',
        # 'meshnet_rot_augm_conf26':    'D:\\meshnet data\\results\\implicit_surf_7_3_rot_augm\\implicit_surf_7_1\\rec\\hausdorff_dist_pred_rec.csv',

            #   # classic meshes variants
        'poisson_gt_classic':                   'D:\\datasets\\own\\implicit_surf_6_classic\\comp_poisson_rec_gt_normals.csv',
        #'poisson_meshlab_classic':              'D:\\datasets\\own\\implicit_surf_6_classic\\comp_poisson_rec_ml_normals.csv',
        #'poisson_cgal_classic':              'D:\\datasets\\own\\implicit_surf_6_classic\\comp_poisson_rec_cgal_normals.csv',
        'poisson_pcpnet_classic':              'D:\\datasets\\own\\implicit_surf_6_classic\\comp_poisson_rec_pcpnet_normals.csv',
        'meshnet_classic':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_10_3_dist_subsample\\implicit_surf_6_classic\\rec\\hausdorff_dist_pred_rec.csv',
        #'meshnet_trans_classic':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_9_4_more_steps\\implicit_surf_6_classic\\rec\\hausdorff_dist_pred_rec.csv',
        #'meshnet_rot_disc_classic':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_9_3_even_bigger\\implicit_surf_6_classic\\rec\\hausdorff_dist_pred_rec.csv',
        'poisson_gt_classic_dense':                   'D:\\datasets\\own\\implicit_surf_6_classic_dense\\comp_poisson_rec_gt_normals.csv',
        #'poisson_meshlab_classic_dense':              'D:\\datasets\\own\\implicit_surf_6_classic_dense\\comp_poisson_rec_ml_normals.csv',
        #'poisson_cgal_classic_dense':              'D:\\datasets\\own\\implicit_surf_6_classic_dense\\comp_poisson_rec_cgal_normals.csv',
        'poisson_pcpnet_classic_dense':              'D:\\datasets\\own\\implicit_surf_6_classic_dense\\comp_poisson_rec_pcpnet_normals.csv',
        'meshnet_classic_dense':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_10_3_dist_subsample\\implicit_surf_6_classic_dense\\rec\\hausdorff_dist_pred_rec.csv',
        #'meshnet_trans_classic_dense':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_9_4_more_steps\\implicit_surf_6_classic_dense\\rec\\hausdorff_dist_pred_rec.csv',
        #'meshnet_rot_disc_classic_dense':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_9_3_even_bigger\\implicit_surf_6_classic_dense_up_3_sigma_5_cert_13\\rec\\hausdorff_dist_pred_rec.csv',
        #'meshnet_tuned_classic_dense':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_9_3_even_bigger\\implicit_surf_6_classic_dense_up_2_sigma_3_cert_5\\rec\\hausdorff_dist_pred_rec.csv',
        'poisson_gt_classic_noisefree':                   'D:\\datasets\\own\\implicit_surf_6_classic_noisefree\\comp_poisson_rec_gt_normals.csv',
        #'poisson_meshlab_classic_noisefree':              'D:\\datasets\\own\\implicit_surf_6_classic_noisefree\\comp_poisson_rec_ml_normals.csv',
        #'poisson_cgal_classic_noisefree':              'D:\\datasets\\own\\implicit_surf_6_classic_noisefree\\comp_poisson_rec_cgal_normals.csv',
        'poisson_pcpnet_classic_noisefree':              'D:\\datasets\\own\\implicit_surf_6_classic_noisefree\\comp_poisson_rec_pcpnet_normals.csv',
        'meshnet_classic_noisefree':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_10_3_dist_subsample\\implicit_surf_6_classic_noisefree\\rec\\hausdorff_dist_pred_rec.csv',
        #'meshnet_trans_classic_noisefree':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_9_4_more_steps\\implicit_surf_6_classic_noisefree\\rec\\hausdorff_dist_pred_rec.csv',
        #'meshnet_rot_disc_classic_noisefree':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_9_3_even_bigger\\implicit_surf_6_classic_noisefree\\rec\\hausdorff_dist_pred_rec.csv',
        'poisson_gt_classic_noisy':                   'D:\\datasets\\own\\implicit_surf_6_classic_noisy\\comp_poisson_rec_gt_normals.csv',
        #'poisson_meshlab_classic_noisy':              'D:\\datasets\\own\\implicit_surf_6_classic_noisy\\comp_poisson_rec_ml_normals.csv',
        #'poisson_cgal_classic_noisy':              'D:\\datasets\\own\\implicit_surf_6_classic_noisy\\comp_poisson_rec_cgal_normals.csv',
        'poisson_pcpnet_classic_noisy':              'D:\\datasets\\own\\implicit_surf_6_classic_noisy\\comp_poisson_rec_pcpnet_normals.csv',
        'meshnet_classic_noisy':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_10_3_dist_subsample\\implicit_surf_6_classic_noisy\\rec\\hausdorff_dist_pred_rec.csv',
        #'meshnet_trans_classic_noisy':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_9_4_more_steps\\implicit_surf_6_classic_noisy\\rec\\hausdorff_dist_pred_rec.csv',
        #'meshnet_rot_disc_classic_noisy':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_9_3_even_bigger\\implicit_surf_6_classic_noisy\\rec\\hausdorff_dist_pred_rec.csv',
        'poisson_gt_classic_extra_noisy':                   'D:\\datasets\\own\\implicit_surf_6_classic_extra_noisy\\comp_poisson_rec_gt_normals.csv',
        #'poisson_meshlab_classic_extra_noisy':              'D:\\datasets\\own\\implicit_surf_6_classic_extra_noisy\\comp_poisson_rec_ml_normals.csv',
        #'poisson_cgal_classic_extra_noisy':              'D:\\datasets\\own\\implicit_surf_6_classic_extra_noisy\\comp_poisson_rec_cgal_normals.csv',
        #'poisson_pcpnet_classic_extra_noisy':              'D:\\datasets\\own\\implicit_surf_6_classic_extra_noisy\\comp_poisson_rec_pcpnet_normals.csv',
        #'meshnet_classic_extra_noisy':               'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_7_3_rot_augm\\implicit_surf_6_classic_extra_noisy\\rec\\hausdorff_dist_pred_rec.csv',

        ## confidence threshold
        #'conf_0':       'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_7_3_rot_augm\\implicit_surf_7_1\\rec\\hausdorff_dist_pred_rec_conf_0.csv',
        #'conf_6':       'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_7_3_rot_augm\\implicit_surf_7_1\\rec\\hausdorff_dist_pred_rec_conf_6.csv',
        #'conf_26':      'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_7_3_rot_augm\\implicit_surf_7_1\\rec\\hausdorff_dist_pred_rec_conf_26.csv',
        #'conf_51':      'C:\\Users\\perler\\Documents\\repos\\meshnet\\results\\implicit_surf_7_3_rot_augm\\implicit_surf_7_1\\rec\\hausdorff_dist_pred_rec_conf_51.csv',
        ##'meshnet_gt_100': 'D:\\datasets\\own\\implicit_surf_7_1\\comp_mc_gt_rec.csv',
    }
    assembled_file = 'C:\\Users\\perler\\Desktop\\comp_res.tsv'

    comp_files = None
    comp_hausdorff = []
    comp_chamfer = []
    for fi, file in enumerate(result_files):
        with open(result_files[file], newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='\"', )
            file_data = np.array([val for val in csv_reader])
            file_data = file_data[1:]  # skip header
            comp_file = file_data[:, Fields.ref_mesh.value]
            comp_file = np.array([os.path.basename(f).lower() for f in comp_file])
            comp_file_sort_ids = arg_sort(comp_file)
            comp_file_sorted = comp_file[comp_file_sort_ids]
            if comp_files is None:
                comp_files = comp_file_sorted
            elif len(comp_files) != len(comp_file_sorted):
                print('Files missing: {}'.format(set(comp_files).difference(set(comp_file_sorted))))
            elif (comp_files != comp_file_sorted).any():
                print('Compared files are different: {}'.format(comp_file_sorted[comp_files[0] != comp_file_sorted]))

            comp_hausdorff.append(file_data[:, Fields.hausdorff_sym.value][comp_file_sort_ids])
            comp_chamfer.append(file_data[:, Fields.chamfer.value][comp_file_sort_ids])

    def transform_comp_data(comp_data):
        transformed_comp_data = np.array(comp_data).transpose().tolist()
        transformed_comp_data = [["{:.2f}".format(float(x)) if x != '-1' and x != '-2' else '' for x in file_comp]
                                 for file_comp in transformed_comp_data]
        transformed_comp_data = [[comp_file_sorted[fi]] + file_comp for fi, file_comp in enumerate(transformed_comp_data)]
        return transformed_comp_data

    def make_mean_formulae(col_start, row_start):
        mean_formulae = [get_column_mean_formula(col + col_start, row_start, row_start + len(comp_files) - 1)
                         for col in range(len(result_files.keys()))]
        mean_formulae = ['mean'] + mean_formulae
        return mean_formulae

    header_fields = ['file'] + list(result_files.keys())
    comp_hausdorff = transform_comp_data(comp_hausdorff)
    comp_chamfer = transform_comp_data(comp_chamfer)
    hausdorff_mean = make_mean_formulae(col_start=1, row_start=3)
    chamfer_mean = make_mean_formulae(col_start=1, row_start=8 + len(comp_files))

    saparator = '\t'

    file_utils.make_dir_for_file(assembled_file)
    csv_lines = ['Symmetric Hausdorff Distance']
    csv_lines += [saparator.join(header_fields)]
    csv_lines += [saparator.join(item) for item in comp_hausdorff]
    csv_lines += [saparator.join(hausdorff_mean)]

    csv_lines += ['', '']
    csv_lines += ['Chamfer Distance']
    csv_lines += [saparator.join(header_fields)]
    csv_lines += [saparator.join(item) for item in comp_chamfer]
    csv_lines += [saparator.join(chamfer_mean)]

    csv_lines_str = '\n'.join(csv_lines)
    with open(assembled_file, "w") as text_file:
        text_file.write(csv_lines_str)
