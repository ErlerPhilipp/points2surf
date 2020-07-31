# This messy code computes vertex colors based on the distance reconstruction <-> GT mesh

from source.base import parula_colormap

from source.base import utils_mp

import numpy as np
import trimesh
import trimesh.proximity


def get_normalization_target(distances: list, cut_percentil=0.9):
    dist_concat = np.concatenate(distances, axis=0)
    dist_concat_sorted = np.sort(dist_concat)
    if cut_percentil is not None and cut_percentil < 1.0:
        percentil_id = int(dist_concat_sorted.shape[0] * cut_percentil)
        return dist_concat_sorted[percentil_id]
    else:
        return dist_concat_sorted[-1]


def get_closest_distance_batched(query_pts: np.ndarray, mesh: trimesh.Trimesh, batch_size=1000):

    import multiprocessing
    num_of_cpu = multiprocessing.cpu_count()

    # process batches because trimesh's signed_distance very inefficient on memory
    # 3k queries on a mesh with 27k vertices and 55k faces takes around 8 GB of RAM
    # dists_ms = np.zeros((query_pts.shape[0],))
    pts_ids = np.arange(query_pts.shape[0])
    pts_ids_split = np.array_split(pts_ids, max(1, int(query_pts.shape[0] / batch_size)))
    params = []
    for pts_ids_batch in pts_ids_split:
        # dists_ms[pts_ids_batch] = trimesh.proximity.closest_point(mesh, query_pts[pts_ids_batch])[1]
        params.append((mesh, query_pts[pts_ids_batch]))

    dist_list = utils_mp.start_process_pool(trimesh.proximity.closest_point, params, num_of_cpu)
    dists = np.concatenate([d[1] for d in dist_list])

    print('got distances for {} vertices'.format(query_pts.shape[0]))
    return dists


def visualize_mesh_with_distances(mesh_file: str, mesh: trimesh.Trimesh,
                                  dist_per_vertex: np.ndarray, normalize_to: float, cut_percentil=0.9):

    dist_per_vertex_normalized = dist_per_vertex / normalize_to

    # use parula colormap: dist=0 -> blue, dist=0.5 -> green, dist=1.0 -> yellow
    parulas_indices = (dist_per_vertex_normalized * (parula_colormap.parula_cm.shape[0] - 1)).astype(np.int32)
    dist_greater_than_norm_target = parulas_indices >= parula_colormap.parula_cm.shape[0]
    parulas_indices[dist_greater_than_norm_target] = parula_colormap.parula_cm.shape[0] - 1
    dist_colors_rgb = [parula_colormap.parula_cm[parula_indices] for parula_indices in parulas_indices]

    file_out_vis = mesh_file + '_vis.ply'
    mesh_vis = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=dist_colors_rgb)
    mesh_vis.export(file_out_vis)

    file_out_stats = mesh_file + '_stats.txt'
    with open(file_out_stats, 'w+') as stats_file:
        stats_file.write(
            'Distance from reconstructed mesh vertex to nearest sample on GT mesh, '
            'Min={}, Max={}, Mean={}, normalized to {}, cut percentil {}'.format(
            np.min(dist_per_vertex), np.max(dist_per_vertex), np.mean(dist_per_vertex),
                normalize_to, cut_percentil)
        )


def make_distance_comparison(in_file_rec_meshes: list, in_file_gt_mesh, cut_percentil=0.9, batch_size=1000):

    import trimesh.proximity

    meshes_rec = [trimesh.load(in_file_rec_mesh) for in_file_rec_mesh in in_file_rec_meshes]
    if type(in_file_gt_mesh) == str:
        mesh_gt = trimesh.load(in_file_gt_mesh)
    elif type(in_file_gt_mesh) == list:
        mesh_gt = [trimesh.load(in_file_gt_mesh) for in_file_gt_mesh in in_file_gt_mesh]
    else:
        raise ValueError('Not implemented!')

    # vertices_rec_dists = [trimesh.proximity.closest_point(mesh_gt, mesh_rec.vertices)[1] for mesh_rec in meshes_rec]
    if type(in_file_gt_mesh) == str:
        vertices_rec_dists = [get_closest_distance_batched(mesh_rec.vertices, mesh_gt, batch_size)
                              for mesh_rec in meshes_rec]
    elif type(in_file_gt_mesh) == list:
        vertices_rec_dists = [get_closest_distance_batched(mesh_rec.vertices, mesh_gt[mi], batch_size)
                              for mi, mesh_rec in enumerate(meshes_rec)]
    else:
        raise ValueError('Not implemented!')

    normalize_to = get_normalization_target(vertices_rec_dists, cut_percentil=cut_percentil)

    for fi, f in enumerate(in_file_rec_meshes):
        visualize_mesh_with_distances(
            f, meshes_rec[fi], dist_per_vertex=vertices_rec_dists[fi],
            normalize_to=normalize_to, cut_percentil=cut_percentil)


def main(in_file_rec_meshes: list, in_file_gt_mesh, cut_percentile=0.9, batch_size=1000):

    print('Visualize distances of {} to {}'.format(in_file_rec_meshes, in_file_gt_mesh))
    make_distance_comparison(
        in_file_rec_meshes=in_file_rec_meshes,
        in_file_gt_mesh=in_file_gt_mesh,
        cut_percentil=cut_percentile,
        batch_size=batch_size
    )


if __name__ == "__main__":

    # # holes close-up
    # mesh_name = '00011827_73c6505f827541168d5410e4_trimesh_096.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/holes/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/holes/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/holes/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/holes/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/holes/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/features_close_up/holes/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    # # flat areas close-up
    # mesh_name = '00019114_87f2e2e15b2746ffa4a2fd9a_trimesh_003.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/flats/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/flats/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/flats/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/flats/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/flats/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/features_close_up/flats/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    # # denoising close-up
    # #mesh_name = '00993706_f8bc5c196ab9685d0182bbed_trimesh_001.ply'
    # mesh_name = 'Armadillo.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/denoising/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/denoising/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/denoising/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/denoising/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/features_close_up/denoising/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/features_close_up/denoising/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    # ## denoising (luckily same rotation everywhere)
    # #mesh_name = 'flower.ply'
    # #in_dirs_rec_meshes = [
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/point2surf/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/spsr+pcpnet/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/spsr+gt/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/deepsdf/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # #    #
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/point2surf/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/spsr+pcpnet/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/spsr+gt/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/deepsdf/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # #    #
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/point2surf/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/spsr+pcpnet/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/spsr+gt/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/deepsdf/' + mesh_name,
    # #    '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # #]
    # #in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/gt/' + mesh_name
    # #main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    # # denoising (different rotation unfortunately)
    # mesh_name = '00010429_fc56088abf10474bba06f659_trimesh_004.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/spsr+pcpnet/' + mesh_name,
    #     #
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/spsr+pcpnet/' + mesh_name,
    #     #
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/spsr+pcpnet/' + mesh_name,
    # ]
    # in_dirs_gt_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/extra_noisy/gt/' + mesh_name,
    #     #
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/noisefree/gt/' + mesh_name,
    #     #
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/noise comparison/original/gt/' + mesh_name,
    # ]
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    # # qualitative abc original
    # mesh_name = '00010218_4769314c71814669ba5d3512_trimesh_013.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_original/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_original/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_original/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_original/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_original/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_original/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    # # qualitative abc noisefree
    # mesh_name = '00994034_9299b4c10539bb6b50b162d7_trimesh_000.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_noisefree/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_noisefree/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_noisefree/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_noisefree/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_noisefree/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_noisefree/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    #  qualitative abc extra noisy
    # mesh_name = '00993692_494894597fe7b39310a44a99_trimesh_000.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_extra_noisy/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_extra_noisy/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_extra_noisy/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_extra_noisy/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_extra_noisy/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/qualitative results/abc_extra_noisy/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    # # qualitative custom_dense
    # mesh_name = 'horse.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_dense/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_dense/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_dense/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_dense/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_dense/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_dense/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    # # qualitative custom_extra_noisy
    # mesh_name = 'hand.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_extra_noisy/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_extra_noisy/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_extra_noisy/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_extra_noisy/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_extra_noisy/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_extra_noisy/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9, batch_size=200)

    # # qualitative custom_noisefree
    # mesh_name = 'happy.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_noisefree/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_noisefree/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_noisefree/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_noisefree/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_noisefree/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_noisefree/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    # # qualitative custom_original
    # mesh_name = 'galera.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_original/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_original/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_original/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_original/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_original/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_original/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9)

    # # qualitative custom_sparse
    # mesh_name = 'angel.ply'
    # in_dirs_rec_meshes = [
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_sparse/point2surf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_sparse/spsr+pcpnet/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_sparse/spsr+gt/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_sparse/deepsdf/' + mesh_name,
    #     '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_sparse/atlasnet/' + mesh_name[:-4] + '.xyz.npy.ply',
    # ]
    # in_dirs_gt_meshes = '/home/perler/Nextcloud/point2surf results/figures/qualitative results/custom_sparse/gt/' + mesh_name
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9, batch_size=300)

    # # qualitative thingi10k supplementary
    # in_dirs_rec_meshes = []
    # in_dirs_gt_meshes = []
    # for mesh_name in ['46460.ply', '73133.ply', '77319.ply', '81762.ply', '527631.ply']:
    #     in_dirs_rec_meshes += [
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/point2surf/original/' + mesh_name,
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/deepsdf/original/' + mesh_name,
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/atlasnet/original/' + mesh_name[:-4] + '.xyz.npy.ply',
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/spsr+pcpnet/original/' + mesh_name,
    #     ]
    #     in_dirs_gt_meshes += ['/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/gt/' + mesh_name] * 4
    # for mesh_name in ['75275.ply', '75652.ply', '83229.ply', '86848.ply', '120477.ply']:
    #     in_dirs_rec_meshes += [
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/point2surf/extra_noisy/' + mesh_name,
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/deepsdf/extra_noisy/' + mesh_name,
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/atlasnet/extra_noisy/' + mesh_name[:-4] + '.xyz.npy.ply',
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/spsr+pcpnet/extra_noisy/' + mesh_name,
    #     ]
    #     in_dirs_gt_meshes += ['/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/gt/' + mesh_name] * 4
    # for mesh_name in ['46463.ply', '76277.ply', '85699.ply', '95444.ply', '103354.ply']:
    #     in_dirs_rec_meshes += [
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/point2surf/noisefree/' + mesh_name,
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/deepsdf/noisefree/' + mesh_name,
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/atlasnet/noisefree/' + mesh_name[:-4] + '.xyz.npy.ply',
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/spsr+pcpnet/noisefree/' + mesh_name,
    #     ]
    #     in_dirs_gt_meshes += ['/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/gt/' + mesh_name] * 4
    # for mesh_name in ['46459.ply', '54725.ply', '73998.ply', '91347.ply', '92880.ply']:
    #     in_dirs_rec_meshes += [
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/point2surf/dense/' + mesh_name,
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/deepsdf/dense/' + mesh_name,
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/atlasnet/dense/' + mesh_name[:-4] + '.xyz.npy.ply',
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/spsr+pcpnet/dense/' + mesh_name,
    #     ]
    #     in_dirs_gt_meshes += ['/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/gt/' + mesh_name] * 4
    # for mesh_name in ['46462.ply', '64444.ply', '64764.ply', '68381.ply', '199664.ply']:
    #     in_dirs_rec_meshes += [
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/point2surf/sparse/' + mesh_name,
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/deepsdf/sparse/' + mesh_name,
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/atlasnet/sparse/' + mesh_name[:-4] + '.xyz.npy.ply',
    #         '/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/spsr+pcpnet/sparse/' + mesh_name,
    #     ]
    #     in_dirs_gt_meshes += ['/home/perler/Nextcloud/point2surf results/figures/supp_thingi10k/gt/' + mesh_name] * 4
    # main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9, batch_size=300)

    # qualitative abc supplementary
    in_dirs_rec_meshes = []
    in_dirs_gt_meshes = []
    for mesh_name in ['00014489_f4297f01e3434034b7051ebb_trimesh_004.ply',
                      '00015750_bca56983eee140db9aa4c9a1_trimesh_091.ply',
                      '00991527_88dccf1e5fa948d4fe1757ed_trimesh_009.ply']:
        in_dirs_rec_meshes += [
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/point2surf/original/' + mesh_name,
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/deepsdf/original/' + mesh_name,
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/atlasnet/original/' + mesh_name[:-4] + '.xyz.npy.ply',
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/spsr+pcpnet/original/' + mesh_name,
        ]
        in_dirs_gt_meshes += ['/home/perler/Nextcloud/point2surf results/figures/supp_abc/gt/original/' + mesh_name] * 4
    for mesh_name in ['00012076_bd0ba1071db44a4cb05e612c_trimesh_011.ply',
                      '00017846_08893609d30e453493c4c079_trimesh_021.ply',
                      '00018330_ae93a6d282364256a7bb3358_trimesh_010.ply']:
        in_dirs_rec_meshes += [
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/point2surf/extra_noisy/' + mesh_name,
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/deepsdf/extra_noisy/' + mesh_name,
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/atlasnet/extra_noisy/' + mesh_name[:-4] + '.xyz.npy.ply',
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/spsr+pcpnet/extra_noisy/' + mesh_name,
        ]
        in_dirs_gt_meshes += ['/home/perler/Nextcloud/point2surf results/figures/supp_abc/gt/extra_noisy/' + mesh_name] * 4
    for mesh_name in ['00011000_8a21002f126e4425a811e70a_trimesh_004.ply',
                      '00011602_c087f04c99464bf7ab2380c4_trimesh_000.ply',
                      '00993805_e549aee7e0b31a7501eb8669_trimesh_012.ply']:
        in_dirs_rec_meshes += [
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/point2surf/noisefree/' + mesh_name,
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/deepsdf/noisefree/' + mesh_name,
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/atlasnet/noisefree/' + mesh_name[:-4] + '.xyz.npy.ply',
            '/home/perler/Nextcloud/point2surf results/figures/supp_abc/spsr+pcpnet/noisefree/' + mesh_name,
        ]
        in_dirs_gt_meshes += ['/home/perler/Nextcloud/point2surf results/figures/supp_abc/gt/noisefree/' + mesh_name] * 4
    main(in_dirs_rec_meshes, in_dirs_gt_meshes, cut_percentile=0.9, batch_size=300)
