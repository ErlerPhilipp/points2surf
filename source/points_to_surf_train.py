from __future__ import print_function

import argparse
import os
import sys
import random
import math
import shutil
import numbers

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

from source.points_to_surf_model import PointsToSurfModel
from source import data_loader
from source import sdf_nn
from source.base import evaluation

debug = False


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='debug',
                        help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.',
                        help='description')
    parser.add_argument('--indir', type=str, default='datasets/abc_minimal',
                        help='input folder (meshes)')
    parser.add_argument('--outdir', type=str, default='models',
                        help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainset.txt',
                        help='training set file name')
    parser.add_argument('--testset', type=str, default='testset.txt',
                        help='test set file name')
    parser.add_argument('--save_interval', type=int, default='10',
                        help='save model each n epochs')
    parser.add_argument('--debug_interval', type=int, default='1',
                        help='print logging info each n epochs')
    parser.add_argument('--refine', type=str, default='',
                        help='refine model at this path')
    parser.add_argument('--gpu_idx', type=int, default=0,
                        help='set < 0 to use CPU')
    parser.add_argument('--patch_radius', type=float, default=0.05,
                        help='Neighborhood of points that is queried with the network. '
                             'This enables you to set the trade-off between computation time and tolerance for '
                             'sparsely sampled surfaces. Use r <= 0.0 for k-NN queries.')

    # training parameters
    parser.add_argument('--net_size', type=int, default=1024,
                        help='number of neurons in the largest fully connected layer')
    parser.add_argument('--nepoch', type=int, default=2,
                        help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=2,
                        help='input batch size')
    parser.add_argument('--patch_center', type=str, default='point',
                        help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0,
                        help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000,
                        help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--sub_sample_size', type=int, default=500,
                        help='number of points of the point cloud that are trained with each patch')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473,
                        help='manual seed')
    parser.add_argument('--single_transformer', type=int, default=0,
                        help='0: two transformers for the global and local information, \n'
                             'rotate local points with matrix trained from global points\n'
                             '1: single transformer for both local and global points')

    parser.add_argument('--uniform_subsample', type=int, default=0,
                        help='1: global sub-sample uniformly sampled from the point cloud\n'
                             '0: distance-depending probability for global sub-sample')
    parser.add_argument('--shared_transformer', type=int, default=0,
                        help='use a single shared QSTN that takes both the local and global point sets as input')
    parser.add_argument('--training_order', type=str, default='random',
                        help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape \n'
                        'remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False,
                        help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--scheduler_steps', type=int, nargs='+', default=[75, 125],
                        help='the lr will be multiplicated with 0.1 at these epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='gradient descent momentum')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean',
                        help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['imp_surf', 'imp_surf_magnitude', 'imp_surf_sign',
                                                                   'patch_pts_ids', 'p_index'],
                        help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'p_index: output debug info to validate the model\n'
                        'imp_surf: distance from query point to patch\n'
                        'imp_surf_magnitude: magnitude for distance from query point to patch\n'
                        'imp_surf_sign: sign for distance from query point to patch\n'
                        'patch_pts_ids: ids for all points in a patch')
    parser.add_argument('--use_point_stn', type=int, default=True,
                        help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True,
                        help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='max',
                        help='symmetry operation')
    parser.add_argument('--points_per_patch', type=int, default=50,
                        help='max. number of points per patch')
    parser.add_argument('--debug', type=int, default=0,
                        help='set to 1 of you want debug outputs to validate the model')

    return parser.parse_args(args=args)


def do_logging(writer, log_prefix, epoch, opt, loss,
               batchind, fraction_done, num_batch, train, output_names, metrics_dict: dict):
    current_step = (epoch + fraction_done) * num_batch * opt.batchSize

    loss_cpu = [l.detach().cpu().item() for l in loss]
    loss_sum = sum(loss_cpu)

    writer.add_scalar('loss/{}/total'.format('train' if train else 'eval'), loss_sum, current_step)
    if len(loss_cpu) > 1:
        for wi, w in enumerate(loss_cpu):
            writer.add_scalar('loss/{}/comp_{}'.format('train' if train else 'eval', output_names[wi]),
                              w, current_step)

    keys_to_log = {'abs_dist_rms', 'accuracy', 'precision', 'recall', 'f1_score'}
    for key in metrics_dict.keys():
        if key in keys_to_log and isinstance(metrics_dict[key], numbers.Number):
            value = metrics_dict[key]
            if math.isnan(value):
                value = 0.0
            writer.add_scalar('metrics/{}/{}'.format('train' if train else 'eval', key), value, current_step)

    if batchind % opt.debug_interval == 0:
        state_string = \
            '[{name} {epoch}: {batch}/{n_batches}] {prefix} loss: {loss:+.2f}, rmse: {rmse:+.2f}, f1: {f1:+.2f}'.format(
                name=opt.name, epoch=epoch, batch=batchind, n_batches=num_batch - 1,
                prefix=log_prefix, loss=loss_sum,
                rmse=metrics_dict['abs_dist_rms'], f1=metrics_dict['f1_score'])
        print(state_string)


def points_to_surf_train(opt):

    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)
    print('Training on {} GPUs'.format(torch.cuda.device_count()))
    print('Training on ' + ('cpu' if opt.gpu_idx < 0 else torch.cuda.get_device_name(opt.gpu_idx)))

    # colored console output, works e.g. on Ubuntu (WSL)
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % opt.name)
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % opt.name)
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % opt.name)

    if os.path.exists(log_dirname) or os.path.exists(model_filename):
        if opt.name != 'test':
            response = input('A training run named "{}" already exists, overwrite? (y/n) '.format(opt.name))
            if response == 'y':
                del_log = True
            else:
                return
        else:
            del_log = True

        if del_log:
            if os.path.exists(log_dirname):
                try:
                    shutil.rmtree(log_dirname)
                except OSError:
                    print("Can't delete " + log_dirname)

    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_names = []
    output_loss_weights = dict()
    pred_dim = 0
    for o in opt.outputs:
        if o == 'imp_surf':
            if o not in target_features:
                target_features.append(o)

            output_names.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            output_loss_weights[o] = 1.0
            pred_dim += 1
        elif o == 'imp_surf_magnitude':
            if o not in target_features:
                target_features.append(o)

            output_names.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            output_loss_weights[o] = 1.0  # try higher weight here
            pred_dim += 1
        elif o == 'imp_surf_sign':
            if o not in target_features:
                target_features.append(o)

            output_names.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            output_loss_weights[o] = 1.0
            pred_dim += 1
        elif o == 'p_index':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
        elif o == 'patch_pts_ids':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
        else:
            raise ValueError('Unknown output: %s' % o)

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')

    # create model
    use_query_point = any([f in opt.outputs for f in ['imp_surf', 'imp_surf_magnitude', 'imp_surf_sign']])
    p2s_model = PointsToSurfModel(
        net_size_max=opt.net_size,
        num_points=opt.points_per_patch,
        output_dim=pred_dim,
        use_point_stn=opt.use_point_stn,
        use_feat_stn=opt.use_feat_stn,
        sym_op=opt.sym_op,
        use_query_point=use_query_point,
        sub_sample_size=opt.sub_sample_size,
        do_augmentation=True,
        single_transformer=opt.single_transformer,
        shared_transformation=opt.shared_transformer,
    )

    start_epoch = 0
    if opt.refine != '':
        print(f'Refining weights from {opt.refine}')
        p2s_model.cuda(device=device)  # same order as in training
        p2s_model = torch.nn.DataParallel(p2s_model)
        p2s_model.load_state_dict(torch.load(opt.refine))
        try:
            # expecting a file name like 'vanilla_model_50.pth'
            model_file = str(opt.refine)
            last_underscore_pos = model_file.rfind('_')
            last_dot_pos = model_file.rfind('.')
            start_epoch = int(model_file[last_underscore_pos+1:last_dot_pos]) + 1
            print(f'Continuing training from epoch {start_epoch}')
        except:
            print(f'Warning: {opt.refine} has no epoch in the name. The Tensorboard log will continue at '
                  f'epoch 0 and might be messed up!')

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # create train and test dataset loaders
    train_dataset = data_loader.PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.trainset,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        center=opt.patch_center,
        cache_capacity=opt.cache_capacity,
        pre_processed_patches=True,
        sub_sample_size=opt.sub_sample_size,
        num_workers=int(opt.workers),
        patch_radius=opt.patch_radius,
        epsilon=-1,  # not necessary for training
        uniform_subsample=opt.uniform_subsample,
    )
    if opt.training_order == 'random':
        train_datasampler = data_loader.RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        train_datasampler = data_loader.SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % opt.training_order)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    test_dataset = data_loader.PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        center=opt.patch_center,
        cache_capacity=opt.cache_capacity,
        pre_processed_patches=True,
        sub_sample_size=opt.sub_sample_size,
        patch_radius=opt.patch_radius,
        num_workers=int(opt.workers),
        epsilon=-1,  # not necessary for training
        uniform_subsample=opt.uniform_subsample,
    )
    if opt.training_order == 'random':
        test_datasampler = data_loader.RandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        test_datasampler = data_loader.SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % opt.training_order)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    print('Training set: {} patches (in {} batches) | Test set: {} patches (in {} batches)'.format(
          len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass

    train_fraction_done = 0.0

    log_writer = SummaryWriter(log_dirname, comment=opt.name)
    log_writer.add_scalar('LR', opt.lr, 0)

    # milestones in number of optimizer iterations
    optimizer = optim.SGD(p2s_model.parameters(), lr=opt.lr, momentum=opt.momentum)

    # SGD changes lr depending on training progress
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)  # constant lr
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.scheduler_steps, gamma=0.1)

    if opt.refine == '':
        p2s_model.cuda(device=device)
        p2s_model = torch.nn.DataParallel(p2s_model)

    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    for epoch in range(start_epoch, opt.nepoch, 1):

        train_enum = enumerate(train_dataloader, 0)

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)

        for train_batchind, batch_data_train in train_enum:

            # batch data to GPU
            for key in batch_data_train.keys():
                batch_data_train[key] = batch_data_train[key].cuda(non_blocking=True)

            # set to training mode
            p2s_model.train()

            # zero gradients
            optimizer.zero_grad()

            pred_train = p2s_model(batch_data_train)

            loss_train = compute_loss(
                pred=pred_train, batch_data=batch_data_train,
                outputs=opt.outputs,
                output_loss_weights=output_loss_weights,
                fixed_radius=opt.patch_radius > 0.0
            )
                
            loss_total = sum(loss_train)

            # back-propagate through entire network to compute gradients of loss w.r.t. parameters
            loss_total.backward()

            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind+1) / train_num_batch

            if debug:
                from source import evaluation
                evaluation.visualize_patch(
                    patch_pts_ps=batch_data_train['patch_pts_ps'][0].cpu(),
                    query_point_ps=batch_data_train['imp_surf_query_point_ps'][0].cpu(),
                    pts_sub_sample_ms=batch_data_train['pts_sub_sample_ms'][0].cpu(),
                    query_point_ms=batch_data_train['imp_surf_query_point_ms'][0].cpu(),
                    file_path='debug/patch_train.off')

            metrics_dict = calc_metrics(outputs=opt.outputs, pred=pred_train, gt_data=batch_data_train)

            do_logging(writer=log_writer, log_prefix=green('train'), epoch=epoch, opt=opt, loss=loss_train,
                       batchind=train_batchind, fraction_done=train_fraction_done, num_batch=train_num_batch,
                       train=True, output_names=output_names, metrics_dict=metrics_dict)
            
            while test_fraction_done <= train_fraction_done and test_batchind + 1 < test_num_batch:

                # set to evaluation mode, no auto-diff
                p2s_model.eval()

                test_batchind, batch_data_test = next(test_enum)

                # batch data to GPU
                for key in batch_data_test.keys():
                    batch_data_test[key] = batch_data_test[key].cuda(non_blocking=True)

                # forward pass
                with torch.no_grad():
                    pred_test = p2s_model(batch_data_test)

                loss_test = compute_loss(
                    pred=pred_test, batch_data=batch_data_test,
                    outputs=opt.outputs,
                    output_loss_weights=output_loss_weights,
                    fixed_radius=opt.patch_radius > 0.0
                )

                metrics_dict = calc_metrics(outputs=opt.outputs, pred=pred_test, gt_data=batch_data_test)

                test_fraction_done = (test_batchind+1) / test_num_batch

                do_logging(writer=log_writer, log_prefix=blue('test'),
                           epoch=epoch, opt=opt, loss=loss_test, batchind=test_batchind,
                           fraction_done=test_fraction_done, num_batch=train_num_batch,
                           train=False, output_names=output_names, metrics_dict=metrics_dict)

        # end of epoch save model, overwriting the old model
        if epoch % opt.save_interval == 0 or epoch == opt.nepoch-1:
            torch.save(p2s_model.state_dict(), model_filename)

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch-1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch-1:
            torch.save(p2s_model.state_dict(), os.path.join(opt.outdir, '%s_model_%d.pth' % (opt.name, epoch)))

        # update and log lr
        lr_before_update = scheduler.get_last_lr()
        if isinstance(lr_before_update, list):
            lr_before_update = lr_before_update[0]
        scheduler.step()
        lr_after_update = scheduler.get_last_lr()
        if isinstance(lr_after_update, list):
            lr_after_update = lr_after_update[0]
        if lr_before_update != lr_after_update:
            print('LR changed from {} to {} in epoch {}'.format(lr_before_update, lr_after_update, epoch))
        current_step = (epoch + 1) * train_num_batch * opt.batchSize - 1
        log_writer.add_scalar('LR', lr_after_update, current_step)

        log_writer.flush()

    log_writer.close()


def compute_loss(pred, batch_data, outputs, output_loss_weights, fixed_radius):

    loss = []

    if 'imp_surf' in outputs:
        o_pred = pred.squeeze()
        o_target = batch_data['imp_surf_ms'].squeeze()
        if not fixed_radius:
            o_patch_radius = batch_data['patch_radius_ms']
            o_target /= o_patch_radius
        loss.append(sdf_nn.calc_loss_distance(pred=o_pred, target=o_target) *
                    output_loss_weights['imp_surf'])
    if 'imp_surf_magnitude' in outputs and 'imp_surf_sign' in outputs:
        o_pred = pred[:, 0].squeeze()
        o_target = batch_data['imp_surf_magnitude_ms'].squeeze()
        if not fixed_radius:
            o_patch_radius = batch_data['patch_radius_ms']
            o_target /= o_patch_radius
        loss.append(sdf_nn.calc_loss_magnitude(pred=o_pred, target=o_target) *
                    output_loss_weights['imp_surf_magnitude'])

        o_pred = pred[:, 1].squeeze()
        o_target = batch_data['imp_surf_dist_sign_ms'].squeeze()
        loss.append(sdf_nn.calc_loss_sign(pred=o_pred, target=o_target) *
                    output_loss_weights['imp_surf_sign'])

    return loss


def calc_metrics(outputs, pred, gt_data):

    def compute_rmse_abs_dist(pred, gt):
        abs_dist = sdf_nn.post_process_magnitude(pred)
        rmse = torch.sqrt(torch.mean((abs_dist.abs() - gt.squeeze().abs()) ** 2))
        return rmse.detach().cpu().item()

    def compare_classification(pred, gt):
        inside_class = sdf_nn.post_process_sign(pred)
        eval_dict = evaluation.compare_predictions_binary_tensors(
            ground_truth=gt.squeeze(), predicted=inside_class, prediction_name='training_metrics')
        return eval_dict

    if 'imp_surf_magnitude' in outputs and 'imp_surf_sign' in outputs:
        abs_dist_rms = compute_rmse_abs_dist(pred=pred[:, 0].squeeze(), gt=gt_data['imp_surf_magnitude_ms'])
        eval_dict = compare_classification(pred=pred[:, 1].squeeze(),
                                           gt=gt_data['imp_surf_dist_sign_ms'])
        eval_dict['abs_dist_rms'] = abs_dist_rms
        return eval_dict
    elif 'imp_surf' in outputs:
        abs_dist_rms = compute_rmse_abs_dist(pred=pred.squeeze(), gt=gt_data['imp_surf_ms'])
        pred_class = pred.squeeze()
        pred_class[pred_class < 0.0] = -1.0
        pred_class[pred_class >= 0.0] = 1.0
        eval_dict = compare_classification(pred=pred_class,
                                           gt=gt_data['imp_surf_dist_sign_ms'])
        eval_dict['abs_dist_rms'] = abs_dist_rms
        return eval_dict
    else:
        return {}


if __name__ == '__main__':
    train_opt = parse_arguments()
    points_to_surf_train(train_opt)
