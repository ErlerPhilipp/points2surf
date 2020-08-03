import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from source.base import utils

input_dims_per_point = 3


class STN(nn.Module):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.net_size_max = net_size_max
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.net_size_max, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(self.net_size_max, int(self.net_size_max / 2))
        self.fc2 = nn.Linear(int(self.net_size_max / 2), int(self.net_size_max / 4))
        self.fc3 = nn.Linear(int(self.net_size_max / 4), self.dim*self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.net_size_max)
        self.bn4 = nn.BatchNorm1d(int(self.net_size_max / 2))
        self.bn5 = nn.BatchNorm1d(int(self.net_size_max / 4))

        if self.num_scales > 1:
            self.fc0 = nn.Linear(self.net_size_max * self.num_scales, self.net_size_max)
            self.bn0 = nn.BatchNorm1d(self.net_size_max)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), self.net_size_max * self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*self.net_size_max:(s+1)*self.net_size_max, :] = \
                    self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, self.net_size_max*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x


class QSTN(nn.Module):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.net_size_max = net_size_max
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.net_size_max, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(self.net_size_max, int(self.net_size_max / 2))
        self.fc2 = nn.Linear(int(self.net_size_max / 2), int(self.net_size_max / 4))
        self.fc3 = nn.Linear(int(self.net_size_max / 4), 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.net_size_max)
        self.bn4 = nn.BatchNorm1d(int(self.net_size_max / 2))
        self.bn5 = nn.BatchNorm1d(int(self.net_size_max / 4))

        if self.num_scales > 1:
            self.fc0 = nn.Linear(self.net_size_max*self.num_scales, self.net_size_max)
            self.bn0 = nn.BatchNorm1d(self.net_size_max)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), self.net_size_max*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*self.net_size_max:(s+1)*self.net_size_max, :] = \
                    self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, self.net_size_max*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x_quat = x + iden

        # convert quaternion to rotation matrix
        x = utils.batch_quat_to_rotmat(x_quat)

        return x, x_quat


class PointNetfeat(nn.Module):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500, use_point_stn=True, use_feat_stn=True,
                 output_size=100, sym_op='max'):
        super(PointNetfeat, self).__init__()

        self.net_size_max = net_size_max
        self.num_points = num_points
        self.num_scales = num_scales
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.output_size = output_size

        if self.use_point_stn:
            self.stn1 = QSTN(net_size_max=net_size_max, num_scales=self.num_scales,
                             num_points=num_points, dim=3, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(net_size_max=net_size_max, num_scales=self.num_scales,
                            num_points=num_points, dim=64, sym_op=self.sym_op)

        self.conv0a = torch.nn.Conv1d(input_dims_per_point, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)
        self.bn0a = nn.BatchNorm1d(64)
        self.bn0b = nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, output_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_size)

        if self.num_scales > 1:
            self.conv4 = torch.nn.Conv1d(output_size, output_size*self.num_scales, 1)
            self.bn4 = nn.BatchNorm1d(output_size*self.num_scales)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.mp1 = None
        else:
            raise ValueError('Unsupported symmetric operation: %s' % self.sym_op)

    def forward(self, x):

        # input transform
        if self.use_point_stn:
            trans, trans_quat = self.stn1(x[:, :3, :])  # transform only point data
            # an error here can mean that your input size is wrong (e.g. added normals in the point cloud files)
            x_transformed = torch.bmm(trans, x[:, :3, :])  # transform only point data
            x = torch.cat((x_transformed, x[:, 3:, :]), dim=1)
        else:
            trans = None
            trans_quat = None

        # mlp (64,64)
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = torch.bmm(trans2, x)
        else:
            trans2 = None

        # mlp (64,128,output_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # mlp (output_size,output_size*num_scales)
        if self.num_scales > 1:
            x = self.bn4(self.conv4(F.relu(x)))

        # symmetric max operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % self.sym_op)

        else:
            x_scales = x.new_empty(x.size(0), self.output_size*self.num_scales**2, 1)
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*self.output_size:(s+1)*self.num_scales*self.output_size, :] = \
                        self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*self.output_size:(s+1)*self.num_scales*self.output_size, :] = \
                        torch.sum(x[:, :, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % self.sym_op)
            x = x_scales

        x = x.view(-1, self.output_size * self.num_scales ** 2)

        return x, trans, trans_quat, trans2


class PointsToSurfModel(nn.Module):  # basing on PointNetDenseCls
    def __init__(self, net_size_max=1024, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True,
                 sym_op='max', use_query_point=False,
                 sub_sample_size=500, do_augmentation=True, single_transformer=False, shared_transformation=False):
        super(PointsToSurfModel, self).__init__()

        self.net_size_max = net_size_max
        self.num_points = num_points
        self.use_query_point = use_query_point
        self.use_point_stn = use_point_stn
        self.sub_sample_size = sub_sample_size
        self.num_query_points = int(self.use_query_point)
        self.do_augmentation = do_augmentation
        self.single_transformer = bool(single_transformer)
        self.shared_transformation = shared_transformation

        if self.single_transformer:
            self.feat_local_global = PointNetfeat(
                net_size_max=net_size_max,
                num_points=self.num_points + self.sub_sample_size,
                num_scales=1,
                use_point_stn=use_point_stn,
                use_feat_stn=use_feat_stn,
                output_size=self.net_size_max,
                sym_op=sym_op)
            self.fc1_local_global = nn.Linear(int(self.net_size_max), int(self.net_size_max))
            self.bn1_local_global = nn.BatchNorm1d(int(self.net_size_max))
        else:
            if self.use_point_stn and self.shared_transformation:
                self.point_stn = QSTN(net_size_max=net_size_max, num_scales=1,
                                      num_points=self.num_points+self.sub_sample_size, dim=3, sym_op=sym_op)

            self.feat_local = PointNetfeat(
                net_size_max=net_size_max,
                num_points=self.num_points,
                num_scales=1,
                use_point_stn=False,
                use_feat_stn=use_feat_stn,
                output_size=self.net_size_max,
                sym_op=sym_op)
            self.feat_global = PointNetfeat(
                net_size_max=net_size_max,
                num_points=self.sub_sample_size,
                num_scales=1,
                use_point_stn=use_point_stn and not self.shared_transformation,
                use_feat_stn=use_feat_stn,
                output_size=self.net_size_max,
                sym_op=sym_op)
            self.fc1_local = nn.Linear(int(self.net_size_max), int(self.net_size_max / 2))
            self.fc1_global = nn.Linear(int(self.net_size_max), int(self.net_size_max / 2))
            self.bn1_local = nn.BatchNorm1d(int(self.net_size_max / 2))
            self.bn1_global = nn.BatchNorm1d(int(self.net_size_max / 2))

        self.fc2 = nn.Linear(int(self.net_size_max / 2) * 2, int(self.net_size_max / 4))
        self.fc3 = nn.Linear(int(self.net_size_max / 4), int(self.net_size_max / 8))
        self.fc4 = nn.Linear(int(self.net_size_max / 8), output_dim)
        self.bn2 = nn.BatchNorm1d(int(self.net_size_max / 4))
        self.bn3 = nn.BatchNorm1d(int(self.net_size_max / 8))

    def forward(self, x):

        patch_features = x['patch_pts_ps'].transpose(1, 2)
        shape_features = x['pts_sub_sample_ms'].transpose(1, 2)
        shape_query_point = x['imp_surf_query_point_ms'].unsqueeze(2)

        # move global points to query point so that both local and global information are centered at the query point
        shape_features -= shape_query_point.expand(shape_features.shape)

        # # debug output for a single patch with its sub-sample
        # if True:
        #     from source import utils_eval
        #     out_file = 'debug/train_sample.ply'
        #     evaluation.visualize_patch(
        #         patch_pts_ps=patch_features[0].detach().cpu().numpy().transpose(),
        #         patch_pts_ms=None,
        #         #query_point_ps=patch_query_point[0].detach().cpu().numpy().transpose().squeeze(),
        #         query_point_ps=torch.zeros(3),
        #         pts_sub_sample_ms=shape_features[0].detach().cpu().numpy().transpose(),
        #         #query_point_ms=shape_query_point[0].detach().cpu().numpy().transpose().squeeze(),
        #         query_point_ms=torch.zeros(3),
        #         file_path=out_file)
        #     print('wrote training sample to {}'.format(out_file))

        if self.single_transformer:
            local_global_features = torch.cat((patch_features,  shape_features), dim=2)
            local_global_features_transformed, _, _, _ = self.feat_local_global(local_global_features)
            patch_features = F.relu(self.bn1_local_global(self.fc1_local_global(local_global_features_transformed)))
        else:
            if self.use_point_stn and self.shared_transformation:
                feats = torch.cat((patch_features,  shape_features), dim=2)
                trans, trans_quat = self.point_stn(feats[:, :3, :])
                shape_features_transformed = torch.bmm(trans, shape_features[:, :3, :])
                patch_features_transformed = torch.bmm(trans, patch_features[:, :3, :])
                shape_features = torch.cat([shape_features_transformed, shape_features[:, 3:, :]], dim=1)
                patch_features = torch.cat([patch_features_transformed, patch_features[:, 3:, :]], dim=1)

            shape_features, trans_global_pts, _, _ = \
                self.feat_global(shape_features)
            shape_features = F.relu(self.bn1_global(self.fc1_global(shape_features)))

            if self.use_point_stn and not self.shared_transformation:
                # rotate patch-space points like the subsample
                patch_features = torch.bmm(trans_global_pts, patch_features)

            patch_features, _, _, _ = \
                self.feat_local(patch_features)
            patch_features = F.relu(self.bn1_local(self.fc1_local(patch_features)))

            # rotate query points like the patch
            patch_features = torch.cat((patch_features,  shape_features), dim=1)

        patch_features = F.relu(self.bn2(self.fc2(patch_features)))
        patch_features = F.relu(self.bn3(self.fc3(patch_features)))
        patch_features = self.fc4(patch_features)

        return patch_features
