from source import points_to_surf_train


# When you see this error:
# 'Expected more than 1 value per channel when training...' which is raised by the BatchNorm1d layer
# for multi-gpu, use a batch size that can't be divided by the number of GPUs
# for single-gpu, use a straight batch size
# see https://github.com/pytorch/pytorch/issues/2584
# see https://forums.fast.ai/t/understanding-code-error-expected-more-than-1-value-per-channel-when-training/9257/12


def full_train(opt):

    points_to_surf_train.train_meshnet(opt)


if __name__ == '__main__':

    # train model on GT data with multiple query points per patch
    full_train(opt=points_to_surf_train.parse_arguments())
    
    print('MeshNet training is finished!')
