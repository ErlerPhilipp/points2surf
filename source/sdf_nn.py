import torch
import torch.nn
import torch.nn.functional


def post_process_distance(pred):
    distance = torch.tanh(pred).pow(2) * torch.sign(pred)
    return distance


def post_process_magnitude(pred):
    distance_magnitude = torch.tanh(pred).pow(2)
    return distance_magnitude


def post_process_sign(pred):
    # geodesic distance of disconnected vertices is negative (returned by estimator)
    distance_pos = pred >= 0.0  # logits to bool
    distance_sign = torch.full_like(distance_pos, -1.0, dtype=torch.float32)
    distance_sign[distance_pos] = 1.0  # bool to sign factor
    return distance_sign


def calc_loss_distance(pred, target):
    # tanh or sigmoid to focus the neurons on short distances
    distance_loss = torch.nn.functional.mse_loss(torch.tanh(pred), torch.tanh(target))
    return distance_loss


def calc_loss_magnitude(pred, target):
    # tanh or sigmoid to focus the neurons on short distances
    magnitude_loss = torch.nn.functional.mse_loss(
        torch.tanh(torch.abs(pred)), torch.tanh(torch.abs(target)))
    return magnitude_loss


def calc_loss_sign(pred, target):
    sign_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, target, reduction='none').mean()
    return sign_loss
