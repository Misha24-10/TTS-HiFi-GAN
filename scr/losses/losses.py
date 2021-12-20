import torch
from torch import nn


def feature_matching_loss(fmap_r, fmap_g):
    loss_value = 0
    loss = nn.L1Loss(reduction='mean')
    for i in range(len(fmap_r)):
        for x, y in zip(fmap_r[i], fmap_g[i]):
            loss_value += loss(x, y)
    return loss_value * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    real_losses = []
    gen_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        real_loss = torch.mean((1 - dr) ** 2)
        gen_loss = torch.mean(dg ** 2)
        loss += (real_loss + gen_loss)
        real_losses.append(real_loss.item())
        gen_losses.append(gen_loss.item())

    return loss, real_losses, gen_losses


def generator_loss(disc_outputs):
    loss_value = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss_value += l

    return loss_value, gen_losses
