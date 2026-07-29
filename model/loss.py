import math

import torch
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)


def end_stop_penalty(z, power=1.0, eps=1e-2):
    """[REBUTTAL ONLY] PILA's end-stop penalty, for latents not bounded by construction.

    Added for the rebuttal's prior-vs-architecture study (end-stop loss inside
    HVAE); nothing in the published pipeline calls it. PILA's own penalty stays
    in PhysVAETrainerSMPL._edge_penalty and is untouched.

    Same barrier as PhysVAETrainerSMPL._edge_penalty -- [-log(z) - log(1-z)]^power --
    but the two log terms are continued linearly once they come within `eps` of an
    end-stop. PILA applies the barrier to z_phy = sigmoid(u), which is strictly
    inside (0, 1), so a clamp is enough there. HVAE's z_phy is a Gaussian draw
    around a Softplus mean that is then clamped to [0, 1]: a plain clamp would make
    the barrier constant for exactly the samples sitting on (or past) an end-stop,
    i.e. zero gradient where the penalty is supposed to act. The linear
    continuation keeps pushing those samples back inside.

    Reduces to the PILA form for any z in [eps, 1-eps].
    """
    def barrier(t):
        safe = t.clamp_min(eps)
        return torch.where(t >= eps, -torch.log(safe), -math.log(eps) + (eps - t) / eps)

    return (barrier(z) + barrier(1.0 - z)).pow(power).mean()


def mse_loss(output, target):
    # Sample-wise MSE loss: sum over features, then mean over batch
    return torch.sum((output - target).pow(2), dim=1).mean()


def l1_loss(output, target):
    # Sample-wise L1 loss: sum over features, then mean over batch
    return torch.sum(torch.abs(output - target), dim=1).mean()


def mse_loss_per_channel(output, target):
    with torch.no_grad():
        return torch.mean(torch.square(output - target), dim=0)

def mse_loss_mogi(output, target):
    final_output = output[-1]
    return F.mse_loss(final_output, target)

def mse_loss_mogi_reg(output, target, alpha=1e-2):
    final_output = output[-1]
    mogi_output = output[2]
    mse_loss = F.mse_loss(final_output, target)

    smoothness_reg = smoothness_loss(mogi_output)

    total_loss = mse_loss + alpha * smoothness_reg 
    return total_loss


def smoothness_loss(output):
    # calculate the difference between adjacent elements of a sequence
    # the output has a shape of (batch, sequence, channels)
    diff = output[:, 1:, :] - output[:, :-1, :]
    return torch.mean(torch.square(diff))
