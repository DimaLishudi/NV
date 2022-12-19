import torch
import torch.nn.functional as F

from collections import defaultdict
from ..utils import MelSpectrogram


def loss(batch, generator, mpd_list, msd_list):
    """
        Args:
            batch - batch of TODO
            generator - generator model
            mdp_list - list of MPD models (raw, x2, x4)
            msd_list - list of MPD models (raw, x2, x4)
    
    """
    # pool targets and predictions

    targets = batch["targets"]
    targets_x2 = F.avg_pool1d(targets, 2, 2)
    targets_x4 = F.avg_pool1d(targets_x2, 2, 2)
    preds = generator(batch["mels"])
    preds_x2   = F.avg_pool1d(preds, 2, 2)
    preds_x4   = F.avg_pool1d(preds_x2, 2, 2)

    all_preds = [preds, preds_x2, preds_x4]
    all_targets = [targets, targets_x2, targets_x4]

    melspec_fn = MelSpectrogram()
    loss_logs = {}

    loss_g_adv = 0
    loss_g_feat = 0
    loss_d = 0

    for i in range(3):
        for disc_list, name in zip([mpd_list, msd_list], ["mdp", "msd"]):
            # adversarial loss
            d_out_target, d_fmaps_target = disc_list[i](all_targets[i])
            d_out_gen, d_fmaps_gen       = disc_list[i](all_preds[i])
            loss_logs[name+f"_x{2**i}_target"] = torch.mean((d_out_target - 1)**2)
            loss_logs[name+f"_x{2**i}_gen"]    = torch.mean(d_out_gen**2)
            loss_d += loss_logs[name+f"_x{2**i}_target"] + loss_logs[name+f"_x{2**i}_gen"]
            loss_g_adv += torch.mean((d_out_gen - 1)**2)

            # feature matching loss
            for fmap_target, fmap_gen in zip(d_fmaps_target, d_fmaps_gen):
                loss_g_feat += F.l1_loss(fmap_target, fmap_gen)

    # melspec loss
    loss_g_mel = F.l1_loss(melspec_fn(preds), batch["mels"])

    # add gen losses to logs
    loss_logs["generator mel"] = loss_g_mel
    loss_logs["generator adv"] = loss_g_adv
    loss_logs["generator feat"] = loss_g_feat


    return loss_g_adv, loss_g_feat, loss_g_mel, loss_d, loss_logs