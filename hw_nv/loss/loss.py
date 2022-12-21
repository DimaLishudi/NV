import torch
import torch.nn.functional as F

from itertools import chain

from ..utils import MelSpectrogram


def loss(batch, pred, generator, mpd_list, msd_list):
    """
        Args:
            batch - batch of target audio and mels
            generator - generator model
            mdp_list - dict of MPD sub-discriminators
            msd_list - dict of MSD sub-discriminators
    
    """
    melspec_fn = MelSpectrogram()
    loss_logs = {}

    loss_g_adv = 0
    loss_g_feat = 0
    loss_mdp = 0
    loss_msd = 0

    for name, d_list in zip(["MDP", "MSD"], [mpd_list, msd_list]):
        for D in d_list:
        # adversarial loss
            d_out_target, d_fmaps_target = D(batch["target"])
            d_out_gen, d_fmaps_gen       = D(pred)
            loss_logs[name+"_target"] += torch.mean((d_out_target - 1)**2)
            loss_logs[name+"_gen"] += torch.mean(d_out_gen**2)
            loss_g_adv += torch.mean((d_out_gen - 1)**2)

            # feature matching loss
            for fmap_target, fmap_gen in zip(d_fmaps_target, d_fmaps_gen):
                loss_g_feat += F.l1_loss(fmap_target, fmap_gen)

    loss_msd = loss_logs["MSD_target"] + loss_logs["MSD_gen"]
    loss_mdp = loss_logs["MDP_target"] + loss_logs["MDP_gen"]

    # melspec loss
    loss_g_mel = F.l1_loss(melspec_fn(pred), batch["mel"])

    # add gen losses to logs
    loss_logs["generator mel"] = loss_g_mel
    loss_logs["generator adv"] = loss_g_adv
    loss_logs["generator feat"] = loss_g_feat


    return loss_g_adv, loss_g_feat, loss_g_mel, loss_msd, loss_mdp, loss_logs