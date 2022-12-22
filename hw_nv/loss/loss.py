import torch
import torch.nn.functional as F

from ..dataset import MelSpectrogram

    
def discriminator_loss(batch, pred, mpd_list, msd_list):
    """
        Args:
            batch - batch of target audio and mels
            generator - generator model
            mdp_list - nn.ModuleList of MPD sub-discriminators
            msd_list - nn.ModuleList of MSD sub-discriminators
    
    """
    loss_logs = {}

    loss_mdp = 0
    loss_msd = 0

    for name, d_list in zip(["MPD", "MSD"], [mpd_list, msd_list]):
        for D in d_list:
            # adversarial loss
            d_out_target, _ = D(batch["audio"])
            d_out_gen, _ = D(pred.detach())
            if name+" target loss" not in loss_logs:
                loss_logs[name+" target loss"] = torch.mean((d_out_target - 1)**2)
                loss_logs[name+" gen loss"] = torch.mean(d_out_gen**2)
            else:
                loss_logs[name+" target loss"] += torch.mean((d_out_target - 1)**2)
                loss_logs[name+" gen loss"] += torch.mean(d_out_gen**2)

    loss_msd = loss_logs["MSD target loss"] + loss_logs["MSD gen loss"]
    loss_mdp = loss_logs["MPD target loss"] + loss_logs["MPD gen loss"]

    loss_logs["MSD total loss"] = loss_msd
    loss_logs["MPD total loss"] = loss_mdp

    return loss_msd, loss_mdp, loss_logs


def generator_loss(batch, pred, mpd_list, msd_list):
    """
        Args:
            batch - batch of target audio and mels
            pred - generator audio prediction
            mdp_list - nn.ModuleList of MPD sub-discriminators
            msd_list - nn.ModuleList of MSD sub-discriminators
    
    """
    mel_fn = MelSpectrogram().to(batch["mel"].device)
    loss_logs = {}

    loss_g_adv = 0
    loss_g_feat = 0

    for name, d_list in zip(["MPD", "MSD"], [mpd_list, msd_list]):
        for D in d_list:
            # adversarial loss
            _, d_fmaps_target = D(batch["audio"])
            d_out_gen, d_fmaps_gen = D(pred)
            loss_g_adv += torch.mean((d_out_gen - 1)**2)

            # feature matching loss
            for fmap_target, fmap_gen in zip(d_fmaps_target, d_fmaps_gen):
                loss_g_feat += F.l1_loss(fmap_target, fmap_gen)

    # melspec loss
    loss_g_mel = 45*F.l1_loss(mel_fn(pred, pad=True), batch["mel"])
    loss_g_feat = 2*loss_g_feat
    loss_g = loss_g_adv + loss_g_feat + loss_g_mel

    # add gen losses to logs
    loss_logs["generator mel loss"] = loss_g_mel
    loss_logs["generator adv loss"] = loss_g_adv
    loss_logs["generator feat loss"] = loss_g_feat
    loss_logs["generator total loss"] = loss_g

    return loss_g, loss_logs