import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import weight_norm, spectral_norm

from tqdm.auto import trange, tqdm
import os
from itertools import chain

from .synthesizer import NVSynthesizer
from .utils import apply_norm
from .. import loss
from .. import model
from ..dataset import get_LJSpeech_dataloader, MelSpectrogramConfig


class NVTrainer(NVSynthesizer):
    """
        Base Class to synthesize sound from melspectrogram and train HiFiGan
        This is a wrapper of HiFiGan Generator and Dicriminators
    """
    def __init__(self, config, checkpoint_path=None):
        super().__init__(config, log=True, checkpoint_path=checkpoint_path)

        mel_config = MelSpectrogramConfig()
        
        # init discriminator models
        self.msd_list = nn.ModuleList([
            model.MSD(log_p, config["architecture"]) for log_p in config["architecture"]["audio_pools"]
        ])
        self.mpd_list = nn.ModuleList([
            model.MPD(p, config["architecture"], mel_config.pad_value) for p in config["architecture"]["periods"]
        ])

        # init weight_normalization/spectral_normalization and move to device
        self.generator = apply_norm(self.generator, weight_norm).to(self.device)
        self.mpd_list = apply_norm(self.mpd_list, weight_norm).to(self.device)
        for i in range(len(self.msd_list)):
            if i == 0:
                self.msd_list[i] = apply_norm(self.msd_list[i], spectral_norm)
            else:
                self.msd_list[i] = apply_norm(self.msd_list[i], weight_norm)
        self.msd_list = self.msd_list.to(self.device)

        # init dataloader
        self.dataloader = get_LJSpeech_dataloader(config["data"])

        # init optimizers and schedulers
        self.g_optimizer = torch.optim.AdamW(self.generator.parameters(), **config["optimizer"])
        self.d_optimizer = torch.optim.AdamW(chain(self.msd_list.parameters(), self.mpd_list.parameters()), **config["optimizer"])


        self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, **config["scheduler"])
        self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer, **config["scheduler"])

        # load checkpoint
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.g_optimizer.load_state_dict(checkpoint["gen_optimizer"])
            self.g_scheduler.load_state_dict(checkpoint["gen_scheduler"])
            self.d_optimizer.load_state_dict(checkpoint["disc_optimizer"])
            self.d_scheduler.load_state_dict(checkpoint["disc_scheduler"])
            self.msd_list.load_state_dict(checkpoint["MSD"])
            self.mpd_list.load_state_dict(checkpoint["MPD"])

        # init checkpoint save dir
        os.makedirs(self.config["paths"]["save_dir"], exist_ok=True)


    def train_loop(self):
        epochs = self.config["trainer"]["epochs"]
        device = self.device

        for epoch in trange(epochs):
            self.logger.add_scalar("epoch", epoch+1)
            self.logger.add_scalar(
                "learning rate", self.g_scheduler.get_last_lr()[0]
            )
            for batch in tqdm(self.dataloader):
                self.current_step += 1
                
                self.logger.set_step(self.current_step)

                # Get Data
                batch["mel"] = batch["mel"].to(device)
                batch["audio"] = batch["audio"].to(device)

                # Forward
                pred = self.generator(batch["mel"]).squeeze()
                
                # Discriminator Backward:
                loss_msd, loss_mdp, d_loss_logs = loss.discriminator_loss(batch, pred, self.mpd_list, self.msd_list)
                self.d_optimizer.zero_grad()
                loss_msd.backward()
                loss_mdp.backward()
                self.d_optimizer.step()

                # Generator Backward:
                g_loss, g_loss_logs = loss.generator_loss(batch, pred, self.mpd_list, self.msd_list)
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Log losses
                for name, loss_val in chain(d_loss_logs.items(), g_loss_logs.items()):
                    self.logger.add_scalar(name, loss_val.item())

            self.scheduler.step()

            if (epoch+1) % self.config["trainer"]["save_epoch"] == 0:
                torch.save(
                    {
                        "generator": self.generator.state_dict(),
                        "gen_optimizer": self.g_optimizer.state_dict(),
                        "gen_scheduler": self.d_scheduler.state_dict(),
                        "MSD" : self.msd_list.state_dict(),
                        "MPD" : self.mpd_list.state_dict(),
                        "disc_optimizer": self.d_optimizer.state_dict(),
                        "disc_scheduler": self.d_scheduler.state_dict(),
                    },
                    os.path.join(self.config["trainer"]["save_dir"], "checkpoint_%d.pth.tar" % self.current_step))
                print("save model at step %d ..." % self.current_step)

            if (epoch+1) % self.config["trainer"]["validation_epoch"] == 0:
                self.generator.eval()
                self.synthesize()
                self.generator.train()

        
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "gen_optimizer": self.g_optimizer.state_dict(),
                "gen_scheduler": self.d_scheduler.state_dict(),
                "MSD" : self.msd_list.state_dict(),
                "MPD" : self.mpd_list.state_dict(),
                "disc_optimizer": self.d_optimizer.state_dict(),
                "disc_scheduler": self.d_scheduler.state_dict(),
            },
            os.path.join(self.config["trainer"]["save_dir"], "checkpoint_final.pth.tar"))
        print("save final model")