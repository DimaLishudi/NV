import numpy as np
import torch
import torchaudio
from torch.nn.utils import remove_weight_norm
import torch.nn.functional as F
from torchvision.utils import make_grid

from .utils import remove_norm
from .. import model
from ..dataset import MelSpectrogram
from ..logger import WanDBWriter
import os



class NVSynthesizer():
    """
        Base Class to synthesize sound from melspectrogram
        This is a wrapper of HiFiGan Generator
    """
    def __init__(self, config, log=False, checkpoint_path=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare generator
        self.generator = model.Generator(config["architecture"])
        if checkpoint_path is not None:
            self.generator.load_state_dict(torch.load(checkpoint_path)["architecture"])
        
        # remove weight norm if it is in place
        try:
            remove_norm(self.generator, remove_weight_norm)
        except Exception: 
            pass
    
        self.generator = self.generator.to(self.device)
        
        # prepare logging
        self.current_step = 0
        self.logger = WanDBWriter(config) if log else None

        # init result dir, prepare inputs
        if "results_dir" in self.config["paths"]:
            self.res_dir = self.config["paths"]["results_dir"]
            os.makedirs(self.res_dir, exist_ok=True)

        if "mels_path" in self.config["paths"]:
            self.mels_path = self.config["paths"]["mels_path"]
        else:
            self.mel_files = []

        self.mel_fn = MelSpectrogram().to(self.device)



    @torch.inference_mode()
    def synthesize(self):
        self.generator.eval()
        mel_loss = 0

        for i, mel_file in enumerate(os.listdir(self.mels_path)):
            mel = torch.from_numpy(np.load(self.mels_path+'/'+mel_file)).to(self.device)
            pred = self.generator(mel)
            mel_pred = self.mel_fn(pred, pad=True).squeeze(1)
            
            # make temp dir to hold results if results directory undefined
            if "results_dir" in self.config["paths"]:
                res_path = self.res_dir + f"/out_{i}.wav"
            else:
                res_path = "./tmp.wav"
            torchaudio.save(res_path, pred.cpu().squeeze(1), self.mel_fn.config.sr)

            # log results
            if self.logger is not None:
                mel_loss += F.l1_loss(mel_pred, mel).item()
                name = f"(input_{i})"
                caption = f"input no. {i}"
                self.logger.add_audio("predicted audio " + name, res_path, caption=caption)
                self.logger.add_spectrogram("source spectrogram " + name, mel, caption=caption)
                self.logger.add_spectrogram("predicted spectrogram " + name, mel_pred, caption=caption)
        if self.logger is not None:    
            self.logger.add_scalar("validation mel loss", mel_loss/len(os.listdir(self.mels_path)))
        
        # clean up
        if "results_dir" not in self.config["paths"]:
            os.remove("./tmp.wav")