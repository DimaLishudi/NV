from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

import librosa  


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251

    # added center option
    center: bool = False


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig=None):
        super(MelSpectrogram, self).__init__()
        if config is None:
            config = MelSpectrogramConfig()
        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            center=config.center
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    # added center for correct audio processing by segments
    def forward(self, audio: torch.Tensor, pad=False) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        if pad:
            win = self.config.win_length
            hop = self.config.hop_length
            pad_len = (win - hop) // 2  # padding to make len(mels) * hop_size = len(audio)
            padded_audio = F.pad(audio, (pad_len, pad_len))
        else:
            padded_audio = audio

        mel = self.mel_spectrogram(padded_audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel