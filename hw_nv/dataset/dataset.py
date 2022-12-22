import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from time import perf_counter
from tqdm.auto import tqdm

import os

from torch.utils.data import Dataset, DataLoader
from .utils import MelSpectrogram


@torch.no_grad()
def get_data_to_buffer(dataset_config: dict):
    buffer = list()

    start = perf_counter()

    dirs = os.listdir(dataset_config["audio_path"])
    for i, file_name in enumerate(tqdm(dirs)):
        if i > 100:
            break
        audio_name = os.path.join(
            dataset_config["audio_path"], file_name)
        audio, _ = torchaudio.load(audio_name)
        audio = audio.squeeze()
        audio /= torch.abs(audio).max() # normalize audio

        buffer.append(audio)

    end = perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


@torch.no_grad()
def get_segment(audio, mel_fn: MelSpectrogram, segment_size: int):
    if len(audio) < segment_size:
        audio = F.pad(audio, (0, segment_size-len(audio)))

    start = torch.randint(0, len(audio) - segment_size, (1,))
    audio = audio[start:start+segment_size]

    mel = mel_fn(audio, pad=True)

    return {"audio" : audio, "mel" : mel}


class BufferDataset(Dataset):
    def __init__(self, buffer: list, segment_size: int):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)
        self.segment_size = segment_size
        self.mel_fn = MelSpectrogram()

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx: int):
        return get_segment(self.buffer[idx], self.mel_fn, self.segment_size)


def get_LJSpeech_dataloader(dataset_config: dict):
    buffer = get_data_to_buffer(dataset_config)
    dataset = BufferDataset(buffer, dataset_config["segment_size"])
    # no need for collator, as our dataset automatically collates to segment_size
    return DataLoader(
        dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=dataset_config["num_workers"]
    )