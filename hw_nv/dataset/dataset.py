import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from time import perf_counter
from tqdm.auto import tqdm

import os

from torch.utils.data import Dataset, DataLoader


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def get_data_to_buffer(dataset_config):
    buffer = list()

    start = perf_counter()
    for i in tqdm(range(dataset_config["datasize"])):

        mel_name = os.path.join(
            dataset_config['mels'], "ljspeech-mel-%05d.npy" % (i))
        mel = np.load(mel_name)

        audio_name = os.path.join( # TODO: Correct name
            dataset_config['audio'], "ljspeech-mel-%05d.npy" % (i))
        audio, _ = torchaudio.load(audio_name).mean(dim=0)

        mels = torch.from_numpy(mels)

        buffer.append({"mel": mel,
                       "target" : audio 
        })

    end = perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


def reprocess_tensor(batch, cut_list, melspec_config):
    mels = [batch[ind]["mel"] for ind in cut_list]
    audios = [batch[ind]["audio"] for ind in cut_list]

    # length_mel = np.array(list())
    # for mel in mels:
    #     length_mel = np.append(length_mel, mel.size(0))

    # mel_pos = list()
    # max_mel_len = int(max(length_mel))
    # for length_mel_row in length_mel:
    #     mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
    #                           (0, max_mel_len-int(length_mel_row)), 'constant'))
    # mel_pos = torch.from_numpy(np.array(mel_pos))

    audios = pad_1D_tensor(audios)
    mels = pad_2D_tensor(mels, pad=melspec_config.pad_value)

    out = {
            "target": audios,
            # "target_pos" : wave_pos,
            "mel": mels,
            # "mel_pos": mel_pos
    }

    return out


def get_collator(batch_expand_size, melspec_config):
    def collate_fn_tensor(batch):
        len_arr = np.array([d["target"].shape(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = batchsize // batch_expand_size

        cut_list = list()
        for i in range(batch_expand_size):
            cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

        output = list()
        for i in range(batch_expand_size):
            output.append(reprocess_tensor(batch, cut_list[i], melspec_config))

        return output
    return collate_fn_tensor


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def get_LJSpeech_dataloader(dataset_config, melspec_config):
    buffer = get_data_to_buffer(dataset_config)
    dataset = BufferDataset(buffer)
    return DataLoader(
        dataset,
        batch_size=dataset_config['batch_expand_size'] * dataset_config['batch_size'],
        shuffle=True,
        collate_fn=get_collator(dataset_config['batch_expand_size'], melspec_config),
        drop_last=True,
        num_workers=dataset_config['num_workers']
    )