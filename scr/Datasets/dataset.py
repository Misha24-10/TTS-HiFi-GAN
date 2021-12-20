import torchaudio
from scr.Melspec.melspec import MelSpectrogram
import torch
import random
from config import device
from typing import Tuple, Dict, Optional, List, Union

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from dataclasses import dataclass


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, mel_config, mel_config_loss, device):
        super().__init__(root=root)
        self.segment_size = 8192
        self.split = True
        self.featurizer = MelSpectrogram(mel_config).to(device)
        self.featurizer_loss = MelSpectrogram(mel_config_loss).to(device)

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform = waveform.to(device)

        if self.split:
            if waveform.size(1) >= self.segment_size:
                max_audio_start = waveform.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                waveform = waveform[:, audio_start:audio_start + self.segment_size]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.segment_size - waveform.size(1)), 'constant')
            mel = self.featurizer(waveform.to(device))

            mel_loss = self.featurizer_loss(waveform.to(device))

        return waveform.squeeze(0), mel.squeeze(), mel_loss.squeeze()


class LJSpeechDataset_fullaudio(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, mel_config, mel_config_loss, device):
        super().__init__(root=root)
        self.segment_size = 8192
        self.split = True
        self.featurizer = MelSpectrogram(mel_config).to(device)
        self.featurizer_loss = MelSpectrogram(mel_config_loss).to(device)

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()
        waveform = waveform.to(device)

        mel = self.featurizer(waveform.to(device))

        mel_loss = self.featurizer_loss(waveform.to(device))

        return waveform.squeeze(0), mel.squeeze(), mel_loss.squeeze()


@dataclass
class Batch:
    waveform: torch.Tensor
    mels: torch.Tensor
    mel_loss: torch.Tensor


class LJSpeechCollator:
    def __call__(self, instances) -> Dict:
        waveform, mels, mel_loss = list(zip(*instances))
        waveform = pad_sequence([
            waveform_ for waveform_ in waveform
        ]).transpose(0, 1)

        mels = pad_sequence([
            mels_ for mels_ in mels
        ]).transpose(0, 1)

        mel_loss = pad_sequence([
            mels_ for mels_ in mel_loss
        ]).transpose(0, 1)

        return Batch(waveform, mels, mel_loss)
