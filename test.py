import os
import torchaudio
import torch
from scr.Model.model import Generator
from scr.Melspec.melspec import MelSpectrogram
from config import device, MelSpectrogramConfig, path_generator, path2predictaudio
from IPython import display


def audio2audio(generator,melspec, path):
    waveform, sample_rate = torchaudio.load(path)
    mel = melspec(waveform.to(device)).squeeze()
    aud = generator(mel.unsqueeze(0)).squeeze().cpu().detach()
    display.display(display.Audio(aud.numpy(), rate=22050))
    torchaudio.save(path[:-4] + "_generated_speesch.wav", aud.unsqueeze(0), sample_rate)


if __name__ == '__main__':
    directory = os.getcwd()
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(path_generator))
    generator.eval()
    melspec = MelSpectrogram(MelSpectrogramConfig()).to(device)
    for file in os.listdir(path2predictaudio):
        if file.endswith(".wav"):
            path = f"{path2predictaudio}/{file}"
            print(path)
            audio2audio(generator,melspec, path)