from scr.Datasets.dataset import LJSpeechDataset, LJSpeechCollator, LJSpeechDataset_fullaudio
from config import MelSpectrogramConfig, MelSpectrogramConfig_loss, device, path_generator, path_mpd, \
    path_msd
import torch
from scr.Melspec.melspec import MelSpectrogram, MelSpectrogramConfig_loss, MelSpectrogramConfig
from torch.utils.data import  Subset
from torch.utils.data import DataLoader
import wandb
from scr.Model.model import MSD, Generator, MPD
import itertools
from itertools import islice

from tqdm import tqdm
from IPython import display
import torch.nn.functional as F
from scr.losses.losses import generator_loss, discriminator_loss, feature_matching_loss
from matplotlib import pyplot as plt


def get_dataloader(datapath='.', batchSize=16):
    dataset = LJSpeechDataset('.', MelSpectrogramConfig(), MelSpectrogramConfig_loss(), device)
    train_ratio = 0.95
    torch.manual_seed(41)
    train_size = int(len(dataset) * train_ratio)
    valtidation_size = len(dataset) - train_size

    indexes = torch.randperm(len(dataset))
    train_indexes = indexes[:train_size]
    validation_indexes = indexes[train_size:]
    train_dataset = Subset(dataset, train_indexes)
    validation_dataset = Subset(dataset, validation_indexes)

    dataloader_train  = DataLoader(train_dataset, batch_size=batchSize, collate_fn= LJSpeechCollator())
    dataloader_valid = DataLoader(validation_dataset, batch_size=batchSize, collate_fn=LJSpeechCollator())
    return dataloader_train, dataloader_valid

def train():
    wandb.login(key='358c4114387c5c7ca207c32ba4343e7c86efc182')
    wandb.init(project='TTS_mel2wav', entity='mishaya') # username in wandb

    dataloader_train, dataloader_valid = get_dataloader()

    melspec = MelSpectrogram(MelSpectrogramConfig()).to(device)
    melspec_loss = MelSpectrogram(MelSpectrogramConfig_loss()).to(device)

    generator = Generator().to(device)
    mpd = MPD().to(device)
    msd = MSD().to(device)
    generator.load_state_dict(torch.load(path_generator))
    mpd.load_state_dict(torch.load(path_mpd))
    msd.load_state_dict(torch.load(path_msd))
    optim_g = torch.optim.AdamW(generator.parameters(), 0.003, betas=[0.8, 0.99])
    optim_disc = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), 0.0002, betas=[0.8, 0.99])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.999, last_epoch=-1)

    generator.train()
    mpd.train()
    msd.train()
    losses = []
    dataloader_fullaudo = DataLoader(LJSpeechDataset_fullaudio('.', MelSpectrogramConfig(), MelSpectrogramConfig_loss(), device), batch_size=1 )
    for epoch in range(100):
        for i, batch in islice(enumerate(tqdm(dataloader_train)),len(dataloader_train)):
            y, x, y_mel = batch.waveform, batch.mels, batch.mel_loss
            y = y.unsqueeze(1)
            y_audio_pred = generator(x)
            y_audio_pred_mel = melspec_loss(y_audio_pred.squeeze(1)).squeeze(1)
            optim_disc.zero_grad()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_audio_pred.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_audio_pred.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_discriminator = loss_disc_s + loss_disc_f
            loss_discriminator.backward()
            optim_disc.step()
            optim_g.zero_grad()
            loss_mel = F.l1_loss(y_mel, y_audio_pred_mel)


            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_audio_pred)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_audio_pred)
            loss_fm_f = feature_matching_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_matching_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_generator =   45 * loss_mel + loss_gen_s + loss_gen_f +loss_fm_s + loss_fm_f
            wandb.log({"loss generator": loss_generator,
                       "loss discriminator:": loss_discriminator,
                       "Mel-Spectrogram Loss in TRAIN": loss_mel})

            loss_generator.backward()
            optim_g.step()
            scheduler_g.step()
            scheduler_d.step()

            if i % 100 == 0:
                print(epoch * len(dataloader_train) + i)
                losses.append(loss_generator)
                print(loss_generator)
                display.display(display.Audio(y_audio_pred.squeeze(1)[0].cpu().detach().numpy(), rate=22050))
                plt.imshow(y_audio_pred_mel.squeeze()[0].cpu().detach().numpy())
                plt.imshow(y_mel.squeeze()[0].cpu().detach().numpy())

                ex = list(islice(dataloader_fullaudo, 1))[0][1]
                prim = generator(ex)

                wandb.log({
                    "Audio valid every n step": [wandb.Audio(y_audio_pred.squeeze(1)[0].cpu().detach().numpy(), caption="Audio in train", sample_rate=22050)],
                    "Audio in train ground true": [wandb.Audio(y.squeeze(1)[0].cpu().detach().numpy(), caption="Audio in train ground true", sample_rate=22050)],
                    "Example of audio": [wandb.Audio(prim.squeeze().cpu().detach().numpy(), caption="Example of audio", sample_rate=22050)],
                    "Spec train ": [wandb.Image(plt.imshow(y_audio_pred_mel.squeeze()[0].cpu().detach().numpy()), caption="Spec train ")],
                    "Spec train GroundTrue": [wandb.Image(plt.imshow(y_mel.squeeze()[0].cpu().detach().numpy()), caption="Spec train GroundTrue")]
                })


        generator.eval()
        val_err_tot = 0
        with torch.no_grad():
            for j, batch in enumerate(dataloader_valid):
                y, x, y_mel = batch.waveform, batch.mels, batch.mel_loss
                y = y.unsqueeze(1)
                y_audio_pred = generator(x)
                y_audio_pred_mel = melspec_loss(y_audio_pred.squeeze(1)).squeeze(1)
                if j % 20 == 0:
                    print(j)
                    display.display(display.Audio(y_audio_pred.squeeze(1)[0].cpu().detach().numpy(), rate=22050))
                    display.display(display.Audio(y.squeeze(1)[0].cpu().detach().numpy(), rate=22050))
                val_err_tot += F.l1_loss(y_mel, y_audio_pred_mel).item()
            val_err = val_err_tot / (j+1)
            wandb.log({"Mel-Spectrogram Loss in VALIDATION": val_err})
            print("Mel-Spectrogram Loss in VALIDATION", val_err)
        generator.train()
        torch.save(generator.state_dict(), f'/content/drive/MyDrive/AUDIO_DLA/TTS GAN/log/generator_try4{epoch}.pt')
        torch.save(mpd.state_dict(), f'/content/drive/MyDrive/AUDIO_DLA/TTS GAN/log/mpd_try4{epoch}.pt')
        torch.save(msd.state_dict(), f'/content/drive/MyDrive/AUDIO_DLA/TTS GAN/log/msd_try4{epoch}.pt')

if __name__ == '__main__':
    train()
