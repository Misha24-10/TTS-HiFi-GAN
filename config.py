device = "cuda"
BATCH_SIZE = 16
DATAPATH = "." # path to LJSPEECH
class config:
    MFT_block_dilations = [[1, 1], [3, 1], [5, 1]]
    leaky_relu = 0.1


path_generator = "/content/generator_try44.pt"
path_mpd = "/content/mpd_try44.pt"
path_msd: str = "/content/msd_try44.pt"

path2predictaudio = "/content/TTS-HiFi-GAN/audio2predict"

path_gen_checkpoint = "..."
path_mpd_checkpoint = "..."
path_msd_checkpoint = "..."


class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    center: bool = False


class MelSpectrogramConfig_loss:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = None
    n_mels: int = 80
    power: float = 1.0
    center: bool = False
