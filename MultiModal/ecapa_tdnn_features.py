import torchaudio
import numpy as np
import math
import os
import pathlib
import random
import torch
import librosa
import matplotlib.pyplot as plt

def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.savefig('myfig.png')

class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=2):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ['remix', '1'], # convert to mono
            ['rate', str(self.sample_rate)], # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length-audio_length)
            noise = noise[..., offset:offset+audio_length]
        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length-noise_length))], dim=-1)

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        print(snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise ) / 2


from speechbrain.pretrained import EncoderClassifier
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
noise_transform = RandomBackgroundNoise(16000, '../dumps/voxceleb/data/')


def extract_speech_embeddings(path):
    signal, fs = torchaudio.load(path)
    signal = noise_transform(signal)
    audio_spectogram = torchaudio.transforms.Spectrogram()(signal)
    print(audio_spectogram.shape)
    plot_spectrogram(audio_spectogram[0, : , :].numpy())
    embeddings = classifier.encode_batch(signal).detach().view(-1)
    np_array = embeddings.numpy()
    np_array = np_array / np.linalg.norm(np_array)
    return np_array


if __name__ == "__main__":
    # f1 = '../dumps/eastwood_lawyers.wav'
    f2 = '/home2/praguna.manvi/plug_env/noise_based_authentication_methods/dumps/voxceleb/data/id00155/1FcFbeC42No/00001.wav'
    f3 = '/home2/praguna.manvi/plug_env/noise_based_authentication_methods/dumps/voxceleb/data/id00155/1FcFbeC42No/00005.wav'
    # f3 = '/home2/praguna.manvi/plug_env/noise_based_authentication_methods/dumps/voxceleb/data/id00678/2vD1571jdyY/00006.wav'
    # f4 = '../dumps/punk.wav'
    # f5 = '../dumps/your_drawers.wav'
    # e1 = extract_speech_embeddings(f1)
    for i in range(1):
        e2 = extract_speech_embeddings(f2)
        e3 = extract_speech_embeddings(f3)

        # e4 = extract_speech_embeddings(f4)
        # e5 = extract_speech_embeddings(f5)
        # print(e1 @ e2)
        print(e2 @ e3)
        # print(e4 @ e5)
