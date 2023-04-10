import torchaudio
import numpy as np
# import torch
from speechbrain.pretrained import EncoderClassifier
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


def extract_speech_embeddings(path):
    signal, fs = torchaudio.load(path)
    embeddings = classifier.encode_batch(signal).detach().view(-1)
    np_array = embeddings.numpy()
    np_array = np_array / np.linalg.norm(np_array)
    return np_array


if __name__ == "__main__":
    # f1 = '../dumps/eastwood_lawyers.wav'
    f2 = '/home2/praguna.manvi/plug_env/noise_based_authentication_methods/dumps/voxceleb/data/id00155/1FcFbeC42No/00001.wav'
    f3 = '/home2/praguna.manvi/plug_env/noise_based_authentication_methods/dumps/voxceleb/data/id00678/2vD1571jdyY/00006.wav'
    # f4 = '../dumps/punk.wav'
    # f5 = '../dumps/your_drawers.wav'
    # e1 = extract_speech_embeddings(f1)
    e2 = extract_speech_embeddings(f2)
    e3 = extract_speech_embeddings(f3)

    # e4 = extract_speech_embeddings(f4)
    # e5 = extract_speech_embeddings(f5)
    # print(e1 @ e2)
    print(e2 @ e3)
    # print(e4 @ e5)
