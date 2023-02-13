import torchaudio
import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


def extract_speech_embeddings(path):
    signal, fs = torchaudio.load(path)
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, orig_freq=fs, new_freq=16000)
        fs = 16000
    embeddings = classifier.encode_batch(signal).detach().view(-1)
    np_array = embeddings.numpy()
    np_array = np_array / np.linalg.norm(np_array)
    return np_array


if __name__ == "__main__":
    f1 = '../dumps/eastwood_lawyers.wav'
    f2 = '../dumps/hawking01.wav'
    f3 = '../dumps/hawking02.wav'
    f4 = '../dumps/punk.wav'
    f5 = '../dumps/your_drawers.wav'
    e1 = extract_speech_embeddings(f1)
    e2 = extract_speech_embeddings(f2)
    e3 = extract_speech_embeddings(f3)
    e4 = extract_speech_embeddings(f4)
    e5 = extract_speech_embeddings(f5)
    print(e1 @ e2)
    print(e2 @ e3)
    print(e4 @ e5)
