import numpy as np
import tensorflow_hub as hub
import os
import scipy.signal
import scipy.io.wavfile as wav
from scipy.spatial.distance import cosine

# constants innit
SAMPLE_RATE = 16000
CLAP_DIR = "./audio_files/"
MODEL_PATH = "./clap_model.npz"
MARGIN = 0.03  # tweak this if detection is being annoying

# yamnet go brr
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

def load_audio_file(path):
    sr, audio = wav.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mono is easier and no one’s got time for stereo
    if sr != SAMPLE_RATE:
        # force it to match sample rate bc yamnet is picky
        duration = len(audio) / sr
        target_len = int(SAMPLE_RATE * duration)
        audio = scipy.signal.resample(audio, target_len)
    return audio.astype(np.float32)

def extract_embedding(waveform):
    # scale to float32 because yamnet has standards apparently
    waveform = waveform / 32768.0
    _, embeddings, _ = yamnet(waveform)
    return np.mean(embeddings.numpy(), axis=0)

def train_clap_model(folder=CLAP_DIR):
    embeddings = []

    for fname in os.listdir(folder):
        if fname.endswith('.wav'):
            audio = load_audio_file(os.path.join(folder, fname))
            emb = extract_embedding(audio)
            embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("🛑 you forgot to add training .wav files")

    # average all the embeddings together – this is your ‘ideal’ clap
    avg_emb = np.mean(embeddings, axis=0)

    # compare each sample to the average and get similarity
    similarities = [1 - cosine(avg_emb, e) for e in embeddings]
    min_sim = min(similarities)
    max_sim = max(similarities)
    std = np.std(similarities)

    # set threshold low enough to detect claps but not everything else
    threshold = min_sim - 0.01 + MARGIN

    # debug info for nerds
    print("🔧 Model trained:")
    print(f"   Min similarity: {min_sim:.3f}")
    print(f"   Max similarity: {max_sim:.3f}")
    print(f"   Std deviation:  {std:.4f}")
    print(f"   → Threshold set to: {threshold:.3f}")

    # yeet it into a file
    np.savez(MODEL_PATH, embedding=avg_emb, threshold=threshold)
    print(f"💾 Saved to {MODEL_PATH}")
    return avg_emb, threshold

def load_clap_model():
    if not os.path.exists(MODEL_PATH):
        print("📁 No model? Training one now.")
        return train_clap_model(CLAP_DIR)

    print(f"📦 Loading model from {MODEL_PATH}")
    data = np.load(MODEL_PATH)
    return data['embedding'], float(data['threshold'])

# run this file alone = retrain the model
if __name__ == "__main__":
    train_clap_model()
