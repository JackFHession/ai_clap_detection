import numpy as np
import tensorflow_hub as hub
import sounddevice as sd
from scipy.spatial.distance import cosine
from train import load_clap_model, extract_embedding, SAMPLE_RATE

# listen for this long
RECORD_SECONDS = 1.0

# load the sound brain
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

def record_audio(duration=1.0):
    # record from mic, duh
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    return audio.flatten()

def detect_clap(clap_embedding, threshold):
    audio = record_audio(RECORD_SECONDS)
    try:
        emb = extract_embedding(audio)
        similarity = 1 - cosine(clap_embedding, emb)
        if similarity >= threshold:
            print(f"🔎 Similarity: {similarity:.3f}")
            print("👏 Clap detected!")
            return True
        else:
            return False
    except Exception as e:
        print(f"⚠️ wtf: {e}")
    return False

if __name__ == "__main__":
    # boot up
    clap_embedding, threshold = load_clap_model()
    print("🟢 Listening for claps. Do it.")

    while True:
        detect_clap(clap_embedding, threshold)
