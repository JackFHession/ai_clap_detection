import torch
import torchaudio
import pyaudio
import numpy as np
from torch import nn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import time
import struct

# === Model ===

class ClapDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === Preprocessor ===

class AudioProcessor:
    def __init__(self):
        self.mel = MelSpectrogram(sample_rate=16000, n_mels=64)
        self.db = AmplitudeToDB()

    def transform(self, audio):
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        mel = self.db(self.mel(audio_tensor))  # [1, 64, T]
        mel = mel[:, :, :64]
        if mel.shape[2] < 64:
            pad = 64 - mel.shape[2]
            mel = torch.nn.functional.pad(mel, (0, pad))
        return mel.unsqueeze(0)  # [B, 1, 64, 64]

# === PyAudio Stream Setup ===

def infer_clap(model, processor, device):
    CHUNK = 16000  # 1 second @ 16kHz
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    print("🎤 Listening for claps... (Ctrl+C to stop)")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.array(struct.unpack(str(CHUNK) + 'h', data), dtype=np.int16)
            audio = audio.astype(np.float32)
            audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-5)

            with torch.no_grad():
                mel = processor.transform(audio).to(device)
                prob = model(mel).item()
                if prob > 0.9:
                    print(f"👏 Clap detected! Confidence: {prob:.2f}")
                else:
                    print(f"🤫 No clap. Confidence: {prob:.2f}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

# === Main ===

if __name__ == "__main__":
    device = torch.device("cpu")

    model = ClapDetector().to(device)
    model.load_state_dict(torch.load("clap_detector.pt", map_location=device))
    model.eval()

    processor = AudioProcessor()
    infer_clap(model, processor, device)
