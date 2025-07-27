import os
import torch
import torchaudio
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# === 1. Dataset ===

class ClapDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.mel = MelSpectrogram(sample_rate=16000, n_mels=64)
        self.db = AmplitudeToDB()
        self._load_data(root_dir)

    def _load_data(self, root_dir):
        for label, class_dir in enumerate(['not_clap', 'clap']):
            class_path = os.path.join(root_dir, class_dir)
            for file in os.listdir(class_path):
                if file.endswith('.wav'):
                    self.data.append(os.path.join(class_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.data[idx])
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        audio = audio.mean(dim=0, keepdim=True)  # mono
        audio = audio[:, :16000]  # 1 sec clip max

        mel = self.db(self.mel(audio))  # [1, 64, Time]
        mel = mel[:, :, :64]  # crop/pad to consistent shape

        if mel.shape[2] < 64:
            pad = 64 - mel.shape[2]
            mel = torch.nn.functional.pad(mel, (0, pad))

        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return mel, label

# === 2. Model ===

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

# === 3. Train Loop ===

def train(model, loader, device):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for mel, label in loader:
            mel, label = mel.to(device), label.to(device)

            output = model(mel)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# === 4. Main Entry ===

if __name__ == "__main__":
    device = torch.device("cpu")
    dataset = ClapDataset("data")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ClapDetector().to(device)
    train(model, loader, device)

    torch.save(model.state_dict(), "clap_detector.pt")
    print("✅ Model saved as clap_detector.pt")
