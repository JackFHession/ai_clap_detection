# clap detector

a simple clap detector using a custom pytorch model. it listens to your microphone in real time and detects claps using a trained convolutional network.

## how to use

1. train the model using `train.py` (coming soon, or use the provided `clap_detector.pt`)  
2. run `python main.py` to test for a single clap  
3. or import and call `infer_clap(model, processor, device)` inside your own code  
4. optionally, integrate into your voice assistant as a wakeword trigger

## how it works

the program converts audio into mel spectrograms (like images of sound).  
a small neural network classifies whether the spectrogram contains a clap.  
if the probability is high enough, it says “clap detected.”

## tips

- best if you're in a quiet environment with a clear clap sound  
- clap volume, timing, and background noise can affect detection accuracy  
- model is saved as `./ltm/clap_detector.pt` — keep it safe and load it during runtime

## about

made by a @JackHession who likes tinkering with ai.  
for other projects, check [hession dynamics on github](https://github.com/Hession-Dynamics) and also [jack, the ceo](https://github.com/jackhession).

