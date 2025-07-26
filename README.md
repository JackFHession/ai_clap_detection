# clap detector

a simple clap detector using google’s yamnet model. it learns your clap sounds from a few example recordings then listens for claps live.

## how to use

1. add some `.wav` files of your claps into `./clap_detector/train_clips/`  
2. run `python train.py` to train the model  
3. run `python main.py` to start detecting claps  

## how it works

yamnet converts audio into embeddings (basically number summaries of sounds).  
the program averages your training claps to get a “clap profile.”  
when it hears something similar enough, it says “clap detected.”

## tips

- best if you record claps in a quiet place  
- if it detects too much or too little, try adjusting the margin in `train.py`  
- the model is saved in `./ltm/clap_model.npz` so keep it safe if you want to reuse it

## about

made by a random birmingham coder who likes tinkering with ai.  
for other projects, check [franklindynamics on github](https://github.com/FranklinDynamics).
