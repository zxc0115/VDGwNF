# Introduction

It is a PyTorch implementation of the Dialog Flow-VAE model.

## Dependency
 - PyTorch 0.4.0
 - Python 3.6
 - NLTK
 ```
 pip install -r requirements.txt
 ```

## Train
- Use pre-trained Word2vec
  Download Glove word embeddings `glove.twitter.27B.200d.txt` from https://nlp.stanford.edu/projects/glove/ and save it to the `./data` folder. The default setting use 200 dimension word embedding trained on Twitter.

- Modify the arguments at the top of `train.py`

- Train model by
  ```
    python train.py 
  ```
The logs and temporary results will be printed to stdout and saved in the `./output` path.

## Evaluation
Modify the arguments at the bottom of `sample.py`
    
Run model testing by:
```
    python sample.py
```
The outputs will be printed to stdout and generated responses will be saved at `results.txt` in the `./output` path.


# References
<a id="1">[1]</a>  X. Gu, K. Cho, J.-W. Ha, and S. Kim, “DialogWAE: Multimodal response gen-eration with conditional wasserstein auto-encoder,” 2018.
