# multiLSTM

## tentative name: `iNERt network`

for **i**ntent, **N**amed **E**ntity **R**ecognition, and **t**opic identification network

## requirements
```
gensim
h5py
keras
keras-contrib
matplotlib
mlxtend
tensorflow
nltk
pandas
pycrfsuite
sklearn
```

this also uses the `keras` attention layer by nigeljyng implemented here:
https://gist.github.com/nigeljyng/37552fb4869a5e81338f82b338a304d3

## purpose

unify "intent detection" of syntactic-pragmatic `speech act` 'intents', semantic `topic` and `named entity` detection into a single neural network. this would allow chat systems to use a single architecture to detect and identify .

## previous work

*Multi-Domain Joint Semantic Frame Parsing using bi-RNN-LSTM* (Hakkani-Tur *et al.* 2016)

- single bi-LSTM network for both NER (using intermediate output) and intent classification (using final output)

## idea

adapt the CoNLL 2003 NER demo with a secondary output to train intent classification, and freeze embeddings to allow future networks to incorporate larger vocabulary (ofc embeddings will need to be retrained, but *simulate* using subset of large embedding matrix by freezing).

network will be modified from the above by using a **CRF layer** for NER and an attention layer for sentence vectorization before classification along multiple axes (speech act, topic).

## todo

1. edit preprocessing scripts to enforce train-test split across tests
2. clean up shared script libraries (`attention.py`, `datasets.py`, `embeddings.py`) 
2. tune model hyperparameters

## network training

1. download the corpus files at http://martinweisser.org
2. extract to the `data` directory
3. run the notebooks in `preprocessing` to generate the network data
4. run the notebook in `training` to train and save a model
5. run the notebooks in `decoding` to evaluate against baselines
6. the class decoder notebook in `decoding` demonstrates a basic idea of implementation in a larger system and allows testing on novel sentences

## note on evaluations

these evaluation numbers are not rigorous analyses because they are conducted over the bootstrapped data (raw data plus synthesized data) using automatic labels that lack human evaluation, and critically, train and test sets were not fixed during testing (after shuffling, train and test indices should be saved and used to reconstitute exact sets for each baseline and experimental trial). they are only presented as rough estimates of performance and to demonstrate that the actual code is functional.

## `sklearn` baseline evaluation for speech acts and topics

these results are from the bootstrapped data, using a linear SVM classifier over tf.idf vectors:

```
# speech act classification

train precision 0.843945255398
train recall    0.726487430623
train accuracy  0.871683309558

test precision 0.764387869858
test recall    0.722603761044
test accuracy  0.809370988447
```

```
# topic classification

train precision 0.882497828431
train recall    0.838802577875
train accuracy  0.934236804565

test precision 0.86166845483
test recall    0.814780976901
test accuracy  0.885750962773
```

## `python-CRFsuite` baseline evaluation for NER

`test accuracy	0.998266078184111`

## `iNERt` results

```
entity accuracy		0.9959235872432259
speech_act accuracy	0.8960205391527599
topic accuracy		0.8241335044929397
```

NB: with adjustments to hyper-parameters, prototype versions of this network have been shown to meet or beat all baselines at once. more tuning is necessary.