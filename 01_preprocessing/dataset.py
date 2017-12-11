import codecs, re, random
from collections import Counter
from mlxtend.preprocessing import one_hot
import numpy as np

# VOCAB_SIZE = 20000
# VOCAB_TAGS = 10
# MAX_SEQ_LENGTH = 30

# function to get vocab, maxvocab
# takes sents : list (tokenized lists of sentences)
# takes maxvocab : int (maximum vocab size incl. UNK, PAD
# takes stoplist : list (words to ignore)
# returns vocab_dict (word to index), inv_vocab_dict (index to word)
def get_vocab(sents, maxvocab=25000, stoplist=[], verbose=False):

    # get vocab list
    vocab = []
    for sent in sents:
        for word in sent:
            vocab.append(word)

    counts = Counter(vocab) # get counts of each word
    vocab_set = list(set(vocab)) # get unique vocab list
    sorted_vocab = sorted(vocab_set, key=lambda x: counts[x], reverse=True) # sort by counts
    sorted_vocab = [i for i in sorted_vocab if i not in stoplist]
    if verbose:
        print("\ntotal vocab size:", len(sorted_vocab), '\n')
    sorted_vocab = sorted_vocab[:maxvocab-2]
    if verbose:
        print("\ntrunc vocab size:", len(sorted_vocab), '\n')
    vocab_dict = {k: v+1 for v, k in enumerate(sorted_vocab)}
    vocab_dict['UNK'] = len(sorted_vocab)+1
    vocab_dict['PAD'] = 0
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}
    return vocab_dict, inv_vocab_dict


# function to convert sents to indexed vectors
# takes list : sents (tokenized sentences)
# takes dict : vocab (word to idx mapping)
# returns list of lists of indexed sentences
def index_sents(sents, vocab_dict, verbose=False):
    if verbose:
        print("starting vectorize_sents()...")
    vectors = []
    # iterate thru sents
    for sent in sents:
        sent_vect = []
        for word in sent:
            if word in vocab_dict.keys():
                idx = vocab_dict[word]
                sent_vect.append(idx)
            else: # out of max_vocab range or OOV
                sent_vect.append(vocab_dict['UNK'])
        vectors.append(sent_vect)
    return(vectors)


# one-hot vectorizes a list of indexed vectors
# takes matrix : list of lists (indexed-vectorized sents)
# takes num : number of total classes (length of one-hot arrays)
# returns one-hot array matrix
def onehot_vectorize(matrix, num):
    result = []
    for vector in matrix:
        a = one_hot(vector.tolist(), dtype='int', num_labels=num)
        result.append(a)
    return np.array(result)


# decode an integer-indexed sequence
# takes indexed_list : one integer-indexedf sentence (list or array)
# takes inv_vocab_dict : dict (index to word)
# returns list of string tokens
def decode_sequence(indexed_list, inv_vocab_dict):
    str = []
    for idx in indexed_list:
        # print(intr)
        str.append(inv_vocab_dict[int(idx)])
    return(str)


# todo: fix/comment this shit
# keras code
# https://github.com/fchollet/keras/issues/2708
# https://github.com/fchollet/keras/issues/1627
def dataGenerator(X, y, vocabsize, batch_size, epochsize):

    i = 0

    while True:
        y_batch = onehot_vectorize(y[i:i + batch_size], vocabsize)
        yield (X[i:i + batch_size], y_batch)
        if i + batch_size >= epochsize:
            i = 0
        else:
            i += batch_size


# def dataonehotGenerator(batch_size,
#                   input_filepath='savedata/',
#                   xfile='X_train.npy',
#                   yfile='y_train_lex.npy',
#                   vocabsize=VOCAB_SIZE,
#                   posvocabsize=VOCAB_TAGS,
#                   epochsize=300000):
#
#     i = 0
#     X = np.load(input_filepath + xfile)
#     y = np.load(input_filepath + yfile)
#
#     while True:
#         # add in data reading/augmenting code here
#         X_batch = onehot_vectorize(X[i:i + batch_size], vocabsize)
#         y_batch = onehot_vectorize(y[i:i + batch_size], posvocabsize)
#         yield (X_batch, y_batch)
#         if i + batch_size >= epochsize:
#             i = 0
#         else:
#             i += batch_size
