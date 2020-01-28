import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import csv

# Read and preprocess the data
def read(filename, target, pos=False, n=1, clean=True, remove=True, window_size=0, ba=0):
    corpus = []
    f = pd.read_excel(filename)
    column = 'Sentences' if not pos else 'pos'
    for i in f[column].index:
        line = f[column][i]
        # Tokenize line if not pos, split into tags otherwise
        line = word_tokenize(line) if not pos else str(f['pos'][i]).split()
        # Save a lowercased line for clean and remove if not pos, a lemma line otherwise
        search_line = [token.lower() for token in line] if not pos else f['lemma'][i].split()
        # Skip sentences with more or less than one target word
        if clean:
            if search_line.count(target) != 1:
                continue
            if pos and len(line) != len(search_line):
                continue
        # Remove target words
        if remove:
            for j in reversed(range(len(search_line))):
                if search_line[j] == target:
                    line[j] = '<targ>'
            # line = line[:line.index('<targ>')+1]
        # Crop to window (take only 2*window_size+1 words around target)
        if window_size:
            line = ['<start>'] * window_size + line + ['<end>'] * window_size
            line = line[line.index('<targ>')-window_size:line.index('<targ>')+1+window_size]
        # Crop to only before or after <targ>
        if ba < 0:
            line = line[:line.index('<targ>')+1]
        elif ba > 0:
            line = line[line.index('<targ>'):]
        # Remove @ and % characters from pos tags
        if pos:
            for j in range(len(line)):
                line[j] = line[j].replace('@', '').replace('%', '')
        # Form n-grams (if n=1, we get the original tokenized sentence)
        line_ngrams = ngrams(line, n=n,
                             pad_left=ba<=0, left_pad_symbol='<start>',
                             pad_right=ba>=0, right_pad_symbol='<end>')
        line = [' '.join(ngram) for ngram in list(line_ngrams)]
        # Append to corpus
        corpus.append(line)
    return corpus

# Mix 'me' and 'myself' sentences into one data array
def mix_basic(me_corpus, myself_corpus):
    X_raw = []
    num_class_datapoints = min(len(me_corpus), len(myself_corpus))
    for i in range(num_class_datapoints):
        X_raw.append(me_corpus[i])
        X_raw.append(myself_corpus[i])
    y = np.zeros(shape=(2*num_class_datapoints,))
    y[1::2] = 1
    return X_raw, y

# Form a vocabulary using a file with n-gram frequencies
def form_vocab(vocab_size=None, threshold=None, n=None, pos=False):
    vocab = []
    type = 'word' if not pos else 'pos'
    with open('../data/most_common_' + str(n) + 'grams_' + type + '.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if vocab_size and len(vocab) >= vocab_size: break
            elif threshold and int(row[1]) < threshold: break
            vocab.append(row[0])
    return vocab

# Turn each sentence into a vector representation using the vocabulary
def vectorize(X_raw, vocab):
    X = np.zeros(shape=(len(X_raw), len(vocab)))
    for i in range(len(X_raw)):
        line = X_raw[i].copy()
        for j in range(len(vocab)):
            while vocab[j] in line:
                X[i][j] += 1
                line.remove(vocab[j])
    return X
