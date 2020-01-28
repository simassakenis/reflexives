from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pos', action='store_true', default=False,
                    help='train on POS data')
parser.add_argument('-n', '--n', type=int, choices={1, 2, 3}, default=1,
                    help='size of n-grams')
parser.add_argument('-b', '--before_after', type=int, choices={-1, 0, 1}, default=0,
                    help='set to -1 to look only at the words before <targ>, set to 1 to look after')
parser.add_argument('-w', '--window_size', type=int, choices={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, default=0,
                    help='number of words before and after <targ> to look at')
args = parser.parse_args()

pos = args.pos
n = args.n
ba = args.before_after
window_size = args.window_size

# Set up thresholds (frequencies above which n-grams make it into the vocabulary)
thresholds = [1e6, 1e5, 1e4] if not pos else [1000, 100, 10]
# Read and preprocess 'me' and 'myeslf' data
X_raw_n = [None] * n
for k in range(n):
    me_corpus = read('../data/me_new_data_clean_15000.xlsx', 'me', pos=pos, n=k+1, window_size=window_size, ba=ba)
    myself_corpus = read('../data/myself_new_data_clean_15000.xlsx', 'myself', pos=pos, n=k+1, window_size=window_size, ba=ba)
    X_raw_n[k], y = mix_basic(me_corpus, myself_corpus)

# Train the model using different thresholds for vocabularies
for threshold in thresholds:
    # Form vocabularies and vectorize the raw data
    vocabs = [None] * n
    X_n = [None] * n
    for k in range(n):
        vocabs[k] = form_vocab(threshold=threshold, n=k+1, pos=pos)
        X_n[k] = vectorize(X_raw_n[k], vocabs[k])
    # Print sample datapoint
    if threshold == thresholds[0]:
        print('\nsample datapoint (original):\n', 'Let me submit something here .') # copied from excel
        print('sample datapoint (preprocessed):\n', X_raw_n[-1][4])
        print('sample datapoint (vectorized):\n', X_n[-1][4])
        print('vocab (n=%d, threshold=%d):\n' % (n, threshold), np.asarray(vocabs[-1]), '\n')
    # Train using 8-fold cross-validation
    val_size = len(X_n[0]) // 10
    accuracies = []
    for i in range(8):
        val_idxs = list(range(i*val_size, i*val_size+val_size))
        train_idxs = list(range(0, i*val_size)) + list(range(i*val_size+val_size, val_size*4))
        y_train, y_val = y[train_idxs], y[val_idxs]

        X_train_n = [None] * n
        X_val_n = [None] * n
        probs = np.zeros(shape=(len(y_val), 2))
        for k in range(n):
            X_train_n[k], X_val_n[k] = X_n[k][train_idxs], X_n[k][val_idxs]
            classifier = MultinomialNB()
            classifier.fit(X_train_n[k], y_train)
            probs += classifier.predict_proba(X_val_n[k])
        accuracies.append(np.mean(y_val == np.argmax(probs, axis=1)))
    # Print the results and sizes of vocabularies with given thresholds
    print('threshold: %d, mean accuracy: %.4f, std: %.4f' % (threshold, np.mean(accuracies), np.std(accuracies)))
    print('(vocab sizes (for 1 to n n-grams): ' + ', '.join([str(len(vocab)) for vocab in vocabs]) + ')')
