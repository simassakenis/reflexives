# naive bayes

## Dependencies

* sklearn
* nltk
* pandas
* numpy

## Usage

### Data

Make sure that *data* folder (in the parent directory) contains all of the following files. The layout should look something like this:

```
data/
    most_common_1grams_pos.csv
    most_common_1grams_words.csv
    most_common_2grams_pos.csv
    most_common_2grams_words.csv
    most_common_3grams_pos.csv
    most_common_3grams_words.csv
    me_new_data_clean_15000.xslx
    myself_new_data_clean_15000.xslx
nb/
    main.py
    utils.py
```

### Model

The model is specified by the four main parameters:
* *pos*: whether to train on part of speech data.
* *n*: the size of n-grams (1, 2, or 3).
* *before_after*: set to -1 to look only at the words before &lt;targ&gt;, set to 1 to look only at the words after &lt;targ&gt;.
* *window_size*: number of words before and after &lt;targ&gt; to look at (1 through 10).

Control these parameters via the flags --pos (-p), --n (-n), --before_after (-b), and --window_size (-w).

### Example usage

```
$ python main.py --pos --n 2 --before_after -1 --window_size 7
```
### Output

The code should output one sample datapoint (just to show how the data is preprocessed) and results with three different vocabulary thresholds. Outputs should be printed to the console.
