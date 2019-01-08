# coding: utf-8
"""
Mai Fina v1.0
*Oct. 2018*

Run command:
python3 run_maifina.py <path_to_seeds> <path_to_output>

Needs:
A folder named "model" containing:
- './model/word_indices.pickle'
- './model/indices_word.pickle'
- a trained model with extension '.h5'

"""

### Imports

from pickle import load
from keras.models import load_model
import numpy as np
import io
import os
import sys
import re
from fuzzywuzzy import process

### Parameters

# Sequence length used during training
SEQUENCE_LEN = 10

# Dictionaries of accepted words
DIC_WORD_IND = './model/word_indices.pickle'
DIC_IND_WORD = './model/indices_word.pickle'

### Helper functions

# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    """
    Computes the next word using the weights obtained from the RNN. Chooses the
    next word sampling from a multinomial distribution. A `temperature` is used
    to broaden the original values.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Clean seeds
def clean_input(sentence, all_words, fw_limit=90):
    """
    Clean sentences from unwanted characters
    sentence : sentence to be cleaned
    all_words : acepted words in the model
    fw_limit (default=90) : FuzzyWuzzy cutoff

    """

    # Cleaning with regex
    new_s = re.sub(r'  ', ' ', sentence)
    new_s = re.sub('"|\)|\_|\?|\¿|\-|\(|\.|\,|\:|\;|\¡|\!|\“|\”|\*|\[|\]', '', new_s)
    new_s = re.sub(r'\s+', ' ', new_s)
    new_s = re.sub(r'a{3,}', 'a', new_s)
    new_s = re.sub(r'e{3,}', 'e', new_s)
    new_s = re.sub(r'i{3,}', 'i', new_s)
    new_s = re.sub(r'o{3,}', 'o', new_s)
    new_s = re.sub(r'u{3,}', 'u', new_s)
    new_s = re.sub(r'’', '', new_s)
    new_s = re.sub(r'`', '', new_s)
    new_s = re.sub(r'\'', '', new_s)
    new_s = re.sub(r'(à|á|â|ä)', 'a', new_s)
    new_s = re.sub(r'(è|é|ê|ë)', 'e', new_s)
    new_s = re.sub(r'(ì|í|î|ï)', 'i', new_s)
    new_s = re.sub(r'(ò|ó|ô|ö)', 'o', new_s)
    new_s = re.sub(r'(ù|ú|û|ü)', 'u', new_s)

    new_s = new_s.split(' ')

    final_sentence = []
    for w in new_s:
        # Accept word if it is in the dictionary
        if w in all_words:
            final_sentence.append(w)
        else:
            # Find fuzzy similar words in the dictionary
            results = process.extract(w, all_words, limit = 1)
            # If there is a similar word in the dictionary above the
            # FuzzyWuzzy limit fw_limit, change it to that
            if results[0][1] > fw_limit:
                final_sentence.append(results[0][0])
            # If they are not close, use the unknown word token
            else:
                final_sentence.append('<unk>')

    return ' '.join(final_sentence)


# Generate examples on epoch end
def generate_text(model, seed, temperature = 0.7, length = 50):
    """
    Generates text from seed.
    model : trained model
    seed : a seed sentence
    temperature (default=0.7) : temperature used for the sampling
    length (default=50) : number of words (and line breaks) to be generated
    """

    str_sentence = seed

    clean_seed = clean_input(seed, all_words)
    clean_seed = clean_seed.split(' ')

    # Use only the last SEQUENCE_LEN words in the seed if it is long
    # Pad sentence with '<unk>' if it is short
    sentence = [0]*SEQUENCE_LEN
    if len(clean_seed) >= SEQUENCE_LEN:
        for i in range(SEQUENCE_LEN):
            sentence[i] = word_indices[clean_seed[-(SEQUENCE_LEN-i)]]
    else:
        for i in range(len(clean_seed)):
            sentence[-(i+1)] = word_indices[clean_seed[-(i+1)]]

    # Loop over worrds to be generated
    for i in range(length):
        x_pred = np.zeros((1, SEQUENCE_LEN))
        for t, word in enumerate(sentence):
            x_pred[0, t] = word

        preds = model.predict(x_pred, verbose=0)[0]
        # Remove <unk> as a possible outcome
        preds = preds[1:]
        next_index = sample(preds, temperature) + 1
        next_word = indices_word[next_index]

        sentence = sentence[1:]
        sentence.append(next_index)
        str_sentence = str_sentence + ' ' + next_word

    return str_sentence

#########################################################
# Main

if __name__ == "__main__":
    # Argument check
    if len(sys.argv) != 3:
        print('\033[91m' + 'Argument Error!\nUsage: python3 run_maifina.py <path_to_seeds> <path_to_output>' + '\033[0m')
        exit(1)
    if not os.path.isfile(sys.argv[1]):
        print('\033[91mERROR: ' + sys.argv[1] + ' is not a file!' + '\033[0m')
        exit(1)

    seeds = sys.argv[1]
    out_file = sys.argv[2]

    # Find weigths in the model folder
    files = [f for f in os.listdir('./model/')]
    for f in files:
        if f[-3:] == '.h5':
            TRAINED_MODEL = os.path.join('./model/', f)

    # Load dictionaries of accepted words
    with open(DIC_WORD_IND, 'rb') as handle:
        word_indices = load(handle)
    with open(DIC_IND_WORD, 'rb') as handle:
        indices_word = load(handle)

    all_words = list(word_indices.keys())

    # Load trained model
    model = load_model(TRAINED_MODEL)

    # Load seeds
    with io.open(seeds, encoding='utf-8') as f:
        # Raw input
        text = f.read().lower().split('\n')
        text = text[:-1]

    with open(out_file, "w") as text_file:
        for seed in text:
            text_file.write(generate_text(model, seed))
            text_file.write('\n')
            text_file.write('\n')
