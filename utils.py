import json
import re

from os.path import commonprefix
from collections import defaultdict

import numpy as np

from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

POS_MAP = {
    'adjective': 'ADJ', 
    'adverb': 'ADV', 
    'auxiliary': 'AUX', 
    'conjunction': 'CONJ', 
    'definite': 'DET', 
    'indefinite': 'DET', 
    'interjection': 'INTJ', 
    'noun': 'NOUN', 
    'plural': 'NOUN', 
    'preposition': 'ADP', 
    'pronoun': 'PRON', 
    'pronoun;': 'PRON', 
    'verb': 'VERB'
}

def glove2dict(src_filename):
    """GloVe Reader.

    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.

    Returns
    -------
    dict
        Mapping words to their GloVe vectors.

    """
    data = {}
    with open(src_filename,  encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data

def tabularize_dictionary(load_path, save_path, max_defs=0):

    with open(load_path) as f:
        dictionary = json.load(f)

    lmt = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    words = sorted(list(dictionary.keys()))
    with open(save_path, 'w') as f:
        for word in words:
            if not dictionary[word]:
                continue

            if max_defs == 0:
                n = len(dictionary[word])
            else:
                n = max_defs

            pos2defs = defaultdict(list)
            for pos, definition in dictionary[word]:
                if pos is not None:
                    pos = pos.lower()
                if pos == 'idioms':
                    continue
                pos = POS_MAP.get(pos)
                pos2defs[pos].append(definition)

            tag = None
            for p in ['VERB', 'ADJ']:
                lemma = lmt(word, p)[0]
                if lemma != word and\
                    dictionary.get(lemma) != dictionary.get(word):
                    prefix = commonprefix([lemma, word])
                    suffix = word[len(prefix):]
                    if p == 'VERB':
                        if suffix.endswith('ing'):
                            tag = 'VBG'
                        elif suffix.endswith('s'):
                            tag = 'VBZ'
                        elif suffix.endswith('d'):
                            tag = 'VBD'
                        elif suffix.endswith('n'):
                            tag = 'VBN'
                    elif p == 'NOUN':
                        tag = 'NNS'
                    elif p == 'ADJ':
                        if suffix.endswith('t'):
                            tag = 'JJS'
                        elif suffix.endswith('r'):
                            tag = 'JJR'

                    if dictionary.get(lemma):
                        for pos, definition in dictionary[lemma]:
                            if pos is not None:
                                pos = pos.lower()
                            if pos == 'idioms':
                                continue
                            pos = POS_MAP.get(pos)
                            if len(definition) == 0:
                                continue
                            if pos == p:
                                f.write("%s\t%s\t%s\t%s\n" % (word, pos, tag, definition))


            for pos in pos2defs:
                for definition in pos2defs[pos][:n]:
                    if len(definition) == 0:
                        continue
                    
                    tag = None
                    skip = 0

                    for p in ['VERB', 'NOUN', 'ADJ']:
                        lemma = lmt(word, p)[0]
                        if lemma != word and\
                           dictionary.get(lemma) == dictionary.get(word):
                            if p == pos:
                                prefix = commonprefix([lemma, word])
                                suffix = word[len(prefix):]
                                if pos == 'VERB':
                                    if suffix.endswith('ing'):
                                        tag = 'VBG'
                                    elif suffix.endswith('s'):
                                        tag = 'VBZ'
                                    elif suffix.endswith('d'):
                                        tag = 'VBD'
                                    elif suffix.endswith('n'):
                                        tag = 'VBN'
                                elif pos == 'NOUN':
                                    tag = 'NNS'
                                elif pos == 'ADJ':
                                    if suffix.endswith('t'):
                                        tag = 'JJS'
                                    elif suffix.endswith('r'):
                                        tag = 'JJR'
                                if skip == 1:
                                    f.write("%s\t%s\t%s\t%s\n" % (word, pos, tag, definition))
                                break
                            else:
                                skip = 1

                    if not skip:
                        f.write("%s\t%s\t%s\t%s\n" % (word, pos, tag, definition))
