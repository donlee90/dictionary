from collections import defaultdict
import csv
import numpy as np
import os
import pandas as pd
from scipy.stats import spearmanr
import vsm

wordsim_home = os.path.join('vsmdata', 'wordsim')

def wordsim_dataset_reader(src_filename, header=False, delimiter=','):    
    """Basic reader that works for all four files, since they all have the 
    format word1,word2,score, differing only in whether or not they include 
    a header line and what delimiter they use.
    
    Parameters
    ----------
    src_filename : str
        Full path to the source file.        
    header : bool (default: False)
        Whether `src_filename` has a header.        
    delimiter : str (default: ',')
        Field delimiter in `src_filename`.
    
    Yields
    ------    
    (str, str, float)
       (w1, w2, score) where `score` is the negative of the similarity 
       score in the file so that we are intuitively aligned with our 
       distance-based code.
    
    """
    with open(src_filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader)
        for row in reader:
            w1, w2, score = row
            # Negative of scores to align intuitively with distance functions:
            score = -float(score)
            yield (w1, w2, score)

def wordsim353_reader():
    """WordSim-353: http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/"""
    src_filename = os.path.join(wordsim_home, 'wordsim353.csv')
    return wordsim_dataset_reader(src_filename, header=True)
 
def mturk287_reader():
    """MTurk-287: http://tx.technion.ac.il/~kirar/Datasets.html"""
    src_filename = os.path.join(wordsim_home, 'MTurk-287.csv')
    return wordsim_dataset_reader(src_filename, header=False)
    
def mturk771_reader():
    """MTURK-771: http://www2.mta.ac.il/~gideon/mturk771.html"""
    src_filename = os.path.join(wordsim_home, 'MTURK-771.csv')
    return wordsim_dataset_reader(src_filename, header=False)

def men_reader():
    """MEN: http://clic.cimec.unitn.it/~elia.bruni/MEN"""
    src_filename = os.path.join(wordsim_home, 'MEN_dataset_natural_form_full')
    return wordsim_dataset_reader(src_filename, header=False, delimiter=' ')   

def word_similarity_evaluation(reader, df, distfunc=vsm.cosine, verbose=True):
    """Word-similarity evalution framework.
    
    Parameters
    ----------
    reader : iterator
        A reader for a word-similarity dataset. Just has to yield
        tuples (word1, word2, score).    
    df : pd.DataFrame
        The VSM being evaluated.        
    distfunc : function mapping vector pairs to floats (default: `vsm.cosine`)
        The measure of distance between vectors. Can also be `vsm.euclidean`, 
        `vsm.matching`, `vsm.jaccard`, as well as any other distance measure 
        between 1d vectors.  
    verbose : bool
        Whether to print information about how much of the vocab
        `df` covers.
    
    Prints
    ------
    To standard output
        Size of the vocabulary overlap between the evaluation set and
        rownames. We limit the evalation to the overlap, paying no price
        for missing words (which is not fair, but it's reasonable given
        that we're working with very small VSMs in this notebook).
    
    Returns
    -------
    float
        The Spearman rank correlation coefficient between the dataset
        scores and the similarity values obtained from `mat` using 
        `distfunc`. This evaluation is sensitive only to rankings, not
        to absolute values.
    
    """    
    sims = defaultdict(list)
    rownames = df.index
    vocab = set()    
    excluded = set()
    for w1, w2, score in reader():
        if w1 in rownames and w2 in rownames:
            sims[w1].append((w2, score))
            sims[w2].append((w1, score))
            vocab |= {w1, w2}
        else:
            excluded |= {w1, w2}
    all_words = vocab | excluded
    if verbose:
        print("Evaluation vocab: {:,} of {:,}".format(len(vocab), len(all_words)))
    # Evaluate the matrix by creating a vector of all_scores for data
    # and all_dists for mat's distances. 
    all_scores = []
    all_dists = []
    for word in vocab:
        vec = df.loc[word]
        vals = sims[word]
        cmps, scores = zip(*vals)
        all_scores += scores
        all_dists += [distfunc(vec, df.loc[w]) for w in cmps]
    rho, pvalue = spearmanr(all_scores, all_dists)
    return rho

def full_word_similarity_evaluation(df, verbose=True):
    """Evaluate a VSM against all four datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    dict
        Mapping dataset names to Spearman r values
        
    """        
    scores = {}
    for reader in (wordsim353_reader, mturk287_reader, mturk771_reader, men_reader):        
        if verbose: 
            print("="*40)
            print(reader.__name__)
        score = word_similarity_evaluation(reader, df, verbose=verbose)
        scores[reader.__name__] = score
        if verbose:            
            print('Spearman r: {0:0.03f}'.format(score))
    mu = np.array(list(scores.values())).mean()
    if verbose:
        print("="*40)
        print("Mean Spearman r: {0:0.03f}".format(mu))
    return scores