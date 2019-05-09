from collections import defaultdict
import csv
import numpy as np
import os
import pandas as pd
from scipy.stats import spearmanr
import vsm

analogies_home = os.path.join('vsmdata', 'question-data')

def analogy_completion(a, b, c, df, distfunc=vsm.cosine):
    """a is to be as c is to predicted, where predicted is the 
    closest to (b-a) + c"""
    for x in (a, b, c):
        if x not in df.index:
            raise ValueError('{} is not in this VSM'.format(x))
    avec = df.loc[a]
    bvec = df.loc[b]
    cvec = df.loc[c]
    newvec = (bvec - avec) + cvec
    dists = df.apply(lambda row: distfunc(newvec, row), axis=1)
    dists = dists.drop([a,b,c])
    return pd.Series(dists).sort_values()

def analogy_evaluation(
        df, 
        src_filename='gram1-adjective-to-adverb.txt', 
        distfunc=vsm.cosine,
        verbose=True):
    """Basic analogies evaluation for a file `src_filename `
    in `question-data/`.
    
    Parameters
    ----------    
    df : pd.DataFrame
        The VSM being evaluated.
    src_filename : str
        Basename of the file to be evaluated. It's assumed to be in
        `analogies_home`.        
    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`, 
        `matching`, `jaccard`, as well as any other distance measure 
        between 1d vectors.
    
    Returns
    -------
    (float, float)
        The first is the mean reciprocal rank of the predictions and 
        the second is the accuracy of the predictions.
    
    """
    src_filename = os.path.join(analogies_home, src_filename)
    # Read in the data and restrict to problems we can solve:
    with open(src_filename) as f:    
        data = [line.split() for line in f.read().splitlines()]
    data = [prob for prob in data if set(prob) <= set(df.index)]
    # Run the evaluation, collecting accuracy and rankings:
    results = defaultdict(int)
    ranks = []
    for a, b, c, d in data:
        ranking = analogy_completion(a, b, c, df=df, distfunc=distfunc)       
        predicted = ranking.index[0]
        # Accuracy:
        results[predicted == d] += 1  
        # Rank of actual, starting at 1:
        rank = ranking.index.get_loc(d) + 1
        ranks.append(rank)        
        if verbose:
            print("{} is to {} as {} is to {} (gold: {} at rank {})".format(
                a, b, c, predicted, d, rank))        
    # Return the mean reciprocal rank and the accuracy results:
    mrr = np.mean(1.0 / (np.array(ranks)))
    return (mrr, results)