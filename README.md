# Learning Word Embeddings from Dictionary
The goal of this project to build a word embedding model that learns vector representations of new words from their English dictionary definitions. We have collected more than 100k word definitions from an online dictionary, and trained a definition embeding model on the collected dataset. We mainly evaluate our model’s embedding on two intrinsic evaluation task: word similarity and word analogy. Furthermore, we test our model’s ability to embed new words given their dictionary definitions. For more detailed information, plese refer to the [report](https://github.com/donlee90/dictionary/blob/master/learning-word-embeddings-from-dictionary.pdf)

## Description
- **`crawler.py`**: Multithreaded crawler for downloading word definitions from `dictionary.com`
- **`models/`**: Contains pytorch models for dictionary embedding
- **`Project.ipynb`**: Contains code for training and evaluation of the model.

## Requirements
- `pandas`
- `BeautifulSoup`
- `pytorch`
- `progressbar`
- `scipy`
- `matplotlib`
- `sklearn`

## Notes
This was a course project for [CS224U: Natural Language Understanding](http://web.stanford.edu/class/cs224u/) at Stanford University. Many of the baselines and evaluation codes were borrowed from the course material.
