import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn import svm
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import text
from nltk.stem import porter
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize

from NLPLib.DSP import CleanedDataset1, Dataset1

def analyse_ds(dataset:Dataset1, use_idf:bool=False, n_grams:int=1, additional_stop_words:set={}, max_features:int=5000):
    stop_words = text.ENGLISH_STOP_WORDS.union(additional_stop_words) # To add more stop words

    stemmer = porter.PorterStemmer()
    pattern = r"\w*[a-z]+"

    analyzer = CountVectorizer(token_pattern=pattern, ngram_range=(n_grams, n_grams), max_features=max_features,lowercase=True, stop_words=stop_words).build_analyzer()
    stem_analyzer = lambda doc: (stemmer.stem(w) for w in analyzer(doc))
    
    # pattern=r"(?u)\b\w\w+\b"
    CountVec = CountVectorizer(analyzer=stem_analyzer)
    
    X = CountVec.fit_transform(tqdm(dataset.getSample(returnLabel=False), total=dataset.getLength()))
    return CountVec.get_feature_names_out(), X

if __name__ == "__main__":
    np.random.seed(10)

    

    n_word = 10
    stop_word = {'reuters'}
    dataset = Dataset1()
    # features_names, X = analyse_ds(dataset, max_features=n_word, additional_stop_words=stop_word)

    # word_freq = X.toarray().sum(axis=0)

    # perm = np.argsort(word_freq)[-n_word:] # Selecting n_word most frequent words
    # word_freq = word_freq[perm]
    # features_names = features_names[perm]

    # print(features_names)
    # print(word_freq)

    # plt.bar(x=features_names, height=word_freq)
    # plt.show()

    # X_reduce = TruncatedSVD(n_components=5).fit_transform(normalize(X))
    # labels = np.array([i for i in dataset.getLabel()])

    # perm_labels = np.argsort(labels)
    # labels = labels[perm_labels]
    # X_reduce = X_reduce[perm_labels, :]

    # X_reduce_1 = X_reduce[np.where(labels)]
    # X_reduce_2 = X_reduce[:np.where(labels)[0][0]]

    # plt.scatter(X_reduce_1[:, 0], X_reduce_1[:, 1], marker='x')
    # plt.scatter(X_reduce_2[:, 0], X_reduce_2[:,  1], marker='o')
    # plt.show()
    

    textSizesTrue = []
    textSizesFake = []
    nbText = 0

    for (sample, label) in tqdm(dataset.getSample(training=True, testing=True), total=dataset.getLength()):
        sentence = sample.split()
        
        if label == 1: textSizesTrue.append(len(sentence))
        else: textSizesFake.append(len(sentence))
        

    textSizes = textSizesFake + textSizesTrue
    print(f"Average number of word: {np.mean(np.array(textSizes))} words")
    print(f"Variance of the number of word: {np.std(np.array(textSizes))}")
    print(f"Minimum  number of word: {np.min(np.array(textSizes))} words")
    print(f"Maximum number of word: {np.max(np.array(textSizes))} words")


    print(f"Number of sentences longer than 3000 words {np.sum(np.array(textSizes) > 3000)}")
    print(f"Number of sentences shorter than 5 words {np.sum(np.array(textSizes) <= 5)}")

    plt.title("Length of the True articles.")
    plt.hist(textSizesTrue, bins=200)
    plt.show()
    plt.title("Length of the Fake articles.")
    plt.hist(textSizesFake, bins=200)
    plt.show()

    

