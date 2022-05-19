import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn import svm, neighbors, ensemble

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import text
from nltk.stem import porter
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from NLPLib.DSP import CleanedDataset1, Dataset1

def train_model(dataset:Dataset1, classifier, use_idf:bool=False, n_grams:int=1, additional_stop_words:set={}, max_features:int=5000):
    #We fit on the train data
    startime = time.time()
    
    # Performing the preprocessing step on the dataset=
    # - Removing stop words
    # - Case lowering
    # - n-gram selection
    # - Selecting only char (no digits/punct)
    # - Stemming
    stop_words = text.ENGLISH_STOP_WORDS.union(additional_stop_words) # To add more stop words

    stemmer = porter.PorterStemmer()
    pattern = r"\w*[a-z]+"

    analyzer = CountVectorizer(token_pattern=pattern, ngram_range=(n_grams, n_grams), max_features=max_features,lowercase=True, stop_words=stop_words).build_analyzer()
    stem_analyzer = lambda doc: (stemmer.stem(w) for w in analyzer(doc))

    CountVec = CountVectorizer(analyzer=stem_analyzer)
    # X = CountVec.fit_transform(tqdm(dataset.getSample(returnLabel=False), total=dataset.getLength()))

    # Compute features
    TfidTransfo = TfidfTransformer(use_idf=use_idf)

    # Pipeline for computing data
    pipeline = Pipeline([('count', CountVec), 
                        ('tfidf', TfidTransfo),
                        ('clf', classifier)])

    
    # Running the pipeline
    pipeline.fit(dataset.x_train, dataset.y_train)
    # pipeline.fit(tqdm(dataset.getSample(returnLabel=False), total=dataset.getLength()))
    
    
    execTime = time.time() - startime
    print(f"Model trained in {execTime}")

    # # We test it on the test set
    # Test_X_Tfidf = Tfidf_vect.transform(dataset.getSample(training=False, returnLabel=False))
    
    predictions_NB = pipeline.predict(dataset.getSample(training=False, testing=True, returnLabel=False))
    
    return accuracy_score(predictions_NB, dataset.y_test)

## Dataloading
# dataset = CleanedDataset1()
dataset = Dataset1(train_size = 0.2)
features_size = [10,20]
stop_word = {'reuters', 'washington', 'seattle'}
classifiers_name = ["SVM", "5-NN", "RandomForest"]
classifiers = [svm.SVC(), neighbors.KNeighborsClassifier(), ensemble.RandomForestClassifier()]

for i, classifier in enumerate(classifiers):
    print("Classifier:", classifiers_name[i])
    for feature in features_size:
        for n in range(1, 5):
            print(f"\tFeature size: {feature}, N-gram size: {n}, TF")
            accuracy_score = train_model(dataset, classifier=classifier, max_features=feature, use_idf=False, n_grams=n, additional_stop_words=stop_word)
            print("\tClassifier Accuracy Score -> ", accuracy_score*100)

            print(f"\tFeature size: {feature}, N-gram size: {n}, TF-IDF")
            accuracy_score = train_model(dataset, classifier=classifier, max_features=feature, use_idf=True, n_grams=n, additional_stop_words=stop_word)
            print("\tClassifier Accuracy Score -> ", accuracy_score*100)
