import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn import svm, neighbors, ensemble
from terminaltables import AsciiTable
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import text
from nltk.stem import porter
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from NLPLib.DSP import CleanedDataset1, Dataset1

# import pickle
import dill as pickle

import argparse

def train_model(dataset:Dataset1, classifier, use_idf:bool=False, n_grams:int=1, additional_stop_words:set={}, max_features:int=5000):
    #We fit on the train data
    startime = time.time()
    
    # Performing the preprocessing step on the dataset
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

    # Compute features
    TfidTransfo = TfidfTransformer(use_idf=use_idf)

    # Pipeline for computing data
    pipeline = Pipeline([('count', CountVec), 
                        ('tfidf', TfidTransfo),
                        ('clf', classifier)])

    
    # Running the pipeline
    pipeline.fit(dataset.x_train, dataset.y_train)
    
    execTime = time.time() - startime
    print(f"Model trained in {execTime} sec")

    # predictions_NB = pipeline.predict(tqdm(dataset.getSample(training=False, testing=True, returnLabel=False)))
    
    return pipeline #, accuracy_score(predictions_NB, dataset.y_test)

if __name__ == '__main__':

    #Args =============================================
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-clf', '--classifier', default='svm', help='Choose a classifier: svm (Support Vector Machine), knn (K-Nearest Neighbors) or rf (Random Forest)')
    parser.add_argument('-l', '--load', type=str, help='Load a model from a pickle object.')
    parser.add_argument('-y', '--year', type=str, default=None, help='Year to train the model on.')
    parser.add_argument('-fs', '--feature-size', type=int, default=10, help='Size of the feature')
    parser.add_argument('-s', '--save', action='store_true', help='Save model.')
    parser.add_argument('-idf', action='store_true', help='Use TF-IDF.')
    parser.add_argument('-gram', default=1, help='Size of n-gram used.')
    parser.add_argument('-part', '--partition', type=float, default=0.8, help='Partition size for training.')
    
    args = parser.parse_args()

    # Checking Args validity
    if args.year not in [None, 'UKN', '2015', '2016', '2017']:
        print(f"Year {args.year} is not valid.")
        raise SystemExit(0)
    elif not ( args.partition > 0 and args.partition < 1 ):
        print(f"Partition {args.partition} should be in (0,1).")
        raise SystemExit(0)
    elif args.feature_size <= 0:
        print(f"Feature size {args.feature_size} should be a positive integer.")
        raise SystemExit(0)
    else:
        # Loading data
        dataset = Dataset1(train_size = args.partition, year=args.year)

        if args.load is not None:
            print("Loading model from", args.load)
            try:
                model = pickle.load(open(args.load, 'rb'))
            except FileNotFoundError:
                print("Unable to load model", args.load)
                raise SystemExit(0)
        else:
            if args.classifier == 'svm':
                print("Training SVM")
                classifier = svm.SVC()
            elif args.classifier == 'knn':
                print("Training KNN")
                classifier = neighbors.KNeighborsClassifier()
            elif args.classifier == 'rf':
                print("Training Random Forest")
                classifier = ensemble.RandomForestClassifier()
            else:
                print(f"Unable to find specified classifier {args.classifier}\nTry with another value for flag -clf.")
                raise SystemExit(0)
            
            model = train_model(dataset,
                                    classifier=classifier, 
                                    max_features=args.feature_size, 
                                    use_idf=args.idf, 
                                    n_grams=args.gram
                                )

            if args.save:
                print("Saving model (Can take a long time)")
                filename = f'./models/{args.classifier}_feature_{args.feature_size}_year_{args.year}_gram_{args.gram}_useIDF_{args.idf}_part_size_{args.partition}'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)


        confusion_matrix = [[0, 0],
                            [0, 0]]

        pred_nb = model.predict(dataset.getSample(training=False, testing=True, returnLabel=False))
        true_labels = [lab for lab in dataset.getLabel(training=False)]

        for true_label, pred in zip(true_labels, pred_nb):
            confusion_matrix[true_label][pred] += 1


        table = [['', 'Predicted Fake', 'Predicted True'],
                ['Real Fake', confusion_matrix[0][0], confusion_matrix[0][1]],
                ['Real True', confusion_matrix[1][0], confusion_matrix[1][1]]]

        t = AsciiTable(table)
        print(t.table)
        print("Test accuracy: {}".format(
            round((confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix), 4))
        )