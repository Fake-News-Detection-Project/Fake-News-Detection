import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import ngrams

import re
from tqdm import tqdm

from NLPLib.DSP import DatasetInterface

class FeatureExtractor():
    def __init__(self, dataset:DatasetInterface, n=2, method='TF', feature_size=1000):
        self.n = n
        self.method = method
        self.feature_size = feature_size

        self.stemmer = porter.PorterStemmer()

        self.i2g = [] # index to gram : [ gram0, gram1, gram2, ]
        self.g2i = dict() # gram to index : { gram0:0, gram1:1, gram2:2, }
        self.gram_count = [] # count # of grams apparition indexwise [ # gram0, # gram1, # gram2, ]
        self.N = 0 # total number of != grams over all corpus
        self.total_grams = 0 # number of grams 
        
        self.dataset = dataset
    
    def featureTermFrequency(self):
         ## Data preprocessing
        clean_text, n_grams_text = [], []
        pattern = "[^a-zA-Z]" # Pattern for only keeping letters 
        stop_words = set( stopwords.words('english') )
    
        for (txt, label) in tqdm(self.dataset.getSample(), total=self.dataset.getLength(), desc="Processing corpus - Cleaning data"): 
            
            txt = re.sub(pattern," ",txt) # Remove punctuation / non letter characters
            txt = txt.lower() # case lowering
            txt = nltk.word_tokenize(txt) # Tokenizing
            txt = [ self.stemmer.stem(word) for word in txt if not word in stop_words ] # Stop words removal + stemming
            txt = ' '.join(txt)
            txt = list(pad_both_ends(txt.split(), n=self.n)) # padding for computing n-grams
            txt = list(ngrams(txt, n=self.n)) # computing n-grams

            for gram in txt: # Counting grams for all texts
                self.total_grams += 1
                if self.g2i.get(gram) == None: 
                    self.i2g.append( gram )
                    self.gram_count.append(0)
                    self.g2i[gram] = self.N
                    self.N  += 1
                self.gram_count[ self.g2i[gram] ] += 1

            clean_text.append(txt)

        # Adding a column to the dataset
        # self.corpus["grams"] = clean_text

        print( self.gram_count[np.argmax(self.gram_count)]  )
        selected_grams_idx = np.flip( np.argsort( self.gram_count ) )[:self.feature_size] # Extracting the most frequent grams
        print(selected_grams_idx)

        selected_grams = [ self.i2g[i] for i in selected_grams_idx ] # List of the most import 
        selected_grams_dict = dict( zip(selected_grams, [i for i in range(self.feature_size)]) )
        
        for gram in selected_grams:
            print(gram, self.gram_count[ self.g2i[gram] ])
        if self.method == 'TF':
            feature_vect = np.zeros( ( len(clean_text), self.feature_size) )
            
            for i, gram_list in enumerate( tqdm(clean_text, desc="Processing corpus - TF Feature extraction\n") ):
                for gram in gram_list:
                    if selected_grams_dict.get(gram) != None:
                        
                        feature_vect[i, selected_grams_dict.get(gram)] += 1
                # print(feature_vect[i])
                if np.sum(feature_vect[i])  != 0:
                    feature_vect[i] /= np.sum(feature_vect[i])
                else:
                    print("IS NAN")
        
            # feature = list(feature_vect)
    
        return feature_vect
    
    