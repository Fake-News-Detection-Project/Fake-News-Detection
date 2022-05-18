import os
from typing import Tuple
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from tqdm import tqdm
import numpy as np
import json
import nltk

  

#Interface
class DatasetInterface:
    def getSample(self, training:bool) -> Tuple[str, int]:
        """Load a random sample from the dataset."""
        pass

    def getLength(self, training:bool) -> int:
        """Get the length of the dataset."""
        pass


class Dataset1(DatasetInterface):
    def __init__(self, true_filepath = "NLPLib/Datasets/Dataset1/True.csv", fake_filepath="NLPLib/Datasets/Dataset1/Fake.csv", train_size = 0.8, seed:int = None):
        true = pd.read_csv(true_filepath)
        true['label'] = np.ones(len(true), dtype=int)
        fake = pd.read_csv(fake_filepath)
        fake['label'] = np.zeros(len(fake), dtype=int)

        self.corpus = pd.concat((true,fake),axis=0)
        
        self.seed = np.random.randint(0, 10000) if seed == None else seed
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.corpus.text, self.corpus.label, train_size=train_size, random_state=seed)
        

    def getSample(self, training: bool = True, testing=False, returnLabel:bool = True) -> Tuple[np.ndarray, float]:

        if training:
            for i in range(self.getLength(True)):
                yield (self.x_train.iloc[i], self.y_train.iloc[i]) if returnLabel else self.x_train.iloc[i]

        if testing:
            for i in range(self.getLength(False)):
                yield (self.x_test.iloc[i], self.y_test.iloc[i]) if returnLabel else self.x_test.iloc[i]
    
    def getLabel(self, training:bool = True):
        for i in range(self.getLength(training)):
            if training:
                yield self.y_train.iloc[i]
            else:
                yield self.y_test.iloc[i]
        
                
    def getLength(self, training = True):
        return len(self.x_train) if training else len(self.x_test)
    


class CleanedDataset1(DatasetInterface):
    def __init__(self, file_path = "NLPLib/Datasets/CleanedDataset1/cleaned_corpus.csv", train_size = 0.8, seed:int = None):

        self.corpus = pd.read_csv(file_path)
        
        self.seed = np.random.randint(0, 10000) if seed == None else seed
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.corpus.final_text, self.corpus.label, train_size=train_size, random_state=seed)
        


    def getSample(self, training: bool = True, testing=False, returnLabel:bool = True) -> Tuple[np.ndarray, float]:

        if training:
            for i in range(self.getLength(True)):
                yield (self.x_train.iloc[i], self.y_train.iloc[i]) if returnLabel else self.x_train.iloc[i]

        if testing:
            for i in range(self.getLength(False)):
                yield (self.x_test.iloc[i], self.y_test.iloc[i]) if returnLabel else self.x_test.iloc[i]
        
    
    
    
    def getLabel(self, training:bool = True):
        for i in range(self.getLength(training)):
            if training:
                yield self.y_train.iloc[i]
            else:
                yield self.y_test.iloc[i]
        
                
    def getLength(self, training = True):
        return len(self.x_train) if training else len(self.x_test)
    
PADDING_WORD = '<PAD>'
UNKNOWN_WORD = '<UNK>'

class PadSequence:
    """
    A callable used to merge a list of samples to form a padded mini-batch of Tensor
    """
    def __call__(self, batch, pad_data=PADDING_WORD, max_len=500, pad_labels=0):
        batch_data, batch_labels = zip(*batch)
        # max_len = max(map(len, batch_labels))
        padded_data = [[b[i] if i < len(b) else pad_data for i in range(max_len)] for b in batch_data]
        # padded_labels = [[l[i] if i < len(l) else pad_labels for i in range(max_len)] for l in batch_labels]
        return padded_data, batch_labels
    
    
    

class RNNDataset(Dataset):
    def __init__(self, true_filepath = "NLPLib/Datasets/Dataset1/True.csv", fake_filepath="NLPLib/Datasets/Dataset1/Fake.csv", lenSequence=500, corpus_percent = 0.5):

        # self.corpus = pd.read_csv(file_path)
        
        true = pd.read_csv(true_filepath)
        true['label'] = np.ones(len(true), dtype=int)
        fake = pd.read_csv(fake_filepath)
        fake['label'] = np.zeros(len(fake), dtype=int)

        self.corpus = pd.concat((true,fake),axis=0)
        
        self.lenSequence = lenSequence
        
        self.corpus_percent = corpus_percent
        
        self.x, self.y = self.corpus.text.to_numpy(), self.corpus.label.to_numpy()
                
        shuffledIndices = np.arange(len(self.corpus))
        np.random.shuffle(shuffledIndices)

        self.x, self.y = self.x[shuffledIndices], self.y[shuffledIndices]
                

    def __getitem__(self, idx) -> Tuple[np.ndarray, float]:
        # sequence = json.loads(self.x[idx].replace("'", "\""))
        sequence = nltk.word_tokenize(self.x[idx])
        return sequence, self.y[idx]
        
                
    def __len__(self):
        return (int)(len(self.y) * self.corpus_percent)




#Testing the DSP (execute in NLPLib folder)
if __name__ =="__main__":
    data_repo = "../data"
    true_filename = "True.csv"
    fake_filename = "Fake.csv"

    dataset = Dataset1(true_filepath=os.path.join(data_repo, true_filename), fake_filepath=os.path.join(data_repo, fake_filename))
    nbTrue = 0
    nbFalse = 0
    for (text, label) in tqdm(dataset.getSample(), desc="Reading corpus", total=dataset.getLength()):
        if label == 0: nbFalse += 1
        if label == 1: nbTrue += 1
    
    print(f"{nbTrue=}")
    print(f"{nbFalse=}")

