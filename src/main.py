from email.policy import default
import torch
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.util import ngrams, pad_sequence
from nltk.lm.preprocessing import pad_both_ends
from nltk import ngrams

from tqdm import tqdm
import numpy as np
import re
from collections import defaultdict
# nlkt.download('stopwords')
from DSP import DataPrep


if __name__ == '__main__':
    ## Dataloading
    data_repo = "../data"
    true_filename = "True.csv"
    fake_filename = "Fake.csv"

    n = 1
    dataObject = DataPrep( true_filepath=os.path.join(data_repo, true_filename), fake_filepath=os.path.join(data_repo, fake_filename), n=n, method='TF', feature_size= 50)
    dataObject.preprocess()
   
    








