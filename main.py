import os

from NLPLib import Dataset1
from NLPLib import FeatureExtractor

if __name__ == '__main__':
    ## Dataloading
    data_repo = "data"
    true_filename = "True.csv"
    fake_filename = "Fake.csv"

    n = 1
    dataset = Dataset1( true_filepath=os.path.join(data_repo, true_filename), fake_filepath=os.path.join(data_repo, fake_filename), train_size=0.2)
    preprocessor = FeatureExtractor(dataset)
    
    features = preprocessor.featureTermFrequency()
    print(len(features))
    










