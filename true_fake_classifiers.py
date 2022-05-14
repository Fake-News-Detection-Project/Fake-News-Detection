import time
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import svm
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from NLPLib.DSP import CleanedDataset1


## Dataloading
dataset = CleanedDataset1()

#We fit on the train data ===================================================================================
startime = time.time()

Tfidf_vect = TfidfVectorizer(max_features=5000, use_idf=False)
Tfidf_vect.fit(tqdm(dataset.getSample(returnLabel=False), total=dataset.getLength()))

Train_X_Tfidf = Tfidf_vect.transform(tqdm(dataset.getSample(returnLabel=False), total=dataset.getLength()))

Train_Y = np.array([a for a in dataset.getLabel()])

print(Train_X_Tfidf.shape)


# We train the model =====================================================================================
SVM = svm.SVC()
SVM.fit(Train_X_Tfidf, Train_Y)

execTime = time.time() - startime
print(f"Model trained in {execTime}")

# We test it on the test set ==============================================================================
Test_X_Tfidf = Tfidf_vect.transform(dataset.getSample(training=False, returnLabel=False))
Test_Y = np.array([a for a in dataset.getLabel(training=False)])

predictions_NB = SVM.predict(Test_X_Tfidf)

print("SVM Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
