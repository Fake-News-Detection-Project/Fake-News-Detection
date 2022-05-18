from gpg import Data
from matplotlib import testing
from NLPLib.DSP import Dataset1
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

dataset = Dataset1()


textSizesTrue = []
textSizesFake = []
nbText = 0

for (sample, label) in tqdm(dataset.getSample(training=True, testing=True), total=dataset.getLength()):
    sentence = sample.split()
    
   
    if label == 1: textSizesTrue.append(len(sentence))
    else: textSizesFake.append(len(sentence))
    

textSizes = textSizesFake + textSizesTrue
print(f"Averegae number of word: {np.mean(np.array(textSizes))} words")
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