# Fake News Detection: Comparison of state of the art methods

Fact-checking is an important part of our modern society. However, an unlimited amount of news are produced each day, and one cannot rely on human verification to classify a news as fake or not. Therefore, fake news can spread really easily and in an uncontrolled manner before beiing checked. This can have a quantitative impact on public opinions [1].  
The purpose of our project is to reproduce the result and performance of some (?) valuable papers on fake news detection using Machine Learning and Natural Language Processing.

## Authors

[Corentin Royer](https://github.com/corentin-ryr)  
[Baptiste Engel](https://github.com/engelba)

## Methodology

As we want initialy to reproduce the results in [2], we will follow the same data preprocesing methodology.

### Dataprocessing

-   Remove punctuation / non letter characters
-   case lowering
-   Stop word removal
-   tokenization
-   stemming (Porter stemmer in the original paper) > Can use different stemmers
-   n-gram word based tokenizer

### Feature comparison

The models are tested with two features:

-   Term frequency
-   Term Frequency-Inverted Document Frequency

### Classifiers

Comparison of 6 simples ML algorithms in [2]: SGD, SVM, LSVM, KNN, Decision Trees.  
5-fold CV is used with 80% of the data for training and 20% for validation.

To run the RNN, execute the file "ptru_fake_rnn.py" with the arguments:

-   --embeddings FILE_WITH_GLOVE_MATRIX
-   --batch_size BATCH_SIZE
-   --epochs NB_EPOCHS
-   --learning_rate VALUE_LEARNING_RATE

We recommend the values:

"python true_fake_rnn.py --embeddings glove.6B.50d.txt --epochs 3 --learning_rate 0.05"

## References

[1] Olan, F., Jayawickrama, U., Arakpogun, E.O. et al. Fake news on Social Media: the Impact on Society. Inf Syst Front (2022). https://doi.org/10.1007/s10796-022-10242-z

[2] H. Ahmed, I. Traore, S. Saad. Detection of Online Fake News Using N-Gram
Analysis and Machine Learning Techniques (2017). DOI: 10.1007/978-3-319-69155-8_9

## Worth to read papers

[Fake News Detection Using Machine Learning Approaches](https://iopscience.iop.org/article/10.1088/1757-899X/1099/1/012040/pdf)  
Method: Input the dataset in a lot of different classifier and average the output. #----

[A Survey on Natural Language Processing for Fake News Detection](https://arxiv.org/pdf/1811.00770.pdf)  
Really interesting overview of the subject, with description of several datasets, methods, ... Probably one of the most intersting paper to base our study on (there are a lot of interesting techniques cited). ####-

Detection of Online Fake News Using N-Gram
Analysis and Machine Learning Techniques  
Original paper for the dataset. The first goal is to reproduce their result.
