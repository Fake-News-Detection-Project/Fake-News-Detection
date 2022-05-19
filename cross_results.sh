python true_fake_classifiers.py -l models/svm_feature_10_year_2016_gram_1_useIDF_False_part_size_0.8 -y 2015 > results/svm_10_tr_2016_test_2015.txt
python true_fake_classifiers.py -l models/svm_feature_10_year_2016_gram_1_useIDF_False_part_size_0.8 -y 2017 > results/svm_10_tr_2016_test_2017.txt
python true_fake_classifiers.py -l models/svm_feature_10_year_2016_gram_1_useIDF_False_part_size_0.8 > results/svm_10_tr_2016_test_ALL.txt
python true_fake_classifiers.py -l models/svm_feature_10_year_2017_gram_1_useIDF_False_part_size_0.8 -y 2015 > results/svm_10_tr_2017_test_2015.txt
python true_fake_classifiers.py -l models/svm_feature_10_year_2017_gram_1_useIDF_False_part_size_0.8 -y 2016 > results/svm_10_tr_2017_test_2016.txt
python true_fake_classifiers.py -l models/svm_feature_10_year_2017_gram_1_useIDF_False_part_size_0.8 > results/svm_10_tr_2017_test_ALL.txt