# SpamClassification

## NativeClassification.ipynb
This file shows experiment  with three algorithms : SVM, Naive Bayes and Gradient Boosted Classifier. 
SVM is best performer followed by Naive Bayes. Gradient Boosted Classifier turned out to be very slow and hence results not included.   

### Feature Selection
#### Naive Bayes
All the results are calculated using enron1 as training dataset. 
Experimented with several features. All the F1 scores are calculated using five fold crossvalidation over training data (enron1)  
1: Plain Naive Bayes : Average F1 score : 725   
followed by Stopword removal : Average F1 score   0.845  
followed by  Setting fit_prior True : Average F1 score 0.8599  
followed by Removing all words with count less than 2 : Average F1 score 0.9341  
followed by setting alpha parameter 0.05: Averge F1 score : 0.959  

Hence for naive bayes we used a combination of all these features.  
#### Failed Attempts 
Setting alpha parameter greater than 0.5 resulted in high loss of f1 score.   
Removing all words with count greater than 5 results  in huge drop in score.    
Tried stemming and tokenization using nltk in countvectorizer, but it was very slow and hence abandoned.  

#### SVM 
1: Plain SVM with stopword removal and removal of terms having count less than 2 :Average F1 score  0.951  
2: SVM with alpha tuning equal to 5e-4 :   Average F1 score 0.979  
Less than this values results in loss of score and higher values like 1e-3 results in significant loss in accuracy.   

### Robustness Results
I tested the model  of Naive Bayes and SVM model trained over enron1 dataset to generate output for enron5 dataset.  
F1 score  for SVM model was 0.986  
F1 score  for  Naive Baye model was 0.95  
















# Installation
All the the files are jupyter notebook files. 
## NaiveClassification.ipynb
This file contains code for classifying email into spam or not using **SVM** and **Naive Bayes**  
**input_data_folder** : path to folder for training (with subdirectories as spam and ham containing files).      
**different_data_folder**= path to folder for test (with subdirectories as spam and ham containing files).    
for running the training and test, just open the file in jupyter notebook and  provide path to these two directories.   

### Prerequisites
wordcloud  
nltk  
numpy  
matplotlib  
sklearn  
pandas  





