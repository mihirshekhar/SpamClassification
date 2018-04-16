
# Dataset Description
We have used two datasets from Enron complete dataset,  enron1 and enron2 to generate our results. 
The dataset can be downloaded from  http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html.  
enron1 : It contains 1500 soam email and 3672 non spam email. It is used to train and build our model.
5-fold Cross validation over this dataset is used to compare results.     
enron2: It contains 1496 spam emails with 4361 non spam emails. I have used this dataset to show the result of model trained over 
enron1 and is shown in the section Robustness results.   
enron-complete-dataset: It is a combination of all enron datasets, enron1 to enron6. We use this dataset, to effciency of  neural network approach in large data scenario. 

## NativeClassification.ipynb
This file shows experiment  with three algorithms : SVM, Naive Bayes and Gradient Boosted Classifier. 
SVM with best average f1 score of  0.981  is best performer followed by Naive Bayes  
with best average f1 score of 0.959.    
Gradient Boosted Classifier turned out to be  slow and hence results are not included.   

#### Naive Bayes (Results and Feature Selection)
The reason for chosing Naive Bayes is that   a large part of keyword terms in spam and not_spam are different). Also Naive Bayes provides a baseline to compare other algorithms performance.   
All the results are calculated using enron1 as training dataset. 
Experimented with several features. All the F1 scores are calculated using five fold crossvalidation over training data (enron1)  
1: Plain Naive Bayes : Average F1 score : 725   
followed by Stopword removal : Average F1 score   0.845  
followed by  Setting fit_prior True : Average F1 score 0.8599  
followed by Removing all words with count less than 2 : Average F1 score 0.9341  
followed by setting alpha parameter 0.05: Averge F1 score : 0.959  

This setting was used for comparing result of Naive Bayes with other algorithm. 

#### SVM 
High dimensional nature of this problem coupled  with effciency of SVM in such a scenario, led me to explore SVM. 
1: Plain SVM with stopword removal and removal of terms having count less than 2 :Average F1 score  0.951    
2: SVM with alpha tuning equal to 5e-4 :   Average F1 score 0.981    
    
We utlised various settings of optimum Naive Bayes, as features were already explored.   

#### Gradient Boosted Classifier
SVM and Naive Bayes represent discriminative and generative classiiers. I wanted to exlore the efficacy of decision trees in this scenario. Hence, used Gradient Boosted Classifier. Unfortuately, the performance of Gradient Boosted Tree was very slow owing to conversion from spare to dense format and hence its results are not included. 


### Observation  (Parameter Tuning )
Setting alpha parameter greater than 0.5 resulted in significant loss of f1 score for Naiver Bayes.  
Alpha values beyond 1e-3 and 1e-4 range yield very poor performance.  
Removing all words with count greater than 4 results was poor choice.
Stemming and tokenization using nltk in count vectorizer is slow and thus discarded from experiemnts.(although it coud have further increased the accuracy)     
Using Bigram and higher order of n-grams along with unigram was performing poorly  and slow as compared to unigrams only.   

### Robustness Results
Tested Naive Bayes and SVM model trained over enron1 dataset to generate output for enron2 dataset.  
F1 score  for SVM model was 0.991  
F1 score  for  Naive Baye model was 0.975  
This shows  that the  model is optimally trained (not overfit or underfit) and robust. 

## NeuralNetwork.ipynb
This file shows experiment with a deep neural network.    

#### Neural Network Architecture 
We used a 5 layer deep architecture, build with dense layers for training our model.
The number of epoch was set to 200 and learning rate was 1e-3.
We used the pretrained google word2vec (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit  ) model.
It includes word vectors for a vocabulary of 3 million words and phrases that they trained on roughly 100 billion words from a Google News dataset.  
First, we generate a document model for each file, taking average of word2vec word vectors.   
We train our neural network architecture on these document features.  

### Results
We experimented with three, four, five and ix layer   architecture to choose the most suitable architecture. 

|          | CrossValidation Results(enron1) | Robustness Results(enron2) |
|----------|---------------------------------|----------------------------|
| 3-layer  | 0.9565                          | 0.807                       |
| 4-layer  | 0.9644                          | 0.8108                     |
| 5-layer  | 0.9644                          | 0.833                     |  
| 6-layer  | 0.9602                          | 0.814                      |  

It can be observed that with 4-layer, 5 layer and 6th layer higher robustness results are approx same with 5 layer architecture showing best result. Scores  of three layered architecture suggests underfitting.   
Hence, I have used 5 layer architecture.

Comparing with other techniques, SVM and Naive Bayes, the score is low. The reason can be attributed to two factors:  
1) Low data : Deep neural Techniques require a large amount of data, as it learns all the features themselves. We experimented by changing the input data by repalcing enron1 by  a combination of all files in enron1 to enron6.     We obtained a cross validated f1-score of  0.9864 on this dataset.

2) Poor document vector: The sparsity of data in neural network is handled by utilising features, that correlates the relationship we are trying to model. We tried using doc2vec for document feature generation. However, I was unable to train the model on enron dataset due to relatively high time and memory constraint associated.
 

# Installation
All the the files are jupyter notebook files and needs to be opened in Jupyter notebook with listed packages installed.
Python version is 2.7
## NaiveClassification.ipynb
This file contains code for classifying email into spam or not using **SVM** and **Naive Bayes**  
**input_data_folder** : path to folder for training and crossvalidation results (with subdirectories as spam and ham containing files). 
**different_data_folder**= path to folder for testing robustness other than enron input_data_folder (with subdirectories as spam and ham containing files).   
For running the training and test, just open the file in jupyter notebook and  provide path to these two directories.   

### Prerequisites
wordcloud  
nltk  
numpy  
matplotlib  
sklearn  
pandas  

## NeuralNetwork.ipynb
This file contains code for classifying email into spam or not using Neural Network.  
**pretrained_emb**: Path to pretrained google word2vec embedding.     
**input_data_folder** : path to folder for training and crossvalidation results (with subdirectories as spam and ham containing files).   
**different_data_folder**= path to folder for testing robustness other than enron input_data_folder (with subdirectories as spam and ham containing files).    
**input_pkl** : path to folder for keeping pkl file of input_data_folder.   
**different_pkl** : path to folder for keeping pkl file of different_data_folder.   
For running the training and test, just open the file in jupyter notebook and  provide path to these two directories.     

### Prerequisites   
numpy  
matplotlib  
sklearn  
pandas  
gensim  
theano  
keras  
nltk  







