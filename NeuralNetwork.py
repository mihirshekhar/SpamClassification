
# coding: utf-8

# In[1]:


import gensim
import os
import re
from nltk.tokenize import RegexpTokenizer
import ntpath
import numpy as np
import cPickle as pickle
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, merge,Conv1D,Flatten,GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix, f1_score
get_ipython().magic(u'matplotlib inline')
#set keras backened to tensorflow 
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
#set_keras_backend("tensorflow")

#pretrained word embeddings path
pretrained_emb = "data/GoogleNews-vectors-negative300.bin"
#input_data_folder
input_data_folder = "data/Enron/enron1"
input_pkl = "enron1.pkl"
#different data folder
different_data_folder ="data/Enron/enron2"
#different_pkl
different_pkl = "enron2.pkl"



# In[2]:


#Load word2vec model in memory if pkl files are not created
#GoogleNews-vectors-negative300.bin is word2vec model trained over google news dataset
def load_word2vec():
    if(os.path.isfile(input_pkl)==False or os.path.isfile(different_pkl)==False ):
        gensim_model = gensim.models.KeyedVectors.load_word2vec_format(
            pretrained_emb, binary=True)
        return gensim_model
    print("loaded")


# In[3]:


#iterate through all filename present in input folder 
#spam label :1, ham label : 0
def read_file_names(input_folder_path):
     path = list()
     for root, directories, filenames in os.walk(input_folder_path):
               for filename in filenames: 
                       filepath = os.path.join(root,filename)  
                       if "spam" or "ham" in filename:
                                path.append(filepath)
     return path
                            
    #removes all non ascii characters
def remove_non_ascii(text):

    return re.sub(r'[^\x00-\x7F]',' ', text) 
 
#read file text from file path and removes all new line characters by whitesopace 
#No need for preprocessing as the data is already preprocessed
def read_file(file_path):
    file = open(file_path, 'r')
    text = file.read().strip()
    file.close()
    text = remove_non_ascii(text)
    return text.lower()
 


# In[4]:


#generates average document vectors as average word2vec vectors present in document.
#Checks created dictionary as pkl file for speedup in retesting
#Tried creating doc2vec model but is taking a lot of time
def create_doc_embedding(input_folder, output_pkl, gensim_model):
    X = dict()
    if(os.path.isfile(output_pkl)==False):
        count = 0
        file_list = read_file_names(input_folder)
        tokenizer = RegexpTokenizer(r'\w+')
        for file in file_list:
            count = count+1
            #print(count)
            f =  open(file, 'r')
            st  =f.read()
            name = ntpath.basename(f.name)
            doc_vec = np.zeros(gensim_model.vector_size) # scope for sys arg.
            word_count = 0
            words =tokenizer.tokenize(st)
            for word in words:
                word_vec = np.zeros(gensim_model.vector_size)
                if word in gensim_model.vocab:
                    word_vec = np.array(gensim_model[word])
                    word_count += 1
                    doc_vec = doc_vec + word_vec
            doc_vec = doc_vec / word_count
            X[name] = doc_vec
        pickle.dump(X, open(output_pkl, "wb" ))
    else :
        X = pickle.load(open(output_pkl, "rb"))
    return X
        


# In[5]:




#Creates numpy array of document vectors with numpy array of class labels
def create_numpy_array(hmap):
    x = list()
    y = list()
    for val in hmap:
        x.append(np.array(hmap[val]))
        if "spam" in val:
            y.append(1)
        else:
            y.append(0)
        # print(np.array(hmap[val]))
    return np.array(x), np.array(y)


# In[6]:


#generate both embeddings
def preprocess_data():
    gensim_model = load_word2vec()
    hash_embedding_input = create_doc_embedding(input_data_folder, input_pkl, gensim_model )
    print("Created embedding 1")
    hash_embedding_diff = create_doc_embedding(different_data_folder, different_pkl,gensim_model )
    print("Created embedding 2")

    return hash_embedding_input, hash_embedding_diff


# 
# if __name__ == "__main__":
#     
#     input_dir, different_dir = preprocess_data()
#     
#     #create trainig and test data in numpy format
#     X,Y = create_numpy_array(input_dir)
#     
#     # number of epochs
#     nb_epoch =200
#      #dimensionality of embedding
#     learning_rate = 0.0001
#     seed = 29
#     input_dim = 300
#     batch_size = 200
#     # network definition
#     kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#     scores = []
#     confusion = np.array([[0, 0], [0, 0]])
#     for train, test in kfold.split(X, Y):
#         model = Sequential()
#         model.add(Dense(128, input_shape=(input_dim,), activation='relu', name='dense_layer_1'))
#    	    #model.add(Dropout(0.5))
#         model.add(Dense(64, activation='relu',name='dense_layer_2'))
#         model.add(Dropout(0.25))
#         model.add(Dense(32, activation='relu',name='dense_layer_3'))
#         model.add(Dropout(0.5))
#         model.add(Dense(16, activation='relu', name='dense_layer_4'))
#         model.add(Dropout(0.25))
#         model.add(Dense(8, activation='relu', name='dense_layer_5'))
#         model.add(Dropout(0.25))
#         model.add(Dense(1, activation='sigmoid', name= 'decision_layer'))
#         model.summary()
# 
# # Compile model
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
#         history = model.fit([X[train]], Y[train],
#                     batch_size=batch_size, nb_epoch=nb_epoch,
#                     verbose=1, validation_data=([X[test]], Y[test]))
# # evaluate the model
#         predictions = model.predict(X[test])
#         pred_label =  np.rint(predictions)
#         confusion += confusion_matrix(Y[test], pred_label)
#         score = f1_score(Y[test], pred_label, pos_label=1)
#         scores.append(score)
# 
#         print(history.history.keys())
#     
# # summarize history for loss
#         plt.plot(history.history['loss'])
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'test'], loc='upper left')
#         plt.show()
#     print('Average F1 Score:', sum(scores)/len(scores))
#     print('Confusion matrix:')
#     print(confusion)
#     
#     
# 
# 
# 
# 

# In[14]:


#calculate  score for the test data
X_diff,Y_diff = create_numpy_array(different_dir)
predictions = model.predict([X_diff])
print(len(predictions))
pred_label =  np.rint(predictions)
confusion = confusion_matrix(Y_diff, pred_label)
score = f1_score(Y_diff, pred_label, pos_label=1)
print('Average F1 Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)


# In[ ]:




