# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:15:52 2022

@author: akhan
"""

#Install required Packages

# pip install tweepy
# #pip install pytorch
# pip install streamlit
# pip install torchtext
# pip install matplotlib
# pip install scikit-learn


#IMPORT ALL REQUIRED PACKAGES

import streamlit as st

import os
import pandas as pd
import numpy as np 
import pickle
from datetime import datetime, timedelta


import tweepy
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from collections import Counter
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import torchtext
from torchtext.data import get_tokenizer

from sklearn.utils import shuffle
#from sklearn.metrics import classification_report
#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import requests, zipfile, io

import re
import nltk
from nltk.tokenize import TweetTokenizer
#from nltk.stem.porter import PorterStemmer
#from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download('stopwords')
#nltk.download('wordnet')
import string

from ast import literal_eval
import gc

from io import BytesIO
#import the functions from other files

#from fetchTweets import fetchTweets
#from preProcessing import *
#from sarcasmEval import sarcasmEval
#from plottingSummaries import plottingSummaries

###############Set Device#####################

# Set the device to use GPU
def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("WARNING: if possible, try to use `GPU`")
  else:
    print("GPU is enabled in this notebook.")

  return device

# Set the device (check if gpu is available)
device = set_device()

##########################################

#initialize variables and load models that will be used later on

numTweet = 1000
batch_size = numTweet
glove_dim = 100
dimension = glove_dim

#load embedding

with open("C:/Users/akhan/pythonProjects/workingSarcasmApp2/Models/glove_embedding.pkl", 'rb') as f:
    glove_embedding = pickle.load(f)




##################################################################

# #Define Model Hyperparameters

model_name = 'LSTM'                 #>>>>> What kind of model? [Only LSTM or GRU possible]

no_layers = 2                       #>>>>> How many layers in the RNN?

hidden_dim = 32                   #>>>>> What dimension of the hidden layer?

output_dim = 1

vocab_size = 30000                  #>>>>> Vocabulary size?

embedding_dim = 100                  #>>>>> Embedding dimension?

#embedding_matrix = embedding_matrix #>>>>> What embedding matrix to use? 
# [Set to None if you do not wish to use pre-trained network]

bi_directionality = True           #>>>>> Do you want to use bidirectional RNN? 
# [Default is False]

use_pre_trained = True             #>>>>> Should the model use the pre-trained embedding weights?

tune_pre_trained = True             #>>>>> Should the model tune the pre-trained embedding weights?
# [Default is False]

drop_prob=0.1                       #>>>>> What dropout probability to use, default is 0.1

epochs = 1                          #>>>>> No. of epochs

clip = 5                            #>>>>> Clippping the gradients

lr = 0.001                          #>>>>> Learning rate

criterion = nn.BCELoss()            #>>>>> Criterion for calculating the loss

out_dim = 1

batch_size = numTweet

##########################################################

# Define the model classes:
    
#Define the Sarcasm Model Class

class SarcasmRNN(nn.Module):
    def __init__(self,no_layers,hidden_dim, output_dim, vocab_size, embedding_dim, embedding_matrix,
                 model_name = 'LSTM', bi_directionality = True, use_pre_trained = True, 
                 tune_pre_trained = True, drop_prob=0.1):
        super(SarcasmRNN,self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.drop_prob = drop_prob
        self.model_name = model_name
        self.directions = bi_directionality
        self.use_pre_trained = use_pre_trained
        self.tune_pre_trained = tune_pre_trained
        if use_pre_trained:
            self.vocab_size = embedding_matrix.shape[0]
            self.embedding_dim = embedding_matrix.shape[1]
            # Embedding Layer
            self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
            self.embedding.weight.requires_grad = self.tune_pre_trained
        else:
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            # Embedding Layer
            self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
    
    # Next layers based ont he model chosen
        if self.model_name == 'LSTM':
      # LSTM Layers
            self.lstm = nn.LSTM(input_size = self.embedding_dim,
                              hidden_size = self.hidden_dim,
                              num_layers = no_layers,
                              bidirectional = self.directions,
                              batch_first = True, 
                              dropout = self.drop_prob)
        else:
            self.gru = nn.GRU(input_size = self.embedding_dim,
                            hidden_size = self.hidden_dim,
                            num_layers = no_layers,
                            bidirectional = self.directions,
                            batch_first = True, 
                            dropout = self.drop_prob)
      
    
    # Dropout layer
        self.dropout = nn.Dropout(drop_prob)

    # Linear and Sigmoid layer
        if self.directions:
            self.fc = nn.Linear(self.hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(self.hidden_dim * 2, output_dim)
    
        self.sig = nn.Sigmoid()
      
    def forward(self,x,hidden):
        if self.model_name == 'LSTM':
            batch_size = x.size(0)
            embeds = self.embedding(x)
            #Shape: [batch_size x max_length x embedding_dim]
            # LSTM out
            lstm_out, hidden = self.lstm(embeds, hidden)
            # Shape: [batch_size x max_length x hidden_dim]
            # Select the activation of the last Hidden Layer
            lstm_out = lstm_out[:,-1,:].contiguous()
            # Shape: [batch_size x hidden_dim]
            ## You can instead average the activations across all the times
            # lstm_out = torch.mean(lstm_out, 1).contiguous()
            # Dropout and Fully connected layer
            out = self.dropout(lstm_out)
            out = self.fc(out)
            # Sigmoid function
            sig_out = self.sig(out)
            # return last sigmoid output and hidden state
            return sig_out, hidden
        else:
            batch_size = x.size(0)
            self.h = self.init_hidden(batch_size)
            # Embedding out
            embeds = self.embedding(x)
            #Shape: [batch_size x max_length x embedding_dim]
            # GRU out
            gru_out, self.h = self.gru(embeds, self.h)
            # Shape: [batch_size x max_length x hidden_dim]
            # Select the activation of the last Hidden Layer
            gru_out = gru_out[:,-1,:].contiguous()
            # Shape: [batch_size x hidden_dim]
            # Dropout and Fully connected layer
            out = self.dropout(gru_out)
            out = self.fc(out)
            # Sigmoid function
            sig_out = self.sig(out)
            # return last sigmoid output and hidden state
            return sig_out

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        if self.directions:
            directionality = 2
        else:
            directionality = 1
    
        if self.model_name == 'LSTM':
            # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
            # initialized to zero, for hidden state and cell state of LSTM
            h0 = torch.zeros((self.no_layers * directionality, batch_size, self.hidden_dim)).to(device)
            c0 = torch.zeros((self.no_layers * directionality, batch_size, self.hidden_dim)).to(device)

            hidden = (h0,c0)
            return hidden
        else:
            hidden = (torch.zeros((self.no_layers * directionality, batch_size, self.hidden_dim)).to(device))
            return hidden


#Define the Sent Model Class

class SentimentRNN(nn.Module):
    def __init__(self,no_layers,hidden_dim, output_dim, vocab_size, embedding_dim, embedding_matrix,
                 model_name = 'LSTM', bi_directionality = True, use_pre_trained = True, 
                 tune_pre_trained = True, drop_prob=0.1):
        super(SentimentRNN,self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.drop_prob = drop_prob
        self.model_name = model_name
        self.directions = bi_directionality
        self.use_pre_trained = use_pre_trained
        self.tune_pre_trained = tune_pre_trained
        if use_pre_trained:
            self.vocab_size = embedding_matrix.shape[0]
            self.embedding_dim = embedding_matrix.shape[1]
            # Embedding Layer
            self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
            self.embedding.weight.requires_grad = self.tune_pre_trained
        else:
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            # Embedding Layer
            self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
    
    # Next layers based ont he model chosen
        if self.model_name == 'LSTM':
      # LSTM Layers
            self.lstm = nn.LSTM(input_size = self.embedding_dim,
                              hidden_size = self.hidden_dim,
                              num_layers = no_layers,
                              bidirectional = self.directions,
                              batch_first = True, 
                              dropout = self.drop_prob)
        else:
            self.gru = nn.GRU(input_size = self.embedding_dim,
                            hidden_size = self.hidden_dim,
                            num_layers = no_layers,
                            bidirectional = self.directions,
                            batch_first = True, 
                            dropout = self.drop_prob)
      
    
    # Dropout layer
        self.dropout = nn.Dropout(drop_prob)

    # Linear and Sigmoid layer
        if self.directions:
            self.fc = nn.Linear(self.hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(self.hidden_dim * 2, output_dim)
    
        self.sig = nn.Sigmoid()
      
    def forward(self,x,hidden):
        if self.model_name == 'LSTM':
            batch_size = x.size(0)
            embeds = self.embedding(x)
            #Shape: [batch_size x max_length x embedding_dim]
            # LSTM out
            lstm_out, hidden = self.lstm(embeds, hidden)
            # Shape: [batch_size x max_length x hidden_dim]
            # Select the activation of the last Hidden Layer
            lstm_out = lstm_out[:,-1,:].contiguous()
            # Shape: [batch_size x hidden_dim]
            ## You can instead average the activations across all the times
            # lstm_out = torch.mean(lstm_out, 1).contiguous()
            # Dropout and Fully connected layer
            out = self.dropout(lstm_out)
            out = self.fc(out)
            # Sigmoid function
            sig_out = self.sig(out)
            # return last sigmoid output and hidden state
            return sig_out, hidden
        else:
            batch_size = x.size(0)
            self.h = self.init_hidden(batch_size)
            # Embedding out
            embeds = self.embedding(x)
            #Shape: [batch_size x max_length x embedding_dim]
            # GRU out
            gru_out, self.h = self.gru(embeds, self.h)
            # Shape: [batch_size x max_length x hidden_dim]
            # Select the activation of the last Hidden Layer
            gru_out = gru_out[:,-1,:].contiguous()
            # Shape: [batch_size x hidden_dim]
            # Dropout and Fully connected layer
            out = self.dropout(gru_out)
            out = self.fc(out)
            # Sigmoid function
            sig_out = self.sig(out)
            # return last sigmoid output and hidden state
            return sig_out

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        if self.directions:
            directionality = 2
        else:
            directionality = 1
    
        if self.model_name == 'LSTM':
            # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
            # initialized to zero, for hidden state and cell state of LSTM
            h0 = torch.zeros((self.no_layers * directionality, batch_size, self.hidden_dim)).to(device)
            c0 = torch.zeros((self.no_layers * directionality, batch_size, self.hidden_dim)).to(device)

            hidden = (h0,c0)
            return hidden
        else:
            hidden = (torch.zeros((self.no_layers * directionality, batch_size, self.hidden_dim)).to(device))
            return hidden

##################################################################

#Load the Pickled Models 

#LOADING THE TRAINED/PICKLED MODELS 
#Think about the paths for these files...
sentModel = torch.load('C:/Users/akhan/pythonProjects/workingSarcasmApp2/Models/sentRNN.pt', map_location=torch.device('cpu'))
sarcasmModel = torch.load('C:/Users/akhan/pythonProjects/workingSarcasmApp2/Models/sarcasmRNN.pt', map_location=torch.device('cpu'))


##################################################################

##Formatting the Streamlit Page#

st.title("The ~ Best ~ Saracasm Detector Ever...")

st.subheader("""
         Perform sarcasm (and sentiment) analysis on a random subset of tweets about a topic.
         """)
         
searchTerm = st.sidebar.text_input('Search Term', key = "searchTerm")
st.sidebar.write('The current search term is:')
st.sidebar.write("'", searchTerm, "'")


#The current twitter dev credentials only allow searching for tweets within the last two weeks. 
twoWeeksAgo = datetime.today() - timedelta(days = 13)
minDate = twoWeeksAgo.strftime("%Y-%m-%d")

date_since = st.sidebar.date_input(
    'Analyze tweets collected since the following day:', 
    key = "date_input", 
    min_value= datetime.today() - timedelta(days = 13),
    max_value = datetime.today()
    )

st.sidebar.write('Returning results since', date_since)         



#searchTerm = st.session_state.searchTerm
#date_since = st.session_state.date_input


#Fetch a random subsample of tweets returned from the twitter API
#Need to find a way to hide secret keys

####### Twitter Auth #############

consumer_key = os.environ.get('consumer_key')
consumer_secret = os.environ.get("consumer_secret")
access_key = os.environ.get("access_key")
access_secret = os.environ.get("access_secret")



# consumer_key = st.secrets['consumer_key']
# consumer_secret = st.secrets["consumer_secret"]
# access_key = st.secrets["access_key"]
# access_secret = st.secrets["access_secret"]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


########### Fetch Tweets #########

def scrape(searchTerm, date_since, numtweet):
    #HERE IS WHERE YOU FILTER TWEETS. SET DEFAULT NUM TWEET******
        # Creating DataFrame using pandas
      df1 = pd.DataFrame(columns=['username',
                              'description',
                              'location',
                              'following',
                              'followers',
                              'totaltweets',
                              'retweetcount',
                              'text',
                              'hashtags'])
  
      # We are using .Cursor() to search
      # through twitter for the required tweets.
      # The number of tweets can be
      # restricted using .items(number of tweets)
      tweets = tweepy.Cursor(api.search_tweets,
                          searchTerm, lang="en",
                          since_id=date_since,
                          tweet_mode='extended').items(numtweet)
  
  
      # .Cursor() returns an iterable object. Each item in
      # the iterator has various attributes
      # that you can access to
      # get information about each tweet
      list_tweets = [tweet for tweet in tweets]
  
      # Counter to maintain Tweet Count
      i = 1
  
      # we will iterate over each tweet in the
      # list for extracting information about each tweet
      for tweet in list_tweets:
              username = tweet.user.screen_name
              description = tweet.user.description
              location = tweet.user.location
              following = tweet.user.friends_count
              followers = tweet.user.followers_count
              totaltweets = tweet.user.statuses_count
              retweetcount = tweet.retweet_count
              hashtags = tweet.entities['hashtags']
  
              # Retweets can be distinguished by
              # a retweeted_status attribute,
              # in case it is an invalid reference,
              # except block will be executed
              try:
                      text = tweet.retweeted_status.full_text
              except AttributeError:
                      text = tweet.full_text
              hashtext = list()
              for j in range(0, len(hashtags)):
                      hashtext.append(hashtags[j]['text'])
  
              # Here we are appending all the
              # extracted information in the DataFrame
              ith_tweet = [username, description,
                          location, following,
                          followers, totaltweets,
                          retweetcount, text, hashtext]
              df1.loc[len(df1)] = ith_tweet
  
              # Function call to print tweet data on screen
            #MODIFY THIS TO PRINT A RANDOM SUBSET OF TWEETS MAYBE?
              #printtweetdata(i, ith_tweet)
              i = i+1
      #If you need to save the df...
      #filename = '/content/drive/MyDrive/Sarcasm Detection/scraped_tweets_'+searchTerm+'.csv'
  
      # we will save our database as a CSV file.
      #df1.to_csv('C:/Users/akhan/pythonProjects/workingSarcasmApp2/dfTest.csv')
      return df1

df1 = scrape(searchTerm, date_since, numTweet)
#st.write(df1.head())


############### PreProcess Returned Tweets ###############

#from preProcessing.py import preProcessing

def preProcessing(df1):
    #THERE ARE MULTIPLE PREPROCESSING STEPS BEFORE INPUTTING INTO THE MODEL. 
    #remove URLs
    pattern=r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))';
    def rem_url(x):
      match = re.findall(pattern, x)
      for m in match:
        url = m[0]
        x = x.replace(url, '')
      return x
    df1['url_removed'] = df1['text'].apply(lambda x:rem_url(x))
    # df1.head()
    #print('URL removed')
    
    ######################################################################################
    #remove twitter handles from text
    tknzr = TweetTokenizer(strip_handles=True)
    df1['msg_tokenied'] = df1['url_removed'].apply(lambda x:tknzr.tokenize(x))
    #df1['handle_removed'] = df1['comment'].apply(lambda x:tknzr.tokenize(x))
    # df1.head()
    #print('Twitter Handles removed')
    # detokenize the tweet again
    # from nltk.tokenize.treebank import TreebankWordDetokenizer
    # df1['orig'] = df1['handle_removed'].apply(lambda x:TreebankWordDetokenizer().detokenize(x))
    # print('Tweets detokenized again')
    
    ######################################################################################
    # remove punctuations
    # string.punctuation
    # def remove_punctuation(text):
    #     punctuationfree="".join([i for i in text if i not in string.punctuation])
    #     return punctuationfree
    # df1['clean_msg']= df1['orig'].apply(lambda x:remove_punctuation(x))
    # # df1.head()
    # print('Punctuation removed')
    
    ######################################################################################
    # delete the unnecessary columns
    # del df1['url_removed']
    #del df1['handle_removed']
    #del df1['orig']
    
    ######################################################################################
    # remove the upper case letters
    # df1['msg_lower'] = df1['clean_msg'].apply(lambda x: x.lower())
    # print('All set to lower case')
    
    ######################################################################################
    # tokenize
    # def tokenization(text):
    #     tokens = re.split('W+',text)
    #     return tokens
    # #applying function to the column
    # tknzr = TweetTokenizer(strip_handles=True)
    # #df1['msg_tokenied'] = df1['msg_lower'].apply(lambda x:tknzr.tokenize(x))
    # df1['msg_tokenied'] = df1['orig'].apply(lambda x:tknzr.tokenize(x))
    # print('Tweet tokenized again')
    
    ######################################################################################
    # removing stop words
    #Stop words present in the library
    # stopwords = nltk.corpus.stopwords.words('english')
    #defining the function to remove stopwords from tokenized text
    # def remove_stopwords(text):
    #     output= [i for i in text if i not in stopwords]
    #     return output
    # #applying the function
    # df1['no_stopwords']= df1['msg_tokenied'].apply(lambda x:remove_stopwords(x))
    # print('Stopwords removed')
    
    ######################################################################################
    #importing the Stemming function from nltk library
    #defining the object for stemming
    # porter_stemmer = PorterStemmer()
    # #defining a function for stemming
    # def stemming(text):
    #   stem_text = [porter_stemmer.stem(word) for word in text]
    #   return stem_text
    # df1['msg_stemmed']=df1['no_stopwords'].apply(lambda x: stemming(x))
    # print('Tweets stemmed')
    
    ######################################################################################
    #defining the object for Lemmatization
    # wordnet_lemmatizer = WordNetLemmatizer()
    # # nltk.download('wordnet')
    # #defining the function for lemmatization
    # def lemmatizer(text):
    #   lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    #   return lemm_text
    # df1['msg_lemmatized']=df1['no_stopwords'].apply(lambda x:lemmatizer(x))
    # print('Tweets Lemmatized')
    
    ######################################################################################

    return df1

df2 = preProcessing(df1)
#st.write(df2.head())



########################## Embedding ###############################

def embedding(df2):
    scrapedTweetToken = df2.msg_tokenied.values

    words = Counter()
    for s in scrapedTweetToken:
        for w in s:
            words[w] += 1
    
    #sort the words
    sorted_words = list(words.keys())
    sorted_words.sort(key=lambda w: words[w], reverse=True)
    
    
    # Identify the most common words to use for anaylsis
    #Let's select only the most used.
    num_words_dict = 30000
    # We reserve two numbers for special tokens.
    most_used_words = sorted_words[:num_words_dict-2]
    
    # We will add two extra Tokens to the dictionary, one for words outside the dictionary (`'UNK'`) and 
    # one for padding the sequences (`'PAD'`).
    
    # dictionary to go from words to idx 
    word_to_idx = {}
    # dictionary to go from idx to words (just in case) 
    idx_to_word = {}
    
    # We include the special tokens first
    PAD_token = 0   
    UNK_token = 1
    
    word_to_idx['PAD'] = PAD_token
    word_to_idx['UNK'] = UNK_token
    
    idx_to_word[PAD_token] = 'PAD'
    idx_to_word[UNK_token] = 'UNK'
    
    # We popullate our dictionaries with the most used words
    for num,word in enumerate(most_used_words):
      word_to_idx[word] = num + 2
      idx_to_word[num+2] = word
        
    # The following function helps create a word embedding matrix    
    #MAY NEED TO DO SOMETHING ABOUT THE INPUT TO CREATE_EMBEDDING MATRIX. DEFINE DIMENSION AS GLOVE_DIM?
    def create_embedding_matrix(word_index,embedding_dict,dimension):
      embedding_matrix=np.zeros((len(word_index)+1,dimension))
     
      for word,index in word_index.items():
        if word in embedding_dict:
          embedding_matrix[index]=embedding_dict[word]
      return embedding_matrix
    
    # Let us use the function to create a word-embedding matrix 
    # My word_to_idx is the same as their word_index
    embedding_matrix=create_embedding_matrix(word_to_idx,embedding_dict=glove_embedding,dimension=glove_dim)
    
    
    # Our goal now is to transform each tweet from a sequence of tokens to a sequence of indexes. 
    # These sequences of indexes will be the input to our pytorch model.
    
    # A function to convert list of tokens to list of indexes
    def tokens_to_idx(sentences_tokens,word_to_idx):
      sentences_idx = []
      for sent in sentences_tokens:
        sent_idx = []
        for word in sent:
          if word in word_to_idx:
            sent_idx.append(word_to_idx[word])
          else:
            sent_idx.append(word_to_idx['UNK'])
        sentences_idx.append(sent_idx)
      return sentences_idx
    
    scrapedTweetTokenIDX = tokens_to_idx(scrapedTweetToken,word_to_idx)
    #x_test_idx = tokens_to_idx(x_test_token,word_to_idx)
    
    
    
    # We need all the sequences to have the same length. 
    # To select an adequate sequence length, let's explore some statistics about the length of the tweets:
    tweet_lens = np.asarray([len(sentence) for sentence in scrapedTweetTokenIDX])
    
    #Format sample of tweets from twitter API as valid_data

    # We cut the sequences which are larger than our chosen maximum length (`max_length`) and fill with zeros the ones that are shorter.
    # We choose the max length
    max_length = 75 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Change this in case you want to use inputs of different lengths.
    
    # A function to make all the sequence have the same lenght
    # Note that the output is a Numpy matrix
    def padding(sentences, seq_len):
     features = np.zeros((len(sentences), seq_len),dtype=int)
     for ii, tweet in enumerate(sentences):
       len_tweet = len(tweet) 
       if len_tweet != 0:
         if len_tweet <= seq_len:
           # If its shorter, we fill with zeros (the padding Token index)
           features[ii, -len(tweet):] = np.array(tweet)[:seq_len]
         if len_tweet > seq_len:
           # If its larger, we take the last 'seq_len' indexes
           features[ii, :] = np.array(tweet)[-seq_len:]
     return features
    
    
    # We convert our list of tokens into a numpy matrix
    # where all instances have the same lenght
    #x_train_pad = padding(x_train_idx,max_length)
    #x_test_pad = padding(x_test_idx,max_length)
    
    inTweet_pad = padding(scrapedTweetTokenIDX, max_length)
    
    # We convert our target list a numpy matrix
    # HAD TROUBLE RUNNING EVAL VERSION OF MODEL, USE DUMMY LABELS TO FORMAT INPUT CORRECTLY. 
    #y_train_np = np.asarray(y_train)
    #y_test_np = np.asarray(y_test)
    emptyLabels = np.zeros(len(inTweet_pad))
    
    
    # Now, let's convert the data to pytorch format.
    
    # create Tensor datasets
    #INPUT OF SCRAPED TWEETS INTO NETWORK MAY NEED TO BE CONVERTED TO A TENSOR. 
    #SEE BELOW FOR CREATING TENSOR.
    
    inference_data = TensorDataset(torch.from_numpy(inTweet_pad), torch.from_numpy(emptyLabels))
    
    #train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train_np))
    #valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test_np))
    
    # Batch size (this is an important hyperparameter)
    #batch_size = numtweet
    
    # dataloaders
    # make sure to SHUFFLE your data
    #train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,drop_last = True)
    #valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size,drop_last = True)
    test_loader = DataLoader(inference_data, shuffle = False, batch_size = batch_size, drop_last = True)
    
    
    # # Each batch of data in our traning proccess will have the folllowing format:
    # # Obtain one batch of training data
    # dataiter = iter(train_loader)
    # sample_x, sample_y = dataiter.next()
    
    return test_loader

test_loader = embedding(df2)
#st.write(test_loader)


########### Model Evaluation ####################

def modelEvalFunc(model, test_loader, batch_size):
    model.eval()
    for inputs, labels in test_loader:
    # Move batch inputs and labels to gpu
        inputs, labels = inputs.to(device), labels.to(device)
    
        if model.model_name == 'LSTM':
            # Initialize hidden state 
            val_h = model.init_hidden(batch_size)
            # Creating new variables for the hidden state
            val_h = tuple([each.data.to(device) for each in val_h])
            # Set gradient to zero
            model.zero_grad()
            # Compute model output
            output, val_h = model(inputs,val_h)
            return output
        
##### Call Eval Function ##########################

sentOutput = modelEvalFunc(sentModel, test_loader, batch_size)
#st.write(sentOutput)

sarcasmOutput = modelEvalFunc(sarcasmModel, test_loader, batch_size)
#st.write(sarcasmOutput)

####### Plotting ###############

sentCount = len([i for i in sentOutput if i > 0.5])

sarcasmCount = len([i for i in sarcasmOutput if i > 0.5])


#################### PLOTTING SENTIMENT ############ 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

names = ['Positive', 'Negative']
size = [sentCount, (numTweet - sentCount)]

# Create a circle at the center of the plot
my_circle = plt.Circle((0,0), 0.7, color='white')

# Give color names
fig = plt.figure(figsize=(10, 10))
plt.pie(size, labels=names, colors=['blue', 'grey'], startangle = 90, autopct='%1.1f%%', textprops={'fontsize': 22})
 
p = plt.gcf()
p.gca().add_artist(my_circle)

p.gca().add_artist(my_circle)
plt.title("Sentiment in Regards to " + "'" + searchTerm + "'" + "\n" + "Worldwide, Collected Since " + str(date_since), bbox={'facecolor':'0.8', 'pad':5}, fontsize = 22)

#plt.savefig('/content/drive/MyDrive/Sarcasm Detection/' + searchTerm + 'SentDonut.png')
#plt.show()

buf = BytesIO()
fig.savefig(buf, format="png")
#st.subheader("Sentiment Summary for " + "'" + searchTerm + "'")
st.image(buf, width =900)

#st.subheader("Sentiment Summary for " + "'" + searchTerm + "'")
#st.pyplot(fig, clear_figure=True)
#st.empty()

#################### PLOTTING SARCASM ############

sarcasmNames = ['Sarcastic', 'Literal']
sarcasmSize = [sarcasmCount, (numTweet - sarcasmCount)]
 
# Create a circle at the center of the plot
sarcasmCircle = plt.Circle((0,0), 0.7, color='white')

fig2 = plt.figure(figsize=(10, 10))
plt.pie(sarcasmSize, labels=sarcasmNames, colors=['purple', 'grey'], startangle = 90, autopct='%1.1f%%', textprops={'fontsize': 22})
p = plt.gcf()
p.gca().add_artist(sarcasmCircle)
plt.title("Sarcasm Towards " + "'" + searchTerm + "'" + "\n" + "Worldwide, Collected Since " + str(date_since), bbox={'facecolor':'0.8', 'pad':5},fontsize = 22)
#plt.savefig('/content/drive/MyDrive/Sarcasm Detection/' + searchTerm + 'SarcasmDonut.png')
# Show the graph


#sarcasmPlot.show()

buf2 = BytesIO()
fig2.savefig(buf2, format="png")
#st.subheader("Sarcasm Summary for " + "'" + searchTerm + "'")
st.image(buf2, width = 900)

#st.subheader("Sarcasm Summary for " + "'" + searchTerm + "'")
#st.write(fig2)
#st.pyplot(fig2, clear_figure=True)


##################### WORD CLOUD ###################

