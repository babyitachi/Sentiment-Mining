# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download("stopwords")

def runAlgo(args):
    #constants
    training_file_path=args.data_directory+'/training.csv'
    
    #read training file
    training_data =pd.read_csv(training_file_path,encoding='ISO-8859-1',header=None)
    training=training_data[:].to_numpy()
    
    preprocessedtweets=[]
    
    for index,sentence in enumerate(training):
        #Remove hyperlinks
        preprocessedtweet = re.sub(r'https?:\/\/.*[\r\n]*','', str(sentence[1]))
        preprocessedtweet = re.sub(r'\W', ' ', preprocessedtweet)
        # remove all single characters
        preprocessedtweet= re.sub(r'\s+[a-zA-Z]\s+', ' ', preprocessedtweet)
        # Remove single characters from the start
        preprocessedtweet = re.sub(r'\^[a-zA-Z]\s+', ' ', preprocessedtweet) 
        # Substituting multiple spaces with single space
        preprocessedtweet = re.sub(r'\s+', ' ', preprocessedtweet, flags=re.I)
        # Removing prefixed 'b'
        preprocessedtweet = re.sub(r'^b\s+', '', preprocessedtweet)
        # Converting to Lowercase
        preprocessedtweet = preprocessedtweet.lower()
        #stemming
        stemmer = PorterStemmer()
        steam_words=[]
        for word in preprocessedtweet.split(" "):
            stem_word = stemmer.stem(word)
            steam_words.append(stem_word)
        preprocessedtweets.append(preprocessedtweet)
    
    pt=pd.DataFrame(preprocessedtweets)
    pttraining=pd.concat([training_data[0],pt[0]], axis=1)
    pttraining.columns=[0,1]
    
    X=pttraining[1]
    y=pttraining[0]
    
    vectorizer = TfidfVectorizer(max_features=10000, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    X_train = vectorizer.fit(X).transform(X)
    
    gnb = MultinomialNB()
    
    trainedModel=gnb.fit(X_train, y.astype('int'))
    
    filename = args.model_directory+'/finalized_model.sav'
    pickle.dump(trainedModel, open(filename, 'wb'))
    filename = args.model_directory+'/finalized_Vectorizer.sav'
    pickle.dump(vectorizer, open(filename, 'wb'))

def read_cli():
    parser = argparse.ArgumentParser(description="Training Script.")
    
    parser.add_argument("--data_directory",
            help="Training Data dir path",
            required=True, type=str, default="./"
    )

    parser.add_argument("--model_directory",
        help="Trained Model output dir path",
        required=True, type=str, default="./"
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = read_cli()
    runAlgo(args)
