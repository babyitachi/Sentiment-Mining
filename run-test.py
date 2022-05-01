# -*- coding: utf-8 -*-

import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pickle
from nltk.corpus import stopwords
import re
nltk.download("stopwords")


def read_cli():
    parser = argparse.ArgumentParser(description="Evaluation Script.")
    
    parser.add_argument("--model_file",
            help="Model file path",
            required=True, type=str, default="model.pickle"
    )

    parser.add_argument("--input_file",
        help="Input file path",
        required=True, type=str, default="inputfile.txt"
    )

    parser.add_argument("--output_file",
        help="Output file path",
        required=True, type=str, default="outputfile.txt"
    )

    args = parser.parse_args()

    return args

def preProcess(scentence):
    # Remove all the special characters
    #Remove hyperlinks
    processed_feature = re.sub(r'https?:\/\/.*[\r\n]*','', scentence)
    processed_feature = re.sub(r'\W', ' ', processed_feature)
    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()
    return processed_feature

def loadModel(args,vectorizer):
    loaded_model = pickle.load(open(args.model_file, 'rb'))
    output=[]
    for i in open(args.model_file,'r'):
        inputScentence=json.loads(i)
        output.append(loaded_model.predict(vectorizer.transform([preProcess(inputScentence)])))
    with open(args.model_file,'r') as op:
        json.dump(output,op)

if __name__ == '__main__':
    args = read_cli()
    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'),use_idf=True)
    loadModel(args,vectorizer)