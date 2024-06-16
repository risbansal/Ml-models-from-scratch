#!/usr/bin/env python
# coding: utf-8

# In[13]:


import csv
import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def read_file(file_path,file_name):
    txt_file = file_path
    csv_file = file_name

    file1 = csv.reader(open(txt_file, "r"), delimiter = ' ')
   
    file2 = csv.writer(open(csv_file, 'w'))
   

    file2.writerows(file1)
    

def preprocessor(data):


    col_list = ["word", "pos", "chunk", "ner"]
    data.columns = col_list

    data.isnull().sum()
    data = data.fillna(method='ffill')


    data["synonyms"] = np.nan
    data["antonyms"] = np.nan
    data["hypernyms"] = np.nan
    data["hyponyms"] = np.nan
    data["holonyms"] = np.nan
    data["meronyms"] = np.nan

    data['synonyms'] = data['synonyms'].astype(object)
    data['antonyms'] = data['antonyms'].astype(object)
    data['hypernyms'] = data['hypernyms'].astype(object)
    data['hyponyms'] = data['hyponyms'].astype(object)
    data['holonyms'] = data['holonyms'].astype(object)
    data['meronyms'] = data['meronyms'].astype(object)

    return data



def get_features(data):
    for index, words in data.iterrows():

        synonyms = []
        antonyms = []
        hypernyms = []
        hyponyms = []
        holonyms = []
        meronyms = []

        for s in wn.synsets(words[0]):

            l = wn.lemmas(words[0])
            for l in s.lemmas():
                synonyms.append(l.name())
                
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
            if s.hypernyms():
                hypernyms.append(s.hypernyms()[0].name().split('.')[0])
            if s.hyponyms():
                 hyponyms.append(s.hyponyms()[0].name().split('.')[0])   
            if s.member_holonyms():
                 holonyms.append(s.member_holonyms()[0].name().split('.')[0])
            if s.part_meronyms():
                 meronyms.append(s.part_meronyms()[0].name().split('.')[0])

           
            if synonyms:
                data.at[index,'synonyms'] = synonyms[0]
            if antonyms:
                data.at[index,'antonyms'] = antonyms[0]
            if hypernyms:
                data.at[index,'hypernyms'] = hypernyms[0]
            if hyponyms:
                data.at[index,'hyponyms'] = hyponyms[0]
            if holonyms:
                data.at[index,'holonyms'] = holonyms[0]
            if meronyms:
                data.at[index,'meronyms'] = meronyms[0]
            

    return data

def main():


	parser = argparse.ArgumentParser()
	parser.add_argument('-tr', '--train', type = str)
	parser.add_argument('-t', '--test', type = str)
	arg = parser.parse_args()

	#getting csv files
	train_file = arg.train
	test_file = arg.test
	read_file(train_file, "train.csv")
	read_file(test_file, "test.csv")

	#reading csv
	train_data = pd.read_csv("train.csv")
	test_data = pd.read_csv("test.csv")

	#Preprocessing
	train_data1 = preprocessor(train_data)
	test_data1 = preprocessor(test_data)

	#Adding features
	train_data2 = get_features(train_data1)
	test_data2 = get_features(test_data1)

	#Handling nan values
	train_data2 = train_data2.replace(np.nan, 'null', regex=True)
	test_data2 = test_data2.replace(np.nan, 'null', regex=True)



	train_data2 = train_data2[:200]
	test_data2 = test_data2[:200]


	#Vectoring data
	X_train = train_data2.drop('ner', axis=1)
	X_test = test_data2.drop('ner', axis=1)

	v = DictVectorizer(sparse=False)
	X_train = v.fit_transform(X_train.to_dict('records'))
	X_test = v.fit_transform(X_test.to_dict('records'))

	y_train = train_data2.ner.values
	y_test = test_data2.ner.values


	#Model training and testing
	classify = LogisticRegression(random_state=0, solver='lbfgs')
	classify.fit(X_train, y_train)
	sc = classify.score(X_train,y_test)
	y_pred = classify.predict(X_train)
	report = classification_report(y_train, y_pred)


	#Printing results
	print("Report")
	print(report)
	print("Accuracy Score:", sc)



main()

