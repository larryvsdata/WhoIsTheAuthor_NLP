# -*- coding: utf-8 -*-
"""
Created on Fri May 18 14:18:51 2018

@author: Erman
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import collections
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

import nltk
from nltk.corpus import stopwords

from os import listdir
from os.path import isfile, join
import string
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.decomposition import PCA


class findAuthor():
    
    def __init__(self):
        self.authors=[]
        self.rawLabels=[]
        self.encodedLabels=[]
        self.rawDicts=[]
        self.allWords=[]
        self.df=pd.DataFrame()
        self.n_components=150
        self.X=[]
        self.X_train=[]
        self.X_test=[]
        self.y_train=[]
        self.y_test=[]
        self.X_train_lda=[]
        self.X_test_lda=[]
        self.X_train_pca=[]
        self.X_test_pca=[]
        self.svm=SVC()
        self.pca=PCA(n_components=self.n_components)
        
        
    def getAuthors(self):
        self.authors=[name for name in os.listdir(".") if os.path.isdir(name)]
        
    def getDictList(self):
        
        for folderName in self.authors:
            files = [f for f in listdir(folderName) if isfile(join(folderName, f))]
            
            for file in files:
                path=folderName+'/'+file
                try:
                    f = open(path,'r', encoding="utf8")
                    print(path)
                    book = f.read()
                    self.rawDicts.append(self.formADict(book))
                    self.rawLabels.append(folderName)
                    f.close()
                except:
                    print("Error while reading "+path)

                
                
                
    def formADict(self,book):
        stop_words = set(stopwords.words('english'))
        exclude = set(string.punctuation)
        toks = nltk.word_tokenize(book)
        toks=[word.lower() for word in toks]
        toks=[word for word in toks if( word not in stop_words ) and  word not in exclude]
        
        freq = nltk.FreqDist(toks)

        length=len(freq)

        for key, val in freq.items():
            freq[key]=freq[key]/length
            
        return freq
    
    def getAllWords(self):
        
        wordList=[]
        for aDict in self.rawDicts:
            print(len(wordList),len(aDict.keys()))
            wordList=list(set(wordList+list(aDict.keys())))
        self.allWords=wordList
        
    def formDataFrame(self):
        self.df=pd.DataFrame(np.zeros((len(self.rawLabels),len(self.allWords))),columns=self.allWords)
        
    def fillTheFrame(self):
        
        for ind in range(len(self.rawDicts)):
            oneDict=self.rawDicts[ind]
            for word in oneDict:
                self.df[word][ind]=oneDict[word]
                
    def encodeLabels(self):
        le=preprocessing.LabelEncoder()
        le.fit(self.rawLabels)
        self.encodedLabels=le.transform(self.rawLabels)
        
    def splitData(self,ratio):
        self.X=self.df.as_matrix()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.encodedLabels, test_size=ratio, random_state=42)
        
        
    def transform(self):
        sc=StandardScaler()
        self.X_train=sc.fit_transform(self.X_train)
        self.X_test=sc.fit_transform(self.X_test)
        
    def transformLDA(self):
        
        lda=LinearDiscriminantAnalysis(n_components=self.n_components,solver='svd')
        self.X_train_lda=lda.fit_transform(self.X_train,self.y_train)
        self.X_test_lda=lda.transform(self.X_test)
        
    def transformPCA(self):
#        sc=StandardScaler()
#        self.X=sc.fit_transform(self.X)        
#        self.pca.fit(self.X)
        self.X_train_pca=self.pca.fit_transform(self.X_train)
        self.X_test_pca=self.pca.transform(self.X_test)    
    
        
    def trainDataLDA(self):
        svm=SVC()
        clf_svm=GridSearchCV(svm,{'kernel':['linear','poly','rbf'],'C':[0.1,1,10,100]})
        clf_svm.fit(self.X_train_lda,self.y_train)
        print(clf_svm.best_params_)
        
        self.svm=clf_svm
        
    def trainDataPCA(self):
        svm=SVC()
        clf_svm=GridSearchCV(svm,{'kernel':['linear','poly','rbf'],'C':[0.1,1,10,100]})
        clf_svm.fit(self.X_train_pca,self.y_train)
        print(clf_svm.best_params_)
        
        self.svm=clf_svm   
        
        
        
    def trainData(self):
        svm=SVC()
        clf_svm=GridSearchCV(svm,{'kernel':['linear','poly','rbf'],'C':[0.1,1,10,100]})
        clf_svm.fit(self.X_train,self.y_train)
        print(clf_svm.best_params_)
        
        self.svm=clf_svm
        
    def predictAndScoreLDA(self):
        y_pred=self.svm.predict(self.X_test_lda)
        print("Accuracy Score: ", accuracy_score(y_pred,self.y_test ))
        print("Confusion Matrix: ")
        print( confusion_matrix(y_pred,self.y_test ))
        print("Classification Report: ")
        print( classification_report(y_pred,self.y_test ))
        
    def predictAndScorePCA(self):
        y_pred=self.svm.predict(self.X_test_pca)
        print("Accuracy Score: ", accuracy_score(y_pred,self.y_test ))
        print("Confusion Matrix: ")
        print( confusion_matrix(y_pred,self.y_test ))
        print("Classification Report: ")
        print( classification_report(y_pred,self.y_test ))
        

        
        
    def predictAndScore(self):
        y_pred=self.svm.predict(self.X_test)
        print("Accuracy Score: ", accuracy_score(y_pred,self.y_test ))
        print("Confusion Matrix: ")
        print( confusion_matrix(y_pred,self.y_test ))
        print("Classification Report: ")
        print( classification_report(y_pred,self.y_test ))
        
    def originalWrapper(self,ratio):
        
        self.getAuthors()
        self.getDictList()
        self.getAllWords()
        self.formDataFrame()
        self.fillTheFrame()
        self.encodeLabels()
        self.splitData(ratio)
        self.transform()
        self.trainData()
        self.predictAndScore()
        
    def WrapperLDA(self,ratio):
        
        self.getAuthors()
        self.getDictList()
        self.getAllWords()
        self.formDataFrame()
        self.fillTheFrame()
        self.encodeLabels()
        self.splitData(ratio)
        self.transform()
        self.transformLDA()
        self.trainDataLDA()
        self.predictAndScoreLDA()
                    
    def WrapperPCA(self,ratio):
        
        self.getAuthors()
        self.getDictList()
        self.getAllWords()
        self.formDataFrame()
        self.fillTheFrame()
        self.encodeLabels()
        self.splitData(ratio)
        self.transform()
        self.transformPCA()
        self.trainDataPCA()
        self.predictAndScorePCA() 
        
    def wrapperSelection(self,option,ratio):
        
        if option==0:
            self.originalWrapper(ratio)
        elif option==1:
            self.WrapperLDA(ratio)
        elif option==2:
            self.WrapperPCA(ratio)
        
        
            
        
        
if __name__ == '__main__':
    
    findAuthor=findAuthor()
    
    ratio=0.33
    option=2
    findAuthor.wrapperSelection(option,ratio)
    
    
    
    
#    findAuthor.getAuthors()
#    findAuthor.getDictList()
#    findAuthor.getAllWords()
#    findAuthor.formDataFrame()
#    findAuthor.fillTheFrame()
#    findAuthor.encodeLabels()
#    
#    findAuthor.splitData(0.3)
#    findAuthor.transform()
#    findAuthor.transformLDA()
#    findAuthor.trainData()
#    findAuthor.predictAndScore()


    
        
            