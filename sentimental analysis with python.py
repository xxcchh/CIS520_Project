#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:05:19 2016

@author: chen
"""
# sentimental anaylsis use vader
import pandas as pd
import re
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()
sentimentalWords = pd.read_csv("topwords.csv", header=None)
sentimentalSentences = pd.read_csv("raw_tweets.csv", header=None)
sentimentalResult = pd.read_csv("sentimentalResult.csv", header=0)

# Ensemble 1 method
# To get sentimenral result for each word in top words list


def dealSentences(df):
    n = len(df)
    df.ix[:, 'neg'] = 0
    df.ix[:, 'neu'] = 0
    df.ix[:, 'pos'] = 0
#    res = {}.fromkeys(['neg', 'neu', 'pos'], [])
    for i in range(n):
        sentence = str(df.iloc[i].values[0])
        sentence = sentence.translate(None, '#')
        re.sub('@\w+', '', sentence)
        sentiment = vader_analyzer.polarity_scores(sentence)
        df.ix[i, 'pos'] = sentiment['pos']
        df.ix[i, 'neg'] = sentiment['neg']
        df.ix[i, 'neu'] = sentiment['neu']
    return df
# To get postitive, neutral, negtive emotions of each word
sentimentalWords = dealSentences(sentimentalData)
# To get postitive, neutral, negtive emotions of each sentence
sentimentalSentences = dealSentences(sentimentalSentences)


# Ensemble 2 method
# Make some changes to the first sentimentalData, focus on the word with #


def newSentimentalResult(df):
    n = len(df)
    sentimentalResultFinal = df
    for i in range(n):
        name = df.ix[i, '0']  # name
        name = str(name)
        if "#" in name:
            pos = df.ix[i, 'pos']
            neg = df.ix[i, 'neg']
            if pos > neg:
                sentimentalResultFinal.ix[i, 'pos'] = 1
            elif neg > pos:
                sentimentalResultFinal.ix[i, 'neg'] = 1
    return sentimentalResultFinal
# To get new postitive, neutral, negtive emotions of each word based on
# former sentimental data
sentimentalResultNew = newSentimentalResult(sentimentalResult)
