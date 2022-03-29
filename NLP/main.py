import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning

import warnings

stopwords = set()

ADD_STOPWORDS = True
ADD_SEMANTICS = True
ADD_DOCUMENT_LENGTH = True
ADD_CAPITALS = True

# Full list available : https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
TAG_POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
# TAG_POS = ['IN', 'NN', 'JJ']

def stopWordsFraction(semanticsDict, key, tokenizedText):
    if len(tokenizedText) != 0:
        semanticsDict[key].append(len(stopwords.intersection(tokenizedText)) / len(tokenizedText))
    else:
        semanticsDict[key].append(0)

def taggedWordsFraction(key, semanticsDict, tokenizedText):
    tags = [t[1] for t in nltk.pos_tag(tokenizedText)]

    for tag in TAG_POS:
        if len(tokenizedText) != 0:
            semanticsDict[key + tag].append(tags.count(tag) / len(tokenizedText))
        else:
            semanticsDict[key + tag].append(0)

def documentLength(key, semanticsDict, tokenizedText):
    # TODO: normalize
    semanticsDict[key].append(len(tokenizedText))

def wordCapitals(key, semanticsDict, tokenizedText):
    fullCapitals = 0
    startCapitals = 0
    containsCapital = 0
    noCapitals = 0
    for token in tokenizedText:
        if token[0].isupper():
            startCapitals = startCapitals + 1
            if token.isupper():
                fullCapitals = fullCapitals + 1
        if token.islower():
            noCapitals = noCapitals + 1
        else:
            containsCapital = containsCapital + 1
    tokenLength = len(tokenizedText)
    semanticsDict[key + 'FullCapital'].append(fullCapitals / tokenLength if tokenLength != 0 else 0)
    semanticsDict[key + 'StartCapital'].append(startCapitals / tokenLength if tokenLength != 0 else 0)
    semanticsDict[key + 'ContainsCapital'].append(containsCapital / tokenLength if tokenLength != 0 else 0)
    semanticsDict[key + 'NoCapitals'].append(noCapitals / tokenLength if tokenLength != 0 else 0)

def initSemantics():
    semanticsDict = dict()
    if ADD_STOPWORDS:
        semanticsDict['stopwordsTitle'] = []
        semanticsDict['stopwordsText'] = []

    if ADD_SEMANTICS:
        for tag in TAG_POS:
            semanticsDict['title' + tag] = []
            semanticsDict['text' + tag] = []
    if ADD_DOCUMENT_LENGTH:
        semanticsDict['titleLength'] = []
        semanticsDict['textLength'] = []
    if ADD_CAPITALS:
        def initCapitals(key):
            semanticsDict[key + 'FullCapital'] = []
            semanticsDict[key + 'StartCapital'] = []
            semanticsDict[key + 'ContainsCapital'] = []
            semanticsDict[key + 'NoCapitals'] = []
        initCapitals('title')
        initCapitals('text')
    return semanticsDict

def addSemantics(df):
    semanticsDict = initSemantics()
    for index, row in df.iterrows():
        print(index)

        tokenizedTitle = word_tokenize(re.sub(r'[^\w\s]', '', str(row['title'])))
        tokenizedText = word_tokenize(re.sub(r'[^\w\s]', '', str(row['text'])))
        if ADD_CAPITALS:
            wordCapitals('title', semanticsDict, tokenizedTitle)
            print(semanticsDict)
            print(tokenizedTitle)

        tokenizedTitle = word_tokenize(re.sub(r'[^\w\s]', '', str(row['title']).lower()))
        tokenizedText = word_tokenize(re.sub(r'[^\w\s]', '', str(row['text']).lower()))
        if ADD_STOPWORDS:
            stopWordsFraction(semanticsDict, 'stopwordsTitle', tokenizedTitle)
            stopWordsFraction(semanticsDict, 'stopwordsText', tokenizedText)

        if ADD_SEMANTICS:
            taggedWordsFraction('title', semanticsDict, tokenizedTitle)
            taggedWordsFraction('text', semanticsDict, tokenizedText)
        if ADD_DOCUMENT_LENGTH:
            documentLength('titleLength', semanticsDict, tokenizedTitle)
            documentLength('textLength', semanticsDict, tokenizedText)

    for key in semanticsDict:
        df[key] = semanticsDict[key]

def runModels(X, y):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.33, random_state=1)

    models = [  ["Logistic regression: ", LogisticRegression(solver='liblinear', multi_class='ovr'), False],
                ["MLPClassifier: ", MLPClassifier(), False],
                ["Linear Discriminant: ", LinearDiscriminantAnalysis(), False],
                ["KNeighbourClassifier: ", KNeighborsClassifier(), False],
                ["Decision Tree:", DecisionTreeClassifier(), False],
                ["Gaussian: ", GaussianNB(), False],
                ["SVC: ", SVC(gamma='auto'), False],
                ["Random Forest: ", RandomForestClassifier(random_state=0), True]]

    # Just ignore the converge warning when the dataset is to small
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for model in models:
        model[1].fit(X_train, Y_train)
        predictions = model[1].predict(X_validation)
        print(model[0])
        print('Accuracy: ',  accuracy_score(Y_validation, predictions))
        print('Recall: ',  recall_score(Y_validation, predictions))
        print('F1 score: ',  f1_score(Y_validation, predictions))
        tn, fp, fn, tp = confusion_matrix(Y_validation, predictions).ravel()
        print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
        if model[2]:
            feature_names = range(X.shape[1])
            importances = model[1].feature_importances_
            std = np.std([tree.feature_importances_ for tree in model[1].estimators_], axis=0)
            forest_importances = pd.Series(importances, index=feature_names)

            fig, ax = plt.subplots()
            forest_importances.plot.bar(yerr=std, ax=ax)
            ax.set_title("Feature importances using MDI")
            ax.set_ylabel("Mean decrease in impurity")
            fig.tight_layout()
            plt.show()
        print()

def mergeFiles():
    # 0: reliable (real)
    # 1: unreliable (fake)

    df1 = pd.read_csv('data/train.csv')
    df1.drop(["id", "author"], axis=1, inplace=True)

    df2 = pd.read_csv('data/Fake.csv')
    df2['label'] = 1
    df2.drop(["subject", "date"], axis=1, inplace=True)

    df3 = pd.read_csv('data/True.csv')
    df3['label'] = 0
    df3.drop(["subject", "date"], axis=1, inplace=True)

    df4 = pd.read_csv('data/fake_or_real_news.csv')
    df4 = df4.replace('FAKE', 1).replace('REAL', 0)
    df4.drop(["Unnamed: 0"], axis=1, inplace=True)

    return df1.append(df2).append(df3).append(df4)

def run():
    print("Reading in datafile...")
    df = mergeFiles()

    # Just take a smaller part of the dataset
    print("Adding semantics to the dataframe...")
    addSemantics(df)


    df.drop(["title", "text"], axis=1, inplace=True)
    print(df)

    X = df.drop(['label'], axis=1).values
    y = df['label'].values
    print("Running the models...")
    runModels(X, y)

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    run()
