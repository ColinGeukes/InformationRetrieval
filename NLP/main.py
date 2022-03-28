import pandas
import nltk
from nltk.tokenize import word_tokenize
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

stopwords = set()
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

def initSemantics():
    semanticsDict = dict()
    semanticsDict['stopwordsTitle'] = []
    semanticsDict['stopwordsText'] = []

    for tag in TAG_POS:
        semanticsDict['title' + tag] = []
        semanticsDict['text' + tag] = []

    return semanticsDict

def addSemantics(df):
    semanticsDict = initSemantics()
    for index, row in df.iterrows():
        tokenizedTitle = word_tokenize(re.sub(r'[^\w\s]', '', str(row['title']).lower()))
        tokenizedText = word_tokenize(re.sub(r'[^\w\s]', '', str(row['text']).lower()))
        stopWordsFraction(semanticsDict, 'stopwordsTitle', tokenizedTitle)
        stopWordsFraction(semanticsDict, 'stopwordsText', tokenizedText)

        taggedWordsFraction('title', semanticsDict, tokenizedTitle)
        taggedWordsFraction('text', semanticsDict, tokenizedText)

    for key in semanticsDict:
        df[key] = semanticsDict[key]

def runModels(X, y):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.33, random_state=1)

    models = [  ["Logistic regression: ", LogisticRegression(solver='liblinear', multi_class='ovr')],
                ["MLPClassifier: ", MLPClassifier()],
                ["Linear Discriminant: ", LinearDiscriminantAnalysis()],
                ["KNeighbourClassifier: ", KNeighborsClassifier()],
                ["Decision Tree:", DecisionTreeClassifier()],
                ["Gaussian: ", GaussianNB()],
                ["SVC: ", SVC(gamma='auto')]]

    for model in models:
        model[1].fit(X_train, Y_train)
        predictions = model[1].predict(X_validation)
        print(model[0], accuracy_score(Y_validation, predictions))

def mergeFiles():
    # 0: reliable (real)
    # 1: unreliable (fake)

    df1 = pandas.read_csv('data/train.csv')
    df1.drop(["id", "author"], axis=1, inplace=True)

    df2 = pandas.read_csv('data/Fake.csv')
    df2['label'] = 1
    df2.drop(["subject", "date"], axis=1, inplace=True)

    df3 = pandas.read_csv('data/True.csv')
    df3['label'] = 0
    df3.drop(["subject", "date"], axis=1, inplace=True)

    df4 = pandas.read_csv('data/fake_or_real_news.csv')
    df4 = df4.replace('FAKE', 1)
    df4 = df4.replace('REAL', 0)
    df4.drop(["Unnamed: 0"], axis=1, inplace=True)

    return df1.append(df2).append(df3).append(df4)

def run():
    print("Reading in datafile...")
    df = mergeFiles()

    # Just take a smaller part of the dataset
    df = df.head(500)

    print("Adding semantics to the dataframe...")
    addSemantics(df)

    df.drop(["title", "text"], axis=1, inplace=True)
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
