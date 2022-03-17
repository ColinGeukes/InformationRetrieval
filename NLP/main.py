import pandas
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

stopwords = set(nltk.corpus.stopwords.words('english'))

def addStopWordsFraction(df):
    stopwordsFractionTitle = []
    stopwordsFractionText = []
    for index, row in df.iterrows():
        tokenizedTitle = word_tokenize(str(row['title']))
        tokenizedText = word_tokenize(str(row['text']))
        if len(tokenizedTitle) != 0:
            stopwordsFractionTitle.append(len(stopwords.intersection(tokenizedTitle)) / len(tokenizedTitle))
        else:
            stopwordsFractionTitle.append(0)

        if len(tokenizedText) != 0:
            stopwordsFractionText.append(len(stopwords.intersection(tokenizedText)) / len(tokenizedText))
        else:
            stopwordsFractionText.append(0)

    df['stopwordsTitle'] = stopwordsFractionTitle
    df['stopwordsText'] = stopwordsFractionText

def runModels(X, y):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.33, random_state=1)

    model = LogisticRegression(solver='liblinear', multi_class='ovr')
    # model = LinearDiscriminantAnalysis()
    # model = KNeighborsClassifier()
    # model = DecisionTreeClassifier()
    # model = GaussianNB()
    # model = SVC(gamma='auto')


    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

def mergeFiles():
    df1 = pandas.read_csv('data/train.csv')
    df1 = df1.head(3000)
    df1.drop(["id", "author"], axis=1, inplace=True)

    return df1

def run():
    print("Reading in datafile...")
    df = mergeFiles()

    print("Adding stopwords to the dataframe...")

    addStopWordsFraction(df)
    df.drop(["title", "text"], axis=1, inplace=True)
    X = df.drop(['label'], axis=1).values
    y = df['label'].values

    print("Running the models...")
    runModels(X, y)

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    run()
