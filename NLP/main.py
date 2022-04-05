import json
import re
import time
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

stopwords = set()

ADD_STOPWORDS = False
ADD_SEMANTICS = False
ADD_DOCUMENT_LENGTH = False
ADD_CAPITALS = False
ADD_URLS = False
ADD_SYMBOLS = False
WORD2VEC = False
CREATE_WORD2VEC_MODEL = False
LOAD_FROM_FILE = False
LOAD_WORD_PLOT_FROM_FILE = True
CREATE_WORD_PLOT = True

# Full list available : https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
TAG_POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
           'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
           'WP', 'WP$', 'WRB']


# TAG_POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB']

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


def addUrls(key, semanticsDict, text):
    # TODO: normalize
    # Regex taken from: https://www.geeksforgeeks.org/python-check-url-string/
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls = re.findall(regex, text)
    semanticsDict[key].append(len(urls))


def addSymbols(key, semanticsDict, text):
    # TODO: normalize
    exclamations = text.count("!")
    questionmarks = text.count("?")
    hashtags = text.count("#")
    singlequotes = text.count("'")
    doublequotes = text.count('"')
    asperands = text.count("@")

    semanticsDict[key + 'Exclamations'].append(exclamations)
    semanticsDict[key + 'Questionmarks'].append(questionmarks)
    semanticsDict[key + 'Hashtags'].append(hashtags)
    semanticsDict[key + 'Singlequotes'].append(singlequotes)
    semanticsDict[key + 'Doublequotes'].append(doublequotes)
    semanticsDict[key + 'Asperands'].append(asperands)


def addWord2Vec(key, sementicsDict, text):
    print("TokenText", text)


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
    if ADD_URLS:
        semanticsDict['containsUrls'] = []
    if ADD_SYMBOLS:
        def initSymbols(key):
            semanticsDict[key + 'Exclamations'] = []
            semanticsDict[key + 'Questionmarks'] = []
            semanticsDict[key + 'Hashtags'] = []
            semanticsDict[key + 'Singlequotes'] = []
            semanticsDict[key + 'Doublequotes'] = []
            semanticsDict[key + 'Asperands'] = []

        initSymbols('titleSymbols')
        initSymbols('textSymbols')

    if WORD2VEC:
        semanticsDict['titleWord2Vec'] = []
        semanticsDict['textWord2Vec'] = []

    return semanticsDict


def wordPlot(wordDict, title, savefile):
    wordDict = dict(sorted(wordDict.items(), key=lambda item: item[1], reverse=True))
    items = {k: wordDict[k] for k in list(wordDict)[:10]}
    plt.ylabel("Occurence")
    plt.title(title)
    plt.bar(range(len(items)), items.values(), align="center")

    plt.xticks(range(len(items)), list(items.keys()), rotation=20)
    plt.savefig(savefile)
    plt.show()


def wordCount(wordDict, text):
    for word in text:
        if word not in stopwords:
            if word in wordDict:
                wordDict[word] += 1
            else:
                wordDict[word] = 1


def split_into_sentences(body):
    return [re.sub('[!@#$:;’”,."()_\'–-…/]', '', x).split() for x in re.split('[.!?]+ ', str(body).lower())]


def addSemantics(df):
    wordDict = [dict(), dict()]
    semanticsDict = initSemantics()

    title_word2vec = None
    text_word2vec = None
    title_word_labels = None
    text_word_labels = None
    title_similar_words = {}
    text_similar_words = {}
    if WORD2VEC:
        # title_word2vec_data = open("word2vec-title.data", "r")
        title_word2vec = KeyedVectors.load("word2vec-title.model", mmap='r')
        # text_word2vec_data = open("word2vec-text.data", "r")
        text_word2vec = KeyedVectors.load("word2vec-text.model", mmap='r')

        with open('title-word-labels.json') as title_json_file:
            title_word_labels = json.load(title_json_file)

        with open('text-word-labels.json') as text_json_file:
            text_word_labels = json.load(text_json_file)

    index2 = 0
    start = time.time()

    if not LOAD_WORD_PLOT_FROM_FILE:
        for index, row in df.iterrows():
            if index2 % 100 == 0:
                print(f"{index2} / {len(df)}, time elapsed: {time.time() - start}")

            title = str(row['title'])
            text = str(row['text'])
            label = int(row['label'])

            tokenizedTitle = word_tokenize(re.sub(r'[^\w\s]', '', title.lower()))
            tokenizedText = word_tokenize(re.sub(r'[^\w\s]', '', text.lower()))

            tokenizedTitle_keepCapitals = word_tokenize(re.sub(r'[^\w\s]', '', title))
            tokenizedText_keepCapitals = word_tokenize(re.sub(r'[^\w\s]', '', text))

            tokenizedTitle_keepSymbols = word_tokenize(title)
            tokenizedTitle_keepSymbols = word_tokenize(text)

            if CREATE_WORD_PLOT:
                wordCount(wordDict[label], tokenizedTitle)
                wordCount(wordDict[label], tokenizedText)
            if ADD_STOPWORDS:
                stopWordsFraction(semanticsDict, 'stopwordsTitle', tokenizedTitle)
                stopWordsFraction(semanticsDict, 'stopwordsText', tokenizedText)

            if ADD_SEMANTICS:
                taggedWordsFraction('title', semanticsDict, tokenizedTitle)
                taggedWordsFraction('text', semanticsDict, tokenizedText)

            if ADD_DOCUMENT_LENGTH:
                documentLength('titleLength', semanticsDict, title)
                documentLength('textLength', semanticsDict, text)

            if ADD_CAPITALS:
                wordCapitals('title', semanticsDict, tokenizedTitle_keepCapitals)
                wordCapitals('text', semanticsDict, tokenizedText_keepCapitals)

            if ADD_URLS:
                addUrls('containsUrls', semanticsDict, text)

            if ADD_SYMBOLS:
                addSymbols('titleSymbols', semanticsDict, title)
                addSymbols('textSymbols', semanticsDict, text)

            if WORD2VEC:
                get_most_similar_label('titleWord2Vec', semanticsDict, title_similar_words, row['title'], title_word2vec,
                                       title_word_labels, 10)
                get_most_similar_label('textWord2Vec', semanticsDict, text_similar_words, row['text'], text_word2vec,
                                       text_word_labels, 10)

            index2 += 1

    if CREATE_WORD_PLOT:
        # Save the data to file
        if LOAD_WORD_PLOT_FROM_FILE:
            with open('word-dict.json', "r") as word_dict_file:
                wordDict = json.load(word_dict_file)
        else:
            with open("word-dict.json", "w") as word_dict_file:
                word_dict_file.write(json.dumps(wordDict))

        # Create plot for unreliable
        wordPlot(wordDict[0], 'Word occurrences for reliable news', 'reliable.pdf')
        wordPlot(wordDict[1], 'Word occurrences for unreliable news', 'unreliable.pdf')

    for key in semanticsDict:
        df[key] = semanticsDict[key]


def get_most_similar_label(key, semanticDict, similar_words_store, body, word2vec, label_prob, topn=20):
    split = split_into_sentences(body)
    total_probs = 0
    words = 0

    for sentence in split:
        for word in sentence:
            try:
                if word in similar_words_store:
                    similar_words = similar_words_store[word]
                else:
                    similar_words = word2vec.most_similar(positive=word, topn=topn)
                    similar_words_store[word] = similar_words

                total_similarity_prob = 0
                word_label_prob = 0
                for similar_word in similar_words:
                    similar_word_str = similar_word[0]
                    similar_word_prob = similar_word[1]
                    word_label_prob += label_prob[similar_word_str][1] / (
                            label_prob[similar_word_str][0] + label_prob[similar_word_str][1]) * \
                                       similar_word_prob
                    total_similarity_prob += similar_word_prob
                total_probs += word_label_prob / total_similarity_prob
                words += 1
            except:
                continue

    similar_label = 0.5
    if words > 0:
        similar_label = total_probs / words

    semanticDict[key].append(similar_label)


def create_word2vec_models(df):
    title_sentences = []
    text_sentences = []
    title_word_label_prob = {}
    text_word_label_prob = {}

    def split_sentences(body, sentences, word_label_prob, label):
        splits = split_into_sentences(body)

        # Add sentences to list of sentence
        sentences.extend(splits)

        # Assign label to words
        for sentence in splits:
            for word in sentence:
                if word not in word_label_prob:
                    word_label_prob[word] = [0, 0]
                word_label_prob[word][label] += 1

    index2 = 0
    for index, row in df.iterrows():
        if index2 % 1000 == 0:
            print(f"{index2} / {len(df)}")

        label = int(row['label'])
        split_sentences(str(row['title']), title_sentences, title_word_label_prob, label)
        split_sentences(str(row['text']), text_sentences, text_word_label_prob, label)
        index2 += 1

    # Create the model
    print("Creating the title word2vec model...")
    title_word2vec = Word2Vec(title_sentences, vector_size=100, window=5, min_count=5, workers=4)
    print("Creating the text word2vec model...")
    text_word2vec = Word2Vec(text_sentences, vector_size=100, window=5, min_count=5, workers=4)

    # Store the model
    print("Storing the models...")
    title_word2vec.wv.save("word2vec-title.model")
    text_word2vec.wv.save("word2vec-text.model")

    print("Storing the word labels...")
    with open("title-word-labels.json", "w") as outfile:
        outfile.write(json.dumps(title_word_label_prob))
    with open("text-word-labels.json", "w") as outfile:
        outfile.write(json.dumps(text_word_label_prob))


def runModels(X, y):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.33, random_state=1)

    models = [["Logistic regression: ", LogisticRegression(solver='liblinear', multi_class='ovr'), False],
              ["MLPClassifier: ", MLPClassifier(), False],
              ["Linear Discriminant: ", LinearDiscriminantAnalysis(), False],
              ["KNeighbourClassifier: ", KNeighborsClassifier(), False],
              ["Decision Tree:", DecisionTreeClassifier(), False],
              ["Gaussian: ", GaussianNB(), False],
              ["SVC: ", SVC(max_iter=10000, gamma='auto'), False],
              ["Random Forest: ", RandomForestClassifier(random_state=0), True]]

    # Just ignore the converge warning when the dataset is to small
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for model in models:
        model[1].fit(X_train, Y_train)
        predictions = model[1].predict(X_validation)
        print(model[0])
        print('Accuracy: ', accuracy_score(Y_validation, predictions))
        print('Recall: ', recall_score(Y_validation, predictions))
        print('Precision: ', precision_score(Y_validation, predictions))
        print('F1 score: ', f1_score(Y_validation, predictions))
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


def toFile(df1, fileName):
    df1.to_csv("./data/" + fileName)


def mergeFiles():
    # 0: reliable (real)
    # 1: unreliable (fake)

    df1 = pd.read_csv('data/train.csv')
    df1.drop(["id", "author"], axis=1, inplace=True)

    df2_fake = pd.read_csv('data/Fake.csv')
    df2_fake['label'] = 1
    df2_fake.drop(["subject", "date"], axis=1, inplace=True)

    df2_true = pd.read_csv('data/True.csv')
    df2_true['label'] = 0
    df2_true.drop(["subject", "date"], axis=1, inplace=True)

    df3 = pd.read_csv('data/fake_or_real_news.csv')
    df3 = df3.replace('FAKE', 1).replace('REAL', 0)
    df3.drop(["Unnamed: 0"], axis=1, inplace=True)

    dst = df1.append(df2_fake).append(df2_true).append(df3)
    return dst
    # return df3[:1500]


def run():
    print("Reading in datafile...")

    if LOAD_FROM_FILE:
        scaled_df = pd.read_csv("./data/d3-baseline-features.csv")
    else:
        df = mergeFiles()
        print(df)

        # Create the word 2 vec model
        if CREATE_WORD2VEC_MODEL:
            print("Creating word2vec model...")
            create_word2vec_models(df)

        # Just take a smaller part of the dataset
        print("Adding semantics to the dataframe...")
        addSemantics(df)

        df.drop(["title", "text"], axis=1, inplace=True)

        # Normalize the matrix
        scaler = MinMaxScaler()
        scaler.fit(df)
        scaled = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled, columns=df.columns)

        print(scaled_df)

        toFile(scaled_df, "baseline-features.csv")

    X = scaled_df.drop(['label'], axis=1).values
    y = scaled_df['label'].values
    print("Running the models...")
    runModels(X, y)


if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    run()
