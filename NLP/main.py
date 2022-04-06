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
import seaborn as svm

# Feature extraction settings
ADD_STOPWORDS = False
ADD_SEMANTICS = False
ADD_DOCUMENT_LENGTH = False
ADD_CAPITALS = False
ADD_URLS = False
ADD_SYMBOLS = False
ADD_WORD2VEC = False

# Data analysis settings
CREATE_WORD2VEC_MODEL = False
CREATE_WORD_PLOT = False
CREATE_HEATMAP = True

# Load data from file
LOAD_DATA_FROM_FILE = True
LOAD_WORD_PLOT_FROM_FILE = False

# Stopwords placeholder
stopwords = set()

# Full list available : https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
TAG_POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
           'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
           'WP', 'WP$', 'WRB']

# TAG_POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB']

# TAG_POS = ['IN', 'NN', 'JJ']


def stopWordsFraction(semanticsDict, key, tokenizedText):
    """Add stopWordsFraction to features

    Args:
        semanticsDict: Pandas dataframe
        key: Name of features
        tokenizedText: Text to generate features from
    """
    if len(tokenizedText) != 0:
        semanticsDict[key].append(len(stopwords.intersection(tokenizedText)) / len(tokenizedText))
    else:
        semanticsDict[key].append(0)


def taggedWordsFraction(key, semanticsDict, tokenizedText):
    """Add taggedWordsFraction to features

    Args:
        semanticsDict: Pandas dataframe
        key: Name of features
        tokenizedText: Text to generate features from
    """
    tags = [t[1] for t in nltk.pos_tag(tokenizedText)]

    for tag in TAG_POS:
        if len(tokenizedText) != 0:
            semanticsDict[key + tag].append(tags.count(tag) / len(tokenizedText))
        else:
            semanticsDict[key + tag].append(0)


def documentLength(key, semanticsDict, tokenizedText):
    """Add documentLength to features

    Args:
        semanticsDict: Pandas dataframe
        key: Name of features
        tokenizedText: Text to generate features from
    """
    semanticsDict[key].append(len(tokenizedText))


def wordCapitals(key, semanticsDict, tokenizedText):
    """Add wordCapitals to features

    Args:
        semanticsDict: Pandas dataframe
        key: Name of features
        tokenizedText: Text to generate features from
    """
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
    """Add urls to features

    Args:
        semanticsDict: Pandas dataframe
        key: Name of features
        tokenizedText: Text to generate features from
    """
    # Regex taken from: https://www.geeksforgeeks.org/python-check-url-string/
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls = re.findall(regex, text)
    semanticsDict[key].append(len(urls))


def addSymbols(key, semanticsDict, text):
    """Add Symbols to features

    Args:
        semanticsDict: Pandas dataframe
        key: Name of features
        tokenizedText: Text to generate features from
    """
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
    """Add Word2Vec to features

    Args:
        semanticsDict: Pandas dataframe
        key: Name of features
        tokenizedText: Text to generate features from
    """
    print("TokenText", text)


def initSemantics():
    """Add semantics to features

    Args:
        semanticsDict: Pandas dataframe
        key: Name of features
        tokenizedText: Text to generate features from
    """
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

    if ADD_WORD2VEC:
        semanticsDict['titleWord2Vec'] = []
        semanticsDict['textWord2Vec'] = []

    return semanticsDict


def wordPlot(wordDict, title, savefile):
    """
    Plot most occuring words to wordDict title and write it to file
    Args:
        wordDict: Dictionary with words occuring in text
        title: title as input
        savefile: File to save figure to
    """
    wordDict = dict(sorted(wordDict.items(), key=lambda item: item[1], reverse=True))
    items = {k: wordDict[k] for k in list(wordDict)[:10]}
    plt.ylabel("Occurence")
    plt.title(title)
    plt.bar(range(len(items)), items.values(), align="center")

    plt.xticks(range(len(items)), list(items.keys()), rotation=20)
    plt.savefig(savefile)
    plt.show()


def print_not_frequent_words(wordDict, title, savefile):
    """
    Generate dictionary of non frequent words, and plot them
    Args:
        wordDict: Dictionary with words occuring in text
        title: title as input
        savefile: File to save figure to
    """
    wordDict = dict(sorted(wordDict.items(), key=lambda item: item[1], reverse=False))
    word_occurences = {}
    for word in wordDict:
        count = wordDict[word]
        if count not in word_occurences:
            word_occurences[count] = 1
        else:
            word_occurences[count] += 1

    counts = []
    entries = []

    total_entries = sum(word_occurences.values())
    previous_entries_amount = 0
    for occurence in word_occurences.items():
        counts.append(occurence[0])
        next = occurence[1] / total_entries
        entries.append(previous_entries_amount + next)
        previous_entries_amount += next
    plt.plot(counts, entries)

    plt.axvline(x=5, color='r', label='axvline - full height')

    plt.title(title)
    plt.xlabel('Counts per word')
    plt.ylabel('Percentage of word inclusion for given count')
    plt.xscale('log')
    plt.savefig(savefile)
    plt.show()


def createHeatmap(df, featureAmount=20, fileName="heatmap.pdf", labelName='label'):
    """
    Create a heatmap of the correlation of the features in df
    Args:
        df: Panda's dataframe to calculate the correlations
        featureAmount (int, optional): Top n features with the highest correlation to labelName 'label'. Defaults to 20.
        fileName (str, optional): Filename to save the heatmap to. Defaults to "heatmap.pdf".
        labelName (str, optional): Label to select the n highest correlating features from. Defaults to 'label'.
    """
    all_correlations = df.corr()
    top_n = all_correlations[labelName].abs().sort_values(ascending=False)
    top_n_names = top_n[0:featureAmount+1].index.values
    df_n = df[top_n_names]

    correlations = df_n.corr()
    top_corr_features = correlations.index
    plt.figure(figsize=(featureAmount+1, featureAmount+1))
    # plot heat map
    g = svm.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    fig = g.get_figure()
    fig.savefig(fileName, dpi=400, bbox_inches='tight')


def wordCount(wordDict, text):
    """
    Add the occurences in words in text to wordDict
    Args:
        wordDict: Dictionary to add the occurences of text to
        text: Text to process
    """
    for word in text:
        if word not in stopwords:
            if word in wordDict:
                wordDict[word] += 1
            else:
                wordDict[word] = 1


def split_into_sentences(body):
    """
    Split the body of text into sentences. This is required for n-gram context processing
    Args:
        body (str): Text to split into sentences

    Returns:
        List[str]: List of sentences found in body
    """
    return [re.sub('[!@#$:;’”,."()_\'–-…/]', '', x).split() for x in re.split('[.!?]+ ', str(body).lower())]


def addFeatures(df):
    """
    Add all required features to the dataframe
    Args:
        df: Panda dataframe to add features to
    """
    wordDict = [dict(), dict()]
    semanticsDict = initSemantics()

    title_word2vec = None
    text_word2vec = None
    title_word_labels = None
    text_word_labels = None
    title_similar_words = {}
    text_similar_words = {}
    if ADD_WORD2VEC:
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

            if ADD_WORD2VEC:
                get_most_similar_label('titleWord2Vec', semanticsDict, title_similar_words, row['title'], title_word2vec, title_word_labels, 10)
                get_most_similar_label('textWord2Vec', semanticsDict, text_similar_words, row['text'], text_word2vec, text_word_labels, 10)

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

        print_not_frequent_words(wordDict[0], 'Word count inclusions for reliable news', 'reliable_counts.pdf')
        print_not_frequent_words(wordDict[1], 'Word count inclusions for unreliable news', 'unreliable_counts.pdf')

    for key in semanticsDict:
        df[key] = semanticsDict[key]


def get_most_similar_label(key, semanticDict, similar_words_store, body, word2vec, label_prob, topn=20):
    """
    Get the most similar label of the word2vec for context dependent text processing
    Args:
        key: Key to save the label to
        semanticDict: pandas dataframe to add the features to
        similar_words_store: Store of similar words
        body: Body of text to process
        word2vec: Word2Vec processor 
        label_prob: Label probability
        topn (int, optional): top n similar word vectors to process. Defaults to 20.
    """
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
                            label_prob[similar_word_str][0] + label_prob[similar_word_str][1]) * similar_word_prob
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
    """
    Create word2vec model from dataframe
    Args:
        df: Pandas dataframe
    """
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
    """
    Run all models on dataframe X (features) and labels y
    Args:
        X: Dataframe of features
        y: Dataframe of results
    """
    X_ids = X.columns.values
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.33, random_state=1)

    models = [
        ["Logistic regression: ", LogisticRegression(solver='liblinear', multi_class='ovr'), False],
        ["MLPClassifier: ", MLPClassifier(), False],
        ["Linear Discriminant: ", LinearDiscriminantAnalysis(), False],
        ["KNeighbourClassifier: ", KNeighborsClassifier(), False],
        ["Decision Tree:", DecisionTreeClassifier(), False],
        ["Gaussian: ", GaussianNB(), False],
        ["SVC: ", SVC(max_iter=10000, gamma='auto'), False],
        ["Random Forest: ", RandomForestClassifier(random_state=0), True]
    ]

    # Just ignore the converge warning when the dataset is to small
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for model in models:
        model[1].fit(X_train, Y_train)
        predictions = model[1].predict(X_validation)
        predData = np.c_[predictions, X_validation]
        columnNames = np.insert(X_ids, 0, 'pred-label', axis=0)
        pdf = pd.DataFrame(predData, columns=columnNames)

        if CREATE_HEATMAP:
            createHeatmap(pdf, fileName=f"{model[0][:-2].lower().replace(' ', '-')}-heatmap.pdf", labelName='pred-label')

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


def toFile(df, fileName):
    """
    Save dataframe df1 to a file called fileName
    Args:
        df1: Pandas dataframe
        fileName: Filename
    """
    df.to_csv("./data/" + fileName)


def mergeFiles():
    """
    Load and merge all datasets used for our application (ds1, ds2, and ds3)

    Returns:
        _type_: Dataframe containing all data
    """

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


def run():
    """
    Run our application
    """
    print("Reading in datafile...")

    if LOAD_DATA_FROM_FILE:
        scaled_df = pd.read_csv("./data/dt-all-features.csv")
    else:
        df = mergeFiles()
        print(df)

        # Create the word 2 vec model
        if CREATE_WORD2VEC_MODEL:
            print("Creating word2vec model...")
            create_word2vec_models(df)

        # Just take a smaller part of the dataset
        print("Adding semantics to the dataframe...")
        addFeatures(df)

        df.drop(["title", "text"], axis=1, inplace=True)

        # Normalize the matrix
        scaler = MinMaxScaler()
        scaler.fit(df)
        scaled = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled, columns=df.columns)

        print(scaled_df)

        toFile(scaled_df, "baseline-features.csv")

    scaled_df = scaled_df.drop("Unnamed: 0", axis=1)
    X = scaled_df.drop(['label'], axis=1)
    y = scaled_df['label']
    print("Running the models...")

    if CREATE_HEATMAP:
        createHeatmap(scaled_df, fileName="features-heatmap.pdf")
    runModels(X, y)


if __name__ == '__main__':
    """
    Main function used for running the application.
    This function first download the required data for nltk, and then calls the run() function.
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    run()
