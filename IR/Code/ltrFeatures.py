import csv
import json
import os
import random
import re
import time

collection_dir = "../../Anserini/collections/msmarco-passage/collection_jsonl"
run_feature_files = [
    "./data/run.marco-test2019-queries-BM25-default.tsv",
    "./data/run.marco-test2019-queries-BM25-default-RM3.tsv",
    "./data/run.marco-test2019-queries-BM25-optimized-AXIOM-MAP.tsv",
    "./data/run.marco-test2019-queries-BM25-optimized-AXIOM-Recall.tsv",
    "./data/run.marco-test2019-queries-BM25-optimized-RM3.tsv"
]
stop_words = "./data/stopwords.txt"


def retrieve_stop_words(file=stop_words):
    file1 = open(file, 'r')
    lines = file1.readlines()

    stop_word_set = set()
    for line in lines:
        stop_word_set.add(line.strip())
    return stop_word_set


def generate_collection_features():
    # Retrieve the doc files
    onlyfiles = [f for f in os.listdir(collection_dir) if os.path.isfile(os.path.join(collection_dir, f))]
    onlyfiles.sort()
    print("Getting features of collection: ", onlyfiles)

    # Retrieve stop words
    stop_word_set = retrieve_stop_words()

    # Loop through collection
    document_features = {}
    for document in onlyfiles:
        generate_document_features(collection_dir + "/" + document, document_features, stop_word_set)

    print("Get run features:", run_feature_files)
    run_features = {}
    name = 1
    for run_file in run_feature_files:
        generate_run_features(run_file, run_features, str(name))
        name += 1

    # Save the features
    data = []
    for document_id in run_features:

        # Retrieve document specific features
        doc_features = []
        for doc_feature in document_features[document_id]:
            doc_features.append("%s:%s" % (doc_feature, document_features[document_id][doc_feature]))

        for query_id in run_features[document_id]:
            # The runs are named with their feature (1, 2... n.)
            features = []
            for feature in run_features[document_id][query_id]:
                features.append("%s:%s" % (feature, run_features[document_id][query_id][feature]))

            # Extend the features with document specific features
            features.extend(doc_features)

            # Append the data
            data.append('%s qid:%s %s' % (document_id, query_id, " ".join(features)))

    # Write the data
    with open('features_full.tsv', 'w') as f:
        for row in data:
            f.write("%s\n" % row)


def generate_run_features(run_file, run_features, name):
    print("Retrieve run features: ", run_file)
    run_data = csv.reader(open(run_file), delimiter=" ")

    for row in run_data:
        # Check if we have the document first
        query = row[0]
        document = row[2]
        score = row[4]

        # If we do not have a reference of document yet.
        if document not in run_features:
            run_features[document] = {
                query: {
                    name: score
                }
            }

        # If we do not have a reference of the query yet.
        elif query not in run_features[document]:
            run_features[document][query] = {
                name: score
            }

        # Just add score, already have an entry.
        else:
            run_features[document][query][name] = score


def generate_document_features(file, features, stop_word_set):
    print("Getting document features: ", file)

    feature_base = len(run_feature_files) + 1

    # Open the doc and read line by line the json.
    file1 = open(file, 'r')
    lines = file1.readlines()
    for line in lines:
        # Retrieve the document
        row = json.loads(line)
        contents = row["contents"]

        # Get document length (also split on dots and commas
        # words = re.split('. |, | ', contents)
        content_without_punctuation = re.sub(r'[^\w\s]', '', contents)
        words = content_without_punctuation.split(" ")

        # Remove empty strings
        words = list(filter(lambda word: len(word) > 1, words))

        # Retrieve the doc length
        doc_length = len(words)

        # Retrieve stop word percentage
        stop_word_percentage = 0
        if doc_length > 0:
            stop_word_percentage = get_stop_word_amount(words, stop_word_set) / doc_length

        # Create the features for the document
        features[row["id"]] = {
            # Store the document length
            feature_base: doc_length,
            # Store the stop word percentage
            feature_base + 1: stop_word_percentage

        }


def get_stop_word_amount(words, stop_word_set):
    hits = 0
    for word in words:
        if word.lower() in stop_word_set:
            hits += 1
    return hits


def normalize_features_full():
    file1 = open('features_full.tsv', 'r')
    rows = file1.readlines()

    # Retrieve the min and max value of a feature
    features_min_max = {}
    for row in rows:
        row_split = row.split(" ")

        for i in range(2, len(row_split)):
            feature_split = row_split[i].split(":")
            feature_name = feature_split[0]
            feature_value = float(feature_split[1].strip())

            # If feature not yet exists
            if feature_name not in features_min_max:
                features_min_max[feature_name] = {
                    "min": min(0., feature_value),
                    "max": feature_value,
                }

            # Update min and max
            else:
                features_min_max[feature_name]["min"] = min(features_min_max[feature_name]["min"], feature_value)
                features_min_max[feature_name]["max"] = max(features_min_max[feature_name]["max"], feature_value)

    # Normalize the features
    normalized_data = []
    for row in rows:
        row_split = row.split(" ")
        feature_space = [0.] * len(features_min_max)

        for i in range(2, len(row_split)):
            feature_split = row_split[i].split(":")
            feature_name = int(feature_split[0])
            feature_value = float(feature_split[1].strip())

            feature_min = features_min_max[str(feature_name)]["min"]
            feature_max = features_min_max[str(feature_name)]["max"]
            feature_space[feature_name - 1] = (feature_value - feature_min) / (feature_max - feature_min)

        feature_space_format = []
        for i in range(0, len(feature_space)):
            feature_space_format.append(str(i + 1) + ":" + str(feature_space[i]))

        normalized_data.append('%s %s %s' % (row_split[0], row_split[1], " ".join(feature_space_format)))

    # Write the data
    with open('features_full_norm.tsv', 'w') as f:
        for row in normalized_data:
            f.write("%s\n" % row)


def split_files():
    print("Splitting files into training, test and vallidation")
    file1 = open('features_full_norm_relevance.tsv', 'r')
    rows = file1.readlines()
    random.shuffle(rows)

    training = rows[0:int(len(rows) * 0.6)]
    test = rows[int(len(rows) * 0.6): int(len(rows) * 0.8)]
    validation = rows[int(len(rows) * 0.8):len(rows)]

    with open('features_full_norm_training.tsv', 'w') as f:
        for row in training:
            f.write("%s" % row)
    with open('features_full_norm_test.tsv', 'w') as f:
        for row in test:
            f.write("%s" % row)
    with open('features_full_norm_validation.tsv', 'w') as f:
        for row in validation:
            f.write("%s" % row)


def match_relevance_to_file():
    print("Match the relevance to query and document")

    # Read the normalized and store them for quick access
    file1 = open('features_full_norm.tsv', 'r')
    rows = file1.readlines()

    # Retrieve the min and max value of a feature
    features_data = {}
    for row in rows:
        row_split = row.strip().split(" ")
        doc_id = row_split[0]
        query_id = row_split[1].split(":")[1]

        # If we do not track it yet, add a new entry
        if doc_id not in features_data:
            features_data[doc_id] = {
                query_id: " ".join(row_split[2:])
            }
        else:
            # We already track it, so expend.
            features_data[doc_id][query_id] = " ".join(row_split[2:])

    # Retrieve a map of the relevance
    query_relevance_file = csv.reader(open("data/2019qrels-pass.txt"), delimiter=" ")
    relevant_matches = []
    for row in query_relevance_file:
        query_id = row[0]
        doc_id = row[2]

        # Check if we have a match
        if doc_id in features_data and query_id in features_data[doc_id]:
            relevant_matches.append(
                '%s qid:%s %s # docid=%s' % (row[3], query_id, features_data[doc_id][query_id], doc_id))

    # Write the data
    with open('features_full_norm_relevance.tsv', 'w') as f:
        for row in relevant_matches:
            f.write("%s\n" % row)


if __name__ == '__main__':
    start = time.time()
    generate_collection_features()
    normalize_features_full()
    match_relevance_to_file()
    split_files()

    end = time.time()
    print("Total run time: ", end - start)
