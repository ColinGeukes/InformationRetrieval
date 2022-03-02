import csv
import json
import os

collection_dir = "/home/colingeukes/Documents/projects/InformationRetrieval/CoreIR/Anserini/collections/msmarco-passage/collection_jsonl"
run_feature_files = [
    "./data/run.marco-test2019-queries-BM25-default.tsv",
    "./data/run.marco-test2019-queries-BM25-default-RM3.tsv",
    "./data/run.marco-test2019-queries-BM25-optimized-AXIOM-MAP.tsv",
    "./data/run.marco-test2019-queries-BM25-optimized-AXIOM-Recall.tsv",
    "./data/run.marco-test2019-queries-BM25-optimized-RM3.tsv"
]


def generate_collection_features():
    # Retrieve the doc files
    onlyfiles = [f for f in os.listdir(collection_dir) if os.path.isfile(os.path.join(collection_dir, f))]
    onlyfiles.sort()
    print("Getting features of collection: ", onlyfiles)

    # Loop through collection
    document_features = {}
    for document in onlyfiles:
        generate_document_features(collection_dir + "/" + document, document_features)

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

            print(features)

            # Append the data
            data.append('%s qid:%s %s' % (document_id, query_id, " ".join(features)))

    # Write the data
    with open('features_full.tsv', 'w') as f:
        for row in data:
            f.write("%s\n" % row)


def generate_run_features(run_file, run_features, name):
    print("Getting document features: ", run_file)
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

    # TODO: WE need to normalize from 0 to 1!


def generate_document_features(file, features):
    print("Getting document features: ", file)

    feature_base = len(run_feature_files) + 1

    # Open the doc and read line by line the json.
    file1 = open(file, 'r')
    lines = file1.readlines()
    for line in lines:
        # Retrieve the document
        row = json.loads(line)
        contents = row["contents"]

        # Create the features for the document
        features[row["id"]] = {
            # Store the document length
            feature_base: len(contents.split(" "))
            # Store the stopword percentage
        }


if __name__ == '__main__':
    generate_collection_features()
