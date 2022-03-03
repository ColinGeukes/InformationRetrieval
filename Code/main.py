import csv

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

DOCUMENTS_PER_QUERY = 10
dict = {}
dict1 = {}
dict2 = {}


def computeCG(list):
    CG = []

    CG.append(list[0])

    for i in range(1, DOCUMENTS_PER_QUERY):
        sum = CG[i - 1] + list[i]
        CG.append(sum)
    return CG


def computeDCG(list, CG):
    b = 2
    DCG = 0

    for i in range(0, DOCUMENTS_PER_QUERY):
        if i < b:
            DCG = CG[i]
        else:
            DCG = CG[i - 1] + list[i] / math.log(i, b)
    # print(DCG)

    return DCG


def openFile(docs_per_query=DOCUMENTS_PER_QUERY):
    tsv_file = open("data/run.marco-test2019-queries.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    lines = []
    with open('data/2019qrels-pass.txt') as f:
        lines = f.readlines()

    list1 = []
    for line in lines:
        line = line.split()

        queryId = line[0]

        if queryId not in list1:
            list1.append(queryId)

    for row in read_tsv:
        row = row[0].split()
        queryId = row[0]
        docId = row[2]
        if queryId not in dict1 and queryId in list1:
            dict1[queryId] = []
        if queryId in list1:
            dict1[queryId].append(docId)

    for key in dict1:
        for value in dict1[key][0:docs_per_query]:
            for line in lines:
                line = line.split()

                queryId = line[0]
                docId = line[2]
                rating = line[3]
                if queryId not in dict2:
                    dict2[queryId] = []

                if queryId == key and value == docId:
                    # print("Key: " + str(key) + " docId:" + str(docId))
                    dict2[queryId].append(int(rating))

    NDCGList = []
    for key in dict2:
        ratingList = dict2[key]
        I = ratingList.copy()
        I.sort(reverse=True)
        DCG = computeDCG(ratingList, computeCG(ratingList))
        DCGI = computeDCG(I, computeCG(I))

        if DCGI == 0:
            print("ISZERO")
        else:
            NDCG = DCG / DCGI
            NDCGList.append([key, NDCG])
            print("Key: " + key + " NDCG: " + str(NDCG))
            print(ratingList)
            print(str(I) + "\n")
    NDCGList.sort(key=lambda x: x[1])
    print(NDCGList)
    # return dict2


def retrieveRelevanceDocuments(ids, length):
    # Retrieve relevances.
    query_relevances_file = csv.reader(open("data/2019qrels-pass.txt"), delimiter=" ")
    query_relevances = {}
    for row in query_relevances_file:
        # Add the query id
        if row[0] not in query_relevances:
            query_relevances[row[0]] = {}

        # Add the snippet relevance
        query_relevances[row[0]][row[2]] = row[3]

    # Get the ranked queries for a given
    query_rankings_file = csv.reader(open("data/run.marco-test2019-queries.tsv"), delimiter=" ")

    for row in query_rankings_file:
        if row[0] in ids and int(row[3]) <= length:
            # Create the entry
            entry = {
                'passage': row[2],
                'rank': row[3],
                'score': row[4],
                'relevance': -1
            }

            # Find the relevance, if it is known.
            if row[0] in query_relevances and row[2] in query_relevances[row[0]]:
                entry['relevance'] = int(query_relevances[row[0]][row[2]])

            # Append the entry.
            ids[row[0]].append(entry)
    return ids


def beadPlot(ids, length=100):
    # Retrieve ids and relevances
    relevant_docs = retrieveRelevanceDocuments(ids, length)

    # Get array list of relevances
    data = []
    y_labels = []
    for query_id in relevant_docs:
        y_labels.append(str(query_id))
        relevance_array = []
        for document in relevant_docs[query_id]:
            relevance_array.append(document['relevance'])
        data.append(relevance_array)

    # Plot the beadplot / heatmap.
    # a = np.random.random((1, length))
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(data, aspect=1.4)

    plt.yticks([0, 1, 2], y_labels, rotation='0')
    plt.ylabel("Queries")

    # The x ticks are hardcoded..
    plt.xticks([0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49], [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.xlabel("Retrieved rank", loc='center')

    ax.set_title("Beadplot relevance documents for queries")
    fig.tight_layout()

    # Show to colormap
    cmap = mpl.cm.viridis
    # bounds = [-1, 0, 1, 2, 3]
    # bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    bounds = np.linspace(-1, 4, 6)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    divider = make_axes_locatable(ax)
    colorbar_axes = divider.append_axes("right",
                                        size="2%",

                                        pad=-1.2)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=bounds,
                        orientation='vertical', cax=colorbar_axes, label='relevance')
    cbar.set_ticks([-0.5, 0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(['?', '0', '1', '2', '3'])

    # , cax=fig.add_axes([0.9, 0.06, 0.02, 0.8]))

    plt.show()
    fig.savefig("beadplot.pdf", bbox_inches='tight')


def generateLearningToRankFormat():
    # Retrieve the run file.
    query_rankings_file = csv.reader(open("data/run.marco-test2019-queries.tsv"), delimiter=" ")

    # Store the data into the LETOR format
    data = []
    for row in query_rankings_file:
        # < target > qid: < qid > < feature >: < value > < feature >: < value > ... < feature >: < value >  # <info>
        data.append('%s qid:%s 1:%s 2:%s' % (row[2], row[0], row[3], row[4]))

    # Write the data
    with open('training.tsv', 'w') as f:
        for row in data:
            f.write("%s\n" % row)


def retrieveFeatureFileFormat():
    # Get the ranked queries for a given
    query_rankings_file = csv.reader(open("data/run.marco-test2019-queries.tsv"), delimiter=" ")

    data = {}

    for row in query_rankings_file:
        if row[0] not in data:
            data[row[0]] = [row[2]]
        else:
            data[row[0]].append(row[2])

    # Store the data into the LETOR format
    formattedData = []
    formattedData.append('[')
    index = 0
    for row in data:
        # Increment index
        index += 1

        formattedData.append('{')
        formattedData.append('\t"qid": "%i",' % int(row))

        # Format the array
        formattedData.append('\t"docIds": ["%s"]' % '", "'.join(data[row]))

        if index < len(data):
            formattedData.append('},')
        else:
            formattedData.append('}')

    formattedData.append(']')

    with open('featureExtraction.json', 'w') as f:
        for row in formattedData:
            f.write("%s\n" % row)


def retrieveFeatureFileFormatBadJson():
    # Get the ranked queries for a given
    query_rankings_file = csv.reader(open("data/run.marco-test2019-queries.tsv"), delimiter=" ")

    data = {}

    for row in query_rankings_file:
        if row[0] not in data:
            data[row[0]] = [row[2]]
        else:
            data[row[0]].append(row[2])

    # Store the data into the LETOR format
    formattedData = []
    index = 0
    for row in data:
        # Increment index
        index += 1

        if index < len(data):
            formattedData.append('{"qid": "%i", "docIds": ["%s"]},' % (int(row), '", "'.join(data[row])))
        else:
            formattedData.append('{"qid": "%i", "docIds": ["%s"]}' % (int(row), '", "'.join(data[row])))

    with open('featureExtraction.json', 'w') as f:
        for row in formattedData:
            f.write("%s\n" % row)


if __name__ == '__main__':
    # retrieveFeatureFileFormatBadJson()
    # generateLearningToRankFormat()
    openFile()
    # beadPlot({
    #     '1113437': [],
    #     '19335': [],
    #     '183378': []})
