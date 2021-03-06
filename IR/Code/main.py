import csv

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
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




def retrieveRelevanceDocuments(ids, length=50):
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
    query_rankings_file = csv.reader(open("data/run.marco-test2019-queries-BM25-default.tsv"), delimiter=" ")

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


def retrieveRelevanceDocumentsLTR(ids, length=50):
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
    query_rankings_file = csv.reader(open("./models/reranking/myNewRankedLists.txt"), delimiter=" ")

    for row in query_rankings_file:
        if row[0] in ids and len(ids[row[0]]) <= length:
            passage_id = row[2].split("=")[1]

            # Create the entry
            entry = {
                'passage': row[2].split("=")[1],
                'rank': row[3],
                'score': row[4],
                'relevance': -1
            }

            # Find the relevance, if it is known.
            if row[0] in query_relevances and passage_id in query_relevances[row[0]]:
                entry['relevance'] = int(query_relevances[row[0]][passage_id])

            # Append the entry.
            ids[row[0]].append(entry)
    return ids


def bead_plot(relevant_docs, title, output_file):
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
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(data, aspect=1.4, vmin=-1, vmax=3, interpolation='none')

    plt.yticks([0, 1, 2], y_labels, rotation='0')
    plt.ylabel("Queries")

    # The x ticks are hardcoded..
    plt.xticks([0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49], [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.xlabel("Retrieved rank", loc='center')

    ax.set_title(title)
    fig.tight_layout()

    # Show to colormap
    cmap = mpl.cm.viridis
    bounds = np.linspace(-1, 4, 5)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    divider = make_axes_locatable(ax)
    colorbar_axes = divider.append_axes("right",
                                        size="2%",

                                        pad=-1.2)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=bounds,
                        orientation='vertical', cax=colorbar_axes, label='relevance')
    cbar.set_ticks([-0.5, 0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(['?', '0', '1', '2', '3'])

    plt.show()
    fig.savefig(output_file, bbox_inches='tight')


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

def bm25_ndcg(docs_per_query=DOCUMENTS_PER_QUERY):
    tsv_file = open("data/run.marco-test2019-queries.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    lines = []
    with open('data/2019qrels-pass.txt') as f:
        lines = f.readlines()
    # print(lines)
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
    # print(dict1)
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
            # print("Key: " + key + " NDCG: " + str(NDCG))
            # print(ratingList)
            # print(str(I) + "\n")
    NDCGList.sort(key=lambda x: x[0])
    return NDCGList
    # print(NDCGList)
    # print(dict2)
    # return dict2

def l2r_ndcg(docs_per_query=DOCUMENTS_PER_QUERY):
    tsv_file = open("models/reranking/myNewRankedLists.txt")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    lines = []
    with open('data/2019qrels-pass.txt') as f:
        lines = f.readlines()
    # print(lines)
    list1 = []
    for line in lines:
        line = line.split()

        queryId = line[0]
        # print(queryId)
        if queryId not in list1:
            list1.append(queryId)

    for row in read_tsv:
        row = row[0].split()
        queryId = row[0]
        docId = int(row[2].replace("docid=",""))
        # print(docId)
        if queryId not in dict1 and queryId in list1:
            dict1[queryId] = []
        if queryId in list1:
            dict1[queryId].append(docId)
    # print(dict1)
    for key in dict1:
        for value in dict1[key][0:docs_per_query]:
            for line in lines:
                line = line.split()
                # print(line)

                queryId = line[0]
                docId = line[2]
                rating = line[3]
                if queryId not in dict2:
                    dict2[queryId] = []

                # print(value == docId)
                # print(rating)
                if queryId == key and int(value) == int(docId):
                    # print("Key: " + str(key) + " docId: " + str(docId))
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
            # print("Key: " + key + " NDCG: " + str(NDCG))
            # print(ratingList)
            # print(str(I) + "\n")
    NDCGList.sort(key=lambda x: x[0], reverse=True)
    # print(NDCGList)
    return NDCGList

def draw_ndcg_plot(ndcg_list):
    query_dict = {}
    query_rankings_file = csv.reader(open("../../Anserini/msmarco-test2019-queries.tsv"))
    for row in query_rankings_file:
        row = row[0].split("\t")
        query_dict[row[0]] = row[1]

    print(query_dict)
    y_labels = []
    for item in ndcg_list:
        y_labels.append(query_dict[item[0]])


    print(y_labels)
    fig = plt.figure(figsize=(15,10), dpi=300)

    y = range(42)

    for i in range(1, 43):
        bm25 = ndcg_list[i - 1][1]
        l2r = ndcg_list[i - 1][2]
        if bm25 < l2r:
            plt.plot([0, bm25], [i - 1, i - 1], color='k')
            plt.plot([bm25, bm25], [i - 1, i - 1], marker='x', color='r')
            plt.plot([bm25, l2r], [i - 1, i - 1], linestyle='dashed', color='k')
            plt.plot([l2r, l2r], [i - 1, i - 1], marker='o', color='b')
        else:
            plt.plot([0, l2r], [i - 1, i - 1], color='k')
            plt.plot([l2r, l2r], [i - 1, i - 1], marker='o', color='b')
            plt.plot([l2r, bm25], [i - 1, i - 1], linestyle='dashed', color='k')
            plt.plot([bm25, bm25], [i - 1, i - 1], marker='x', color='r')
            # plt.plot([bm25, 1], [i - 1, i - 1], marker='x')



    plt.yticks(y, y_labels, rotation='horizontal')

    blue_star = mlines.Line2D([], [], color='b', marker='o', linestyle='None',
                              markersize=10, label='L2R')
    red_square = mlines.Line2D([], [], color='r', marker='x', linestyle='None',
                               markersize=10, label='BM25')

    plt.legend(handles=[blue_star, red_square], loc='upper right')
    plt.margins(y = 0.1)
    plt.xlabel("NDCG@10 score")
    fig.subplots_adjust(left=0.3)
    plt.savefig("ndcg.pdf")
    plt.show()

if __name__ == '__main__':
    # retrieveFeatureFileFormatBadJson()
    # generateLearningToRankFormat()
    bm25_ndcg_list = bm25_ndcg()
    l2r_ndcg_list = l2r_ndcg()
    # print(bm25_ndcg_list)
    # print(len(bm25_ndcg_list))
    # print(l2r_ndcg_list)
    # print(len(l2r_ndcg_list))

    ndcg_list = []
    for i in range(len(bm25_ndcg_list)):
        difference = bm25_ndcg_list[i][1] - l2r_ndcg_list[i][1]
        ndcg_list.append([bm25_ndcg_list[i][0], bm25_ndcg_list[i][1], l2r_ndcg_list[i][1], difference])
    ndcg_list.sort(key=lambda x: x[3])
    print(ndcg_list)
    draw_ndcg_plot(ndcg_list)

    bead_plot(retrieveRelevanceDocuments({
        '1113437': [],
        '19335': [],
        '183378': []}),
        "Beadplot Baseline Relevance Documents for Queries",
        "beadplot_baseline.pdf")

    bead_plot(retrieveRelevanceDocumentsLTR({
        '1113437': [],
        '19335': [],
        '183378': []}),
        "Beadplot Learning to Rank Documents for Queries",
        "beadplot_ltr.pdf"
    )
