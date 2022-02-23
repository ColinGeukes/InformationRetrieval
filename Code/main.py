import csv
import math

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
            DCG = CG[i-1] + list[i] / math.log(i, b)
    # print(DCG)

    return DCG


def openFile():
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
        for value in dict1[key][0:DOCUMENTS_PER_QUERY]:
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
            print("Key: " + key + " NDCG: " + str(NDCG))
            print(ratingList)
            print(str(I) + "\n")



if __name__ == '__main__':
    openFile()
