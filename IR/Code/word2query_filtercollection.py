import csv

def filtersrccollection():
    # Get the ranked queries for a given
    query_rankings_file = csv.reader(open("./2019qrels-pass.txt"), delimiter=" ")
    
    
    doc_ids = set()
    for row in query_rankings_file:
        doc_ids.add(row[2])
   
   
    index_rows = []
    col_rows = []
   
   
    with open("./src-collection.txt") as f:
        lines = f.readlines()

    index = 0
    for line in lines:
        index = index + 1
        if (str(index) in doc_ids):
            index_rows.append(index)
            col_rows.append(line)
    
    
    with open("filtered-src-collection.txt", 'w') as f:
        for row in col_rows:
            f.write(f"{row}")

    with open("docindexes.txt", 'w') as f:
        for index in index_rows:
            f.write(f"{index}\n")

if __name__ == '__main__':
    filtersrccollection()