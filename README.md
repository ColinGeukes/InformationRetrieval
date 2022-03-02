# InformationRetrieval
## Anserini

This guide was based on [this github page](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md). 

### Commands

In order to convert the MS MARCO into anserini jsonnl files:

```bash
python tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path collections/msmarco-passage/collection.tsv \
 --output-folder collections/msmarco-passage/collection_jsonl
```

For indexing (Note that the amount of threads is set to 1 in this case, otherwise it floods my memory. 
):

```bash
sh target/appassembler/bin/IndexCollection -threads 1 -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator -input collections/msmarco-passage/collection_jsonl \
 -index indexes/msmarco-passage/lucene-index-msmarco -storePositions -storeDocvectors -storeRaw 
```

Then execute:

```bash
target/appassembler/bin/SearchCollection -parallelism 6\
  -index indexes/msmarco-passage/lucene-index-msmarco/ \
  -topics src/main/resources/topics-and-qrels/topics.msmarco-passage.dev-subset.txt \
  -topicreader TsvInt \
  -output runs/run.msmarco-passage.bm25-baseline.topics.msmarco-passage.dev-subset.txt \
  -bm25
```

And can be evaluated with:

```bash
tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt runs/run.msmarco-passage.bm25-baseline.topics.msmarco-passage.dev-subset.txt
```
 
For the MSMARCO leaderboard:

```bash
sh target/appassembler/bin/SearchCollection -parallelism 6\
    -index indexes/msmarco-passage/lucene-index-msmarco/ \
    -topics src/main/resources/topics-and-qrels/topics.msmarco-passage.dev-subset.txt \
    -topicreader TsvInt \
    -output runs/run.msmarco-passage.bm25.default.tsv \
    -format msmarco \
    -bm25
```

```bash
python tools/scripts/msmarco/msmarco_passage_eval.py \
    src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt runs/run.msmarco-passage.bm25.default.tsv
```

# Running 200 test queries
Download: https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
Download: https://trec.nist.gov/data/deep/2019qrels-pass.txt 


## Default

```bash
target/appassembler/bin/SearchCollection -parallelism 6\
  -index indexes/msmarco-passage/lucene-index-msmarco/ \
  -topics ./msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./runs/run.marco-test2019-queries-default.tsv \
  -bm25 -bm25.k1 0.90 -bm25.b 0.60
```

```bash
tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 ./2019qrels-pass.txt runs/run.marco-test2019-queries-default.tsv
```

Source for following statements: https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md
## Tuned for recall@1000

```bash
target/appassembler/bin/SearchCollection -parallelism 6\
  -index indexes/msmarco-passage/lucene-index-msmarco/ \
  -topics ./msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./runs/run.marco-test2019-queries-tuned1.tsv \
  -bm25 -bm25.k1 0.90 -bm25.b 0.60 -rm3
```

```bash
tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 ./2019qrels-pass.txt runs/run.marco-test2019-queries-tuned1.tsv
```

## Tuned MRR@10/MAP

```bash
target/appassembler/bin/SearchCollection -parallelism 6\
  -index indexes/msmarco-passage/lucene-index-msmarco/ \
  -topics ./msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./runs/run.marco-test2019-queries-tuned2.tsv \
  -bm25 -bm25.k1 0.90 -bm25.b 0.40
```

```bash
tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 ./2019qrels-pass.txt runs/run.marco-test2019-queries-tuned2.tsv
```



Run: 



## Word2Vec improvement
Initially we followed the following experiment: https://github.com/castorini/anserini/blob/master/docs/experiments-doc2query.md

These are the predicted queries based on our seq2seq model, based on top k sampling with 10 samples for each document in the corpus.
These queries are concatenated in a file, that can be downloaded with the following commands:
Again, all these commands are taken from https://github.com/castorini/anserini/blob/master/docs/experiments-doc2query.md and configured where necessary for our own setup
``` 
# Grab tarball from either one of two sources:
wget https://www.dropbox.com/s/57g2s9vhthoewty/msmarco-passage-pred-test_topk10.tar.gz -P collections/msmarco-passage
wget https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/base/msmarco-passage-pred-test_topk10.tar.gz -P collections/msmarco-passage

# Unpack tarball:
tar -xzvf collections/msmarco-passage/msmarco-passage-pred-test_topk10.tar.gz -C collections/msmarco-passage
```

To validate the collection
```
wc collections/msmarco-passage/pred-test_topk10.txt
```
Should have the following result:
```
 8841823 536425855 2962345659 collections/msmarco-passage/pred-test_topk10.txt
```

The orignal passages collection set can be concatenated with the predicted queries with the following command:
```
python tools/scripts/msmarco/augment_collection_with_predictions.py \
 --collection-path collections/msmarco-passage/collection.tsv \
 --output-folder collections/msmarco-passage/collection_jsonl_expanded_topk10 \
 --predictions collections/msmarco-passage/pred-test_topk10.txt --stride 1
```

This can then be indexed with the following command: 
```bash
sh target/appassembler/bin/IndexCollection -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator -threads 4 \
 -input collections/msmarco-passage/collection_jsonl_expanded_topk10 \
 -index indexes/msmarco-passage/lucene-index-msmarco-expanded-topk10 \
 -storePositions -storeDocvectors -storeRaw
```

We then perform our own retrieval with
```bash
target/appassembler/bin/SearchCollection -parallelism 6  -index indexes/msmarco-passage/lucene-index-msmarco-expanded-topk10/   -topics ./msmarco-test2019-queries.tsv   -topicreader TsvInt   -output ./runs/run.marco-test2019-queries-default.tsv   -bm25 -bm25.k1 0.90 -bm25.b 0.60
```

and then evaluate with 
```bash
tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 ./2019qrels-pass.txt runs/run.marco-test2019-queries-default.tsv
```

## RankLib (Learning to Rank)
Download 

RankLib: https://sourceforge.net/projects/lemur/files/lemur/
Qrels Dev: https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv
Qrels Train: https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv

Training:

java -jar ./RankLib-2.17.jar -train ./qrels.train.tsv -test ./qrels.dev.tsv -validate ./2019qrels-pass.txt -ranker 6 -metric2t NDCG@10 -metric2T ERR@10 -save mymodel.txt





