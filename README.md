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
  -bm25 -bm25.k1 0.0.82 -bm25.b 0.68
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
  -bm25 -bm25.k1 0.0.60 -bm25.b 0.62
```

```bash
tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 ./2019qrels-pass.txt runs/run.marco-test2019-queries-tuned2.tsv
```



Run: 

## RankLib (Learning to Rank)
Download 

RankLib: https://sourceforge.net/projects/lemur/files/lemur/
Qrels Dev: https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv
Qrels Train: https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv

Training:

java -jar ./RankLib-2.17.jar -train ./qrels.train.tsv -test ./qrels.dev.tsv -validate ./2019qrels-pass.txt -ranker 6 -metric2t NDCG@10 -metric2T ERR@10 -save mymodel.txt





