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

