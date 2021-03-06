printf "Run Experiment 1:\n"
../../Anserini/target/appassembler/bin/SearchCollection -parallelism 6 -hits 20000\
  -index ../../Anserini/indexes/msmarco-passage/lucene-index-msmarco/ \
  -topics ../../Anserini/msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./data/run.marco-test2019-queries-BM25-default.tsv \
  -bm25 -bm25.k1 0.90 -bm25.b 0.60

printf "\n\nRun Experiment 2:\n"
../../Anserini/target/appassembler/bin/SearchCollection -parallelism 6 -hits 20000\
  -index ../../Anserini/indexes/msmarco-passage/lucene-index-msmarco/ \
  -topics ../../Anserini/msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./data/run.marco-test2019-queries-BM25-default-RM3.tsv \
  -bm25 -bm25.k1 0.90 -bm25.b 0.60 -rm3

printf "\n\nRun Experiment 3:\n"
../../Anserini/target/appassembler/bin/SearchCollection -parallelism 6 -hits 20000\
  -index ../../Anserini/indexes/msmarco-passage/lucene-index-msmarco/ \
  -topics ../../Anserini/msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./data/run.marco-test2019-queries-BM25-optimized-RM3.tsv \
  -bm25 -bm25.k1 0.49 -bm25.b 0.60 -rm3

printf "\n\nRun Experiment 4:\n"
../../Anserini/target/appassembler/bin/SearchCollection -parallelism 6 -hits 20000\
  -index ../../Anserini/indexes/msmarco-passage/lucene-index-msmarco/ \
  -topics ../../Anserini/msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./data/run.marco-test2019-queries-BM25-optimized-AXIOM-MAP.tsv \
  -bm25 -bm25.k1 0.48 -bm25.b 0.59 -axiom

printf "\n\nRun Experiment 5:\n"
../../Anserini/target/appassembler/bin/SearchCollection -parallelism 6 -hits 20000\
  -index ../../Anserini/indexes/msmarco-passage/lucene-index-msmarco/ \
  -topics ../../Anserini/msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./data/run.marco-test2019-queries-BM25-optimized-AXIOM-Recall.tsv \
  -bm25 -bm25.k1 0.48 -bm25.b 0.60 -axiom