printf "Run Experiment 1:\n"
../../Anserini/target/appassembler/bin/SearchCollection -parallelism 6 \
  -index ../../Anserini/indexes/msmarco-passage/lucene-index-msmarco-beamsize5/ \
  -topics ../../Anserini/msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./data/run.marco-test2019-queries-beam5-BM25-default.tsv \
  -bm25 -bm25.k1 0.90 -bm25.b 0.60

printf "\n\n Evaluate 1:"
../../Anserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 ./data/2019qrels-pass.txt data/run.marco-test2019-queries-beam5-BM25-default.tsv

printf "\n\nRun Experiment 2:\n"
../../Anserini/target/appassembler/bin/SearchCollection -parallelism 6 \
  -index ../../Anserini/indexes/msmarco-passage/lucene-index-msmarco-beamsize5/ \
  -topics ../../Anserini/msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./data/run.marco-test2019-queries-beam5-BM25-default-RM3.tsv \
  -bm25 -bm25.k1 0.90 -bm25.b 0.60 -rm3

printf "\n\n Evaluate 2:"
../../Anserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 ./data/2019qrels-pass.txt data/run.marco-test2019-queries-beam5-BM25-default-RM3.tsv

printf "\n\nRun Experiment 3:\n"
../../Anserini/target/appassembler/bin/SearchCollection -parallelism 6 \
  -index ../../Anserini/indexes/msmarco-passage/lucene-index-msmarco-beamsize5/ \
  -topics ../../Anserini/msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./data/run.marco-test2019-queries-beam5-BM25-optimized-RM3.tsv \
  -bm25 -bm25.k1 0.49 -bm25.b 0.60 -rm3

printf "\n\n Evaluate 3:"
../../Anserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 ./data/2019qrels-pass.txt data/run.marco-test2019-queries-beam5-BM25-optimized-RM3.tsv

printf "\n\nRun Experiment 4:\n"
../../Anserini/target/appassembler/bin/SearchCollection -parallelism 6 \
  -index ../../Anserini/indexes/msmarco-passage/lucene-index-msmarco-beamsize5/ \
  -topics ../../Anserini/msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./data/run.marco-test2019-queries-beam5-BM25-optimized-AXIOM-MAP.tsv \
  -bm25 -bm25.k1 0.48 -bm25.b 0.59 -axiom

printf "\n\n Evaluate 4:"
../../Anserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 ./data/2019qrels-pass.txt data/run.marco-test2019-queries-beam5-BM25-optimized-AXIOM-MAP.tsv

printf "\n\nRun Experiment 5:\n"
../../Anserini/target/appassembler/bin/SearchCollection -parallelism 6 \
  -index ../../Anserini/indexes/msmarco-passage/lucene-index-msmarco-beamsize5/ \
  -topics ../../Anserini/msmarco-test2019-queries.tsv \
  -topicreader TsvInt \
  -output ./data/run.marco-test2019-queries-beam5-BM25-optimized-AXIOM-Recall.tsv \
  -bm25 -bm25.k1 0.48 -bm25.b 0.60 -axiom

printf "\n\n Evaluate 5:"
../../Anserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m map -c -m recall.1000 ./data/2019qrels-pass.txt data/run.marco-test2019-queries-beam5-BM25-optimized-AXIOM-Recall.tsv
