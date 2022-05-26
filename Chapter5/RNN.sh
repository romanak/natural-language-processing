python qa_rnn_v2.py ~/DATA/bAbI/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt ~/DATA/bAbI/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt 6 > rnn.results

python qa_rnn_v2.py ~/DATA/bAbI/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt ~/DATA/bAbI/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt 8 >> rnn.results

python qa_rnn_v2.py ~/DATA/bAbI/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt ~/DATA/bAbI/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt -1 >> rnn.results
