python babify_data.py ~/DATA/RATNAPARKHI/PPAttachData/training.clean  > train.pp.babi
python babify_data.py ~/DATA/RATNAPARKHI/PPAttachData/test.clean > test.pp.babi
cp *.pp.babi ../../../codedump/CH6/FLOYD/data/
