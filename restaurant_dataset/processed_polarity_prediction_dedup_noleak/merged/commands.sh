python merge.py --sentences ../train_sentences.txt  --targets ../train_targets.txt --polarities ../train_labels.txt --output train_data.txt
python merge.py --sentences ../valid_sentences.txt  --targets ../valid_targets.txt --polarities ../valid_labels.txt --output valid_data.txt
python merge.py --sentences ../test_sentences.txt  --targets ../test_targets.txt --polarities ../test_labels.txt --output test_data.txt

