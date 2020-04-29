from glob import glob

with open('train_sentences.txt', 'r') as f: train_sentences = f.readlines()
with open('test_sentences.txt', 'r') as f: test_sentences = f.readlines()
with open('valid_sentences.txt', 'r') as f: valid_sentences = f.readlines()

with open('train_targets.txt', 'r') as f: train_targets = f.readlines()
with open('test_targets.txt', 'r') as f: test_targets = f.readlines()
with open('valid_targets.txt', 'r') as f: valid_targets = f.readlines()

with open('train_valid_targets.txt', 'r') as f: train_valid_targets = f.readlines()
with open('train_valid_sentences.txt', 'r') as f: train_valid_sentences = f.readlines()

train = set(zip(train_sentences, train_targets))
valid = set(zip(valid_sentences, valid_targets))
test = set(zip(test_sentences, test_targets))

train_valid = set(zip(train_valid_sentences, train_valid_targets))

print(len(train.intersection(test)))

print(len(train.intersection(valid)))

print(len(train_valid.intersection(test)))

