import random

labels_file = 'train_valid_labels.txt'
words_file = 'train_valid_sentences.txt'
targets_file = 'train_valid_targets.txt'

with open(labels_file) as f:
    all_labels = f.readlines()

with open(words_file) as f:
    all_words = f.readlines()

with open(targets_file) as f:
    all_targets = f.readlines()

dataset = list(zip(all_labels, all_words, all_targets))
random.shuffle(dataset)
all_labels = [a[0] for a in dataset]
all_words = [a[1] for a in dataset]
all_targets = [a[2] for a in dataset]

with open('train_labels.txt', 'w') as f:
    f.writelines(all_labels[:-540])

with open('train_sentences.txt', 'w') as f:
    f.writelines(all_words[:-540])

with open('train_targets.txt', 'w') as f:
    f.writelines(all_targets[:-540])

with open('valid_labels.txt', 'w') as f:
    f.writelines(all_labels[-540:])

with open('valid_sentences.txt', 'w') as f:
    f.writelines(all_words[-540:])

with open('valid_targets.txt', 'w') as f:
    f.writelines(all_targets[-540:])
