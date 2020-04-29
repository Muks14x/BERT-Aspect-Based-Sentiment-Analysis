import random

labels_file = 'train_valid_labels.txt'
words_file = 'train_valid_words.txt'

with open(labels_file) as f:
    all_labels = f.readlines()

with open(words_file) as f:
    all_words = f.readlines()

dataset = list(zip(all_labels, all_words))
random.shuffle(dataset)
all_labels = [a[0] for a in dataset]
all_words = [a[1] for a in dataset]

with open('train_labels.txt', 'w') as f:
    f.writelines(all_labels[:-700])

with open('train_words.txt', 'w') as f:
    f.writelines(all_words[:-700])

with open('valid_labels.txt', 'w') as f:
    f.writelines(all_labels[-700:])

with open('valid_words.txt', 'w') as f:
    f.writelines(all_words[-700:])
