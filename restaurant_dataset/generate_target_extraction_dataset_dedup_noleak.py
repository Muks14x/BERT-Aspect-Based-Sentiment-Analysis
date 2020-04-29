from bs4 import BeautifulSoup
import mosestokenizer
import os

outdir = 'processed_target_extraction_dedup_noleak'

# Ensure the datasets appear in chronological order.
# Repeated sentences with different labelling would be taken from newer datasets during deduplication.
datasets = {'train_valid': ['ABSA-15_Restaurants_Train_Final.xml', 'ABSA16_Restaurants_Train_SB1_v2.xml', 'Restaurants_Train_v2.xml'],
            'test': ['ABSA15_Restaurants_Test.xml', 'Restaurants_Test_Data_phaseB.xml']}

sentences = []
labels = []
tokenize = mosestokenizer.MosesTokenizer('en')

def processSentence(sentence):
    text = sentence.text.strip('\n')
    opinions = sentence.findAll('opinion')
    target_str = 'target'
    if not opinions:
        opinions = sentence.findAll('aspectterm')
        # Opinions are called aspectTerms in this file, and targets are called 'terms
        target_str = 'term'

    targets = [] #(target_str, beg, end)
    for op in opinions:

        if op[target_str] == 'NULL':
            continue
        else:
            beg = int(op['from']) # text.find(op['target'])
            end = int(op['to']) # beg + len(op['target'])
            targets.append((op[target_str], beg, end))
            # targets.append(op['target'], , op['end'])
    
    sorted_targets = sorted(targets, key=lambda x: x[1])
    # If assertion holds, targets don't overlap
    assert sorted_targets == sorted(targets, key=lambda x: x[2])
    
    # Skip sentences without usable targets
    if len(targets) == 0:
        return None
    
    words = []
    labels = []

    idx0 = 0
    for target in targets:
        idx1 = target[1]
        idx2 = target[2]
        non_target_words = tokenize(text[idx0:idx1])
        target_words = tokenize(text[idx1:idx2])
        
        words.extend(non_target_words)
        words.extend(target_words)

        for _ in non_target_words:
            labels.append("O")
        target_start = True
        for _ in target_words:
            if target_start:
                labels.append("B-TARGET")
                target_start = False
            else:
                labels.append("I-TARGET")
        
        idx0 = idx2
    
    if idx0 < len(text):
        non_target_words = tokenize(text[idx0:])
        words.extend(non_target_words)
        for _ in non_target_words:
            labels.append("O")

    assert len(words) == len(labels)
    return words, labels


def readfile(file):
    print('Reading {}'.format(file))
    with open(file, 'r') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
    sentence_soup = soup.findAll('sentence')
    print("{} sentences found".format(len(sentence_soup)))
    all_sentences = [] # (words, labels)
    for sentence in sentence_soup:
        ret = processSentence(sentence)
        if ret:
            all_sentences.append(ret)
    
    print("{} sentences with terms extracted".format(len(all_sentences)))
    return all_sentences


def writeoutputs(sentences, words_outpath, labels_outpath):
    print("Writing {} sentences to {} and {}".format(len(sentences), words_outpath, labels_outpath))
    fwords = open(words_outpath, 'w')
    flabels = open(labels_outpath, 'w')

    for words, labels in sentences:
        fwords.write(' '.join(words) + '\n')
        flabels.write(' '.join(labels) + '\n')

    flabels.close()
    fwords.close()
    return


sentences = {}
for setname, files_list in datasets.items():
    sentences[setname] = []
    for file in files_list:
        sentences[setname].extend(readfile(file))
    # writeoutputs(sentences, os.path.join(outdir, '{}_words.txt'.format(setname)),
    #                         os.path.join(outdir, '{}_labels.txt'.format(setname)))
    # Picking unique sentences
    uniq_sentences_dict = {}
    for words, labels in sentences[setname]:
        words = tuple(words)
        # Sentence-Label pairs that appear later are ranked higher
        #  (order of preference - better if the dataset is newer)
        uniq_sentences_dict[words] = labels
    sentences[setname] = list(uniq_sentences_dict.items())

# Remove test instances in train_valid
test_data_in = set([a for a, _ in sentences['test']])
train_valid_new = []
for i, (words, labels) in enumerate(sentences['train_valid']):
    if words not in test_data_in:
        train_valid_new.append((words, labels))

sentences['train_valid'] = train_valid_new

assert len({a[0] for a in sentences['train_valid']}.intersection(test_data_in)) == 0

for setname in datasets.keys():
    writeoutputs(sentences[setname], os.path.join(outdir, '{}_words.txt'.format(setname)),
                            os.path.join(outdir, '{}_labels.txt'.format(setname)))
