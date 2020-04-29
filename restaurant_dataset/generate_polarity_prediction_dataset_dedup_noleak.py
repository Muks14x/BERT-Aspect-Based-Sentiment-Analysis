from bs4 import BeautifulSoup
import mosestokenizer
import os

outdir = 'processed_polarity_prediction_dedup_noleak'

# Ensure the datasets appear in chronological order.
# Repeated sentences with different labelling would be taken from newer datasets during deduplication.
datasets = {'train_valid': ['ABSA-15_Restaurants_Train_Final.xml', 'ABSA16_Restaurants_Train_SB1_v2.xml', 'Restaurants_Train_v2.xml'],
            'test': ['ABSA15_Restaurants_Test.xml']} # 'Restaurants_Test_Data_phaseB.xml' - doesn't have polarities

sentences = []
labels = []
tokenize = mosestokenizer.MosesTokenizer('en')
# polarity_labels = {'negative': '0', 'positive': '1'}

def processSentence(sentence):
    text = sentence.text.strip('\n')
    opinions = sentence.findAll('opinion')
    target_str = 'target'
    if not opinions:
        opinions = sentence.findAll('aspectterm')
        # Opinions are called aspectTerms in this file, and targets are called 'terms
        target_str = 'term'

    targets = [] #(target_str, beg, end, polarity)
    for op in opinions:

        if op[target_str] == 'NULL':
            continue
        else:
            beg = int(op['from']) # text.find(op['target'])
            end = int(op['to']) # beg + len(op['target'])
            polarity = op['polarity']
            targets.append((op[target_str], beg, end, polarity))
            # targets.append(op['target'], , op['end'])
    
    sorted_targets = sorted(targets, key=lambda x: x[1])
    # If assertion holds, targets don't overlap
    assert sorted_targets == sorted(targets, key=lambda x: x[2])
    
    # Skip sentences without usable targets
    if len(targets) == 0:
        return None

    tokenized_words = tokenize(text)
    tokenized_targets = []
    target_polarities = []

    for target in targets:
        idx1 = target[1]
        idx2 = target[2]
        tokenized_targets.append(tokenize(text[idx1:idx2]))
        target_polarities.append(target[3])

    assert len(tokenized_targets) == len(target_polarities)
    return tokenized_words, tokenized_targets, target_polarities


def readfile(file):
    print('Reading {}'.format(file))
    with open(file, 'r') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
    sentence_soup = soup.findAll('sentence')
    print("{} sentences found".format(len(sentence_soup)))
    all_sentences = [] # (tokenized_words, tokenized_targets, target_polarities)
    for sentence in sentence_soup:
        ret = processSentence(sentence)
        if ret is not None:
            all_sentences.append(ret)
    
    print("{} sentences with terms extracted".format(len(all_sentences)))
    return all_sentences


def writeoutputs(sentences, sentences_outpath, targets_outpath, labels_outpath):
    print("Writing {} sentences to {}, {} and {}".format(len(sentences), sentences_outpath, targets_outpath, labels_outpath))
    fsentences = open(sentences_outpath, 'w')
    ftargets = open(targets_outpath, 'w')
    flabels = open(labels_outpath, 'w')

    for words, tgt, lbl in sentences:
        # for tgt, lbl in zip(targets, labels):
        fsentences.write(' '.join(words) + '\n')
        ftargets.write(' '.join(tgt) + '\n')
        flabels.write(lbl + '\n')
    
    fsentences.close()
    ftargets.close()
    flabels.close()
    return


sentences = {}
for setname, files_list in datasets.items():
    sentences[setname] = []
    for file in files_list:
        sentences[setname].extend(readfile(file))
    # writeoutputs(sentences, os.path.join(outdir, '{}_sentences.txt'.format(setname)),
    #                         os.path.join(outdir, '{}_targets.txt'.format(setname)),
    #                         os.path.join(outdir, '{}_labels.txt'.format(setname)))
    # Picking unique sentences
    sentences_dict = {}
    for words, targets, labels in sentences[setname]:
        words = tuple(words)
        # SentenceTarget-Label pairs that appear later are ranked higher
        #  (order of preference - better if the dataset is newer)
        for tgt, lbl in zip(targets, labels):
            tgt = tuple(tgt)
            sentences_dict[words, tgt] = lbl

    sentences[setname] = []
    for words, tgt in sentences_dict.keys():
        lbl = sentences_dict[words, tgt]
        sentences[setname].append((words, tgt, lbl))

test_data_in = {a[0] for a in sentences['test']}
train_valid_new = []
for i, (words, tgt, lbl) in enumerate(sentences['train_valid']):
    if words in test_data_in:
        pass
    else:
        train_valid_new.append((words, tgt, lbl))

sentences['train_valid'] = train_valid_new
assert len({a[0] for a in sentences['train_valid']}.intersection(sentences['test'])) == 0

for setname, files_list in datasets.items():
    writeoutputs(sentences[setname], os.path.join(outdir, '{}_sentences.txt'.format(setname)),
                            os.path.join(outdir, '{}_targets.txt'.format(setname)),
                            os.path.join(outdir, '{}_labels.txt'.format(setname)))
