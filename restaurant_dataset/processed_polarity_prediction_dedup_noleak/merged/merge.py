from transformers import BertTokenizer
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--sentences", type=str)
parser.add_argument("--targets", type=str)
parser.add_argument("--polarities", type=str)

parser.add_argument("--output", type=str)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenize = tokenizer.tokenize

polarity_convert = {'positive': 'POS', 'negative': 'NEG'}

args = parser.parse_args()

fs = open(args.sentences, 'r')
ft = open(args.targets, 'r')
fp = open(args.polarities, 'r')

fo = open(args.output, 'w')

for sent, targ, pol in zip(fs, ft, fp):
    if pol.strip() not in polarity_convert.keys():
        continue
    fo.write('{}\t{}\t{}\n'.format(' '.join(tokenize(sent.strip())), ' '.join(tokenize(targ.strip())), polarity_convert[pol.strip()]))

fo.close()

fp.close()
ft.close()
fs.close()
