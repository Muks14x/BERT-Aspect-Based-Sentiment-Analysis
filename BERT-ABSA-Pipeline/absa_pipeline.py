from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer,
                                  BertForSequenceClassification, BertForTokenClassification)
from transformers import get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# from sklearn.metrics import classification_report, f1_score

logging.basicConfig(filename=str(datetime.datetime.now()).replace(' ', '_')+'.log',
                    filemode='a',
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class TermExtraction(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeaturesSequenceClassification(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputFeaturesTokenClassification(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask



def readfile(filepath):
    '''
    reads data from a file
    '''
    fsentences = open(filepath)
    sentences = []

    for line in fsentences:
        line = line.strip('\n')
        sentences.append(line.split(' '))

    fsentences.close()

    return sentences


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
    
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_dataset(cls, sentences_file):
        """Reads a file."""
        return readfile(sentences_file)


class TermExtractionProcessor(DataProcessor):
    """Processor for the Yelp data set."""

    # def get_train_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_dataset(os.path.join(data_dir, "train_words.txt"), os.path.join(data_dir, "train_labels.txt")), "train")

    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_dataset(os.path.join(data_dir, "valid_words.txt"), os.path.join(data_dir, "valid_labels.txt")), "dev")

    # def get_test_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_dataset(os.path.join(data_dir, "test_words.txt"), os.path.join(data_dir, "test_labels.txt")), "test")

    def get_labels(self):
        return ["O", "B-TARGET", "I-TARGET", "[CLS]", "[SEP]"]

    def get_examples(self, data_path):
        return self._create_examples(self._read_dataset(data_path), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for i,sentence in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            # Fake labels: Only used to generate label masks of appropriate lengths.
            label = ["O" for _ in sentence]
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples


class ABSAProcessor(DataProcessor):
    """Processor for the ABSA dataset."""

    # def get_train_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_dataset(os.path.join(data_dir, "train_sentences.txt"),
    #                            os.path.join(data_dir, "train_targets.txt"),
    #                            os.path.join(data_dir, "train_labels.txt")),
    #         "train")

    # def get_dev_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_dataset(os.path.join(data_dir, "valid_sentences.txt"),
    #                            os.path.join(data_dir, "valid_targets.txt"),
    #                            os.path.join(data_dir, "valid_labels.txt")),
    #         "dev")

    # def get_test_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_dataset(os.path.join(data_dir, "test_sentences.txt"),
    #                            os.path.join(data_dir, "test_targets.txt"),
    #                            os.path.join(data_dir, "test_labels.txt")),
    #         "test")

    def get_examples(self, data):
        # data should be list[(sentence, target, label)]
        return self._create_examples(data, 'test')

    def get_labels(self):
        return ["negative", "positive"]
    
    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, target) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if isinstance(sentence, list):
                text_a = ' '.join(sentence)
            else:
                text_a = sentence
            if isinstance(target, list):
                text_b = ' '.join(target)
            else:
                text_b = target
            # Fake label, not actually used
            label = "positive"
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples


def convert_examples_to_features_token_classification(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}

    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeaturesTokenClassification(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features


def convert_examples_to_features_sequence_classification(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeaturesSequenceClassification(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def extract_terms(args, processor, model, tokenizer, device, data_path):
    eval_examples = processor.get_examples(data_path)
    label_list = processor.get_labels()
    eval_features = convert_examples_to_features_token_classification(eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    y_pred = []
    label_map = {i : label for i, label in enumerate(label_list,1)}
    for input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)

        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_2 = []
            for j, _ in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    y_pred.append(temp_2)
                    break
                else:
                    temp_2.append(label_map[logits[i][j]])

    # Extract the terms as text
    dataset = [] # (sentence, [term1, term2, ...]) pairs
    for example, pred in zip(eval_examples, y_pred):
        sentence = example.text_a
        words = sentence.split(' ')
        terms_list = []
        term = []
        inside_term = False
        for i in range(len(words) + 1): 
            if inside_term and (i == len(words) or pred[i] == "O"):
                terms_list.append(term)
                term = []
                inside_term = False
                continue
            elif i == len(words):
                continue
            elif pred[i] == "B-TARGET":
                term.append(words[i])
                inside_term = True
            elif inside_term and (pred[i] == "I-TARGET"):
                term.append(words[i])

        sentence = sentence.replace('@@ ', '')
        terms_list = [' '.join(t).replace('@@ ', '') for t in terms_list]
        dataset.append((sentence, terms_list))

    return dataset


def classify_polarity(args, processor, model, tokenizer, device, dataset_input):
    eval_examples = processor.get_examples(dataset_input)
    label_list = processor.get_labels()
    eval_features = convert_examples_to_features_sequence_classification(eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    # eval_loss, eval_accuracy = 0, 0
    # nb_eval_steps, nb_eval_examples = 0, 0
    predicted_labels = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            inputs = {'input_ids': input_ids,
                  'attention_mask': input_mask,
                  'labels': label_ids}
            # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            eval_outputs = model(**inputs)
            # logits = model(input_ids, segment_ids, input_mask)
            logits = eval_outputs[1]
            # tmp_eval_loss = eval_outputs[0]

        predicted_labels = predicted_labels + torch.argmax(logits, dim=-1).tolist()
        # true_labels = true_labels + label_ids.tolist()

        logits = logits.detach().cpu().numpy()
        # label_ids = label_ids.to('cpu').numpy()
        # tmp_eval_accuracy = accuracy(logits, label_ids)

        # eval_loss += tmp_eval_loss.mean().item()
        # eval_accuracy += tmp_eval_accuracy

        # nb_eval_examples += input_ids.size(0)
        # nb_eval_steps += 1

    # print(true_labels)
    # print(predicted_labels)
    # eval_loss = eval_loss / nb_eval_steps
    # eval_accuracy = eval_accuracy / nb_eval_examples
    # return eval_loss, eval_accuracy, predicted_labels, true_labels
    predicted_labels = [label_list[p] for p in predicted_labels]
    return (predicted_labels, eval_examples)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data file.")
    parser.add_argument("--output_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The path to save outputs to.")
    # parser.add_argument("--bert_model", default=None, type=str, required=True,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                     "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
    #                     "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task.")
    parser.add_argument("--target_extraction_model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The target extraction model to load")
    parser.add_argument("--polarity_classification_model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The polarity classification model to load.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    # parser.add_argument("--num_train_epochs",
    #                     default=3.0,
    #                     type=float,
    #                     help="Total number of training epochs to perform.")
    # parser.add_argument("--warmup_proportion",
    #                     default=0.1,
    #                     type=float,
    #                     help="Proportion of training to perform linear learning rate warmup for. "
    #                          "E.g., 0.1 = 10%% of training.")
    # parser.add_argument("--weight_decay", default=0.01, type=float,
    #                     help="Weight deay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    processors = {"absa_polarity_classification": ABSAProcessor, "term_extraction":TermExtractionProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    logger.info("Args: {}".format(args))


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.target_extraction_model_dir):
        raise ValueError("Model directory ({}) doesn't exist.".format(args.target_extraction_model_dir))
    if not os.path.exists(args.polarity_classification_model_dir):
        raise ValueError("Model directory ({}) doesn't exist.".format(args.polarity_classification_model_dir))
    if os.path.exists(args.output_path):
        raise ValueError("Output path ({}) already exists!.".format(args.output_path))

    task_name = args.task_name.lower()

    polarity_processor = processors['absa_polarity_classification']()
    term_extraction_processor = processors['term_extraction']()

    term_extractor_labels = term_extraction_processor.get_labels()
    num_term_extractor_labels = len(term_extractor_labels) + 1

    polarity_labels = polarity_processor.get_labels()
    num_polarity_labels = len(polarity_labels) + 1

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Prepare models
    term_extraction_config = BertConfig.from_pretrained(args.target_extraction_model_dir, num_labels=num_term_extractor_labels, finetuning_task=args.task_name)
    term_extraction_model = TermExtraction.from_pretrained(args.target_extraction_model_dir,
              from_tf = False,
              config = term_extraction_config)
    term_extraction_tokenizer = BertTokenizer.from_pretrained(args.target_extraction_model_dir, do_lower_case=args.do_lower_case)

    term_extraction_model.to(device)

    polarity_config = BertConfig.from_pretrained(args.polarity_classification_model_dir, num_labels=num_polarity_labels, finetuning_task=args.task_name)
    polarity_model = BertForSequenceClassification.from_pretrained(args.polarity_classification_model_dir,
            from_tf = False,
            config = polarity_config)
    polarity_tokenizer = BertTokenizer.from_pretrained(args.polarity_classification_model_dir, do_lower_case=args.do_lower_case)
    polarity_model.to(device)

    # Perform ABSA
    term_extracted_dataset = extract_terms(args, term_extraction_processor, term_extraction_model, term_extraction_tokenizer, device, args.data_path)

    polarity_pred_input = []
    for sentence, terms_list in term_extracted_dataset:
        for term in terms_list:
            polarity_pred_input.append((sentence, term))
    
    polarities_pred, eval_examples = classify_polarity(args, polarity_processor, polarity_model, polarity_tokenizer, device, polarity_pred_input)
    
    # Group by eval_examples.text_a being the same
    with open(args.output_path, 'w') as f:
        for polarity, example in zip(polarities_pred, eval_examples):
            f.write("{}\t{}\t{}\n".format(example.text_a.replace('@@ ', ''), example.text_b.replace('@@ ', ''), polarity))


if __name__ == '__main__':
    main()
