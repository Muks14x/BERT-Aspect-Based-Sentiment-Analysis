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
from transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)
from transformers import get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from sklearn.metrics import classification_report, f1_score

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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def readfiles(sentences_file, targets_file, labels_file):
    '''
    reads sentences_file, targets_file, labels_file
    '''
    fsentences = open(sentences_file)
    ftargets = open(targets_file)
    flabels = open(labels_file)
    sentences = []
    targets = []
    labels = []

    for line in fsentences:
        line = line.strip('\n')
        sentences.append(line.split(' '))

    for line in ftargets:
        line = line.strip('\n')
        targets.append(line.split(' '))

    for line in flabels:
        line = line.strip('\n')
        labels.append(line)

    fsentences.close()
    ftargets.close()
    flabels.close()

    data = zip(sentences, targets, labels)
    return data


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
    def _read_dataset(cls, sentences_file, targets_file, labels_file):
        """Reads a words file and a labels file."""
        return readfiles(sentences_file, targets_file, labels_file)



class ABSAProcessor(DataProcessor):
    """Processor for the ABSA dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_dataset(os.path.join(data_dir, "train_sentences.txt"),
                               os.path.join(data_dir, "train_targets.txt"),
                               os.path.join(data_dir, "train_labels.txt")),
            "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_dataset(os.path.join(data_dir, "valid_sentences.txt"),
                               os.path.join(data_dir, "valid_targets.txt"),
                               os.path.join(data_dir, "valid_labels.txt")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_dataset(os.path.join(data_dir, "test_sentences.txt"),
                               os.path.join(data_dir, "test_targets.txt"),
                               os.path.join(data_dir, "test_labels.txt")),
            "test")

    def get_labels(self):
        return ["negative", "positive"]
    
    def _create_examples(self, lines, set_type):
        examples = []
        allowed_labels = set(self.get_labels())
        for i, (sentence, target, label) in enumerate(lines):
            if label not in allowed_labels:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = ' '.join(target)
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
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
                InputFeatures(input_ids=input_ids,
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


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def evaluate(args, processor, model, tokenizer, device, eval_on, return_predictions=False):
    if eval_on == "dev":
        eval_examples = processor.get_dev_examples(args.data_dir)
    elif eval_on == "test":
        eval_examples = processor.get_test_examples(args.data_dir)
    else:
        raise ValueError("eval on dev or test set only")
    label_list = processor.get_labels()
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)

    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
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
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    true_labels = []
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
            tmp_eval_loss = eval_outputs[0]

        predicted_labels = predicted_labels + torch.argmax(logits, dim=-1).tolist()
        true_labels = true_labels + label_ids.tolist()

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    # print(true_labels)
    # print(predicted_labels)
    # eval_loss = eval_loss / nb_eval_steps
    # eval_accuracy = eval_accuracy / nb_eval_examples
    # return eval_loss, eval_accuracy, predicted_labels, true_labels
    report = classification_report(true_labels, predicted_labels, digits=4)
    f1 = f1_score(true_labels, predicted_labels)
    if return_predictions:
        return report, f1, (predicted_labels, true_labels, eval_examples)
    else:
        return report, f1, None


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir_bestmodel",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the best model will be written to.")

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
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    # parser.add_argument("--eval_on",
    #                     default="dev",
    #                     help="Whether to run eval on the dev set or test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
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
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {"absa_polarity_classification": ABSAProcessor}

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

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if os.path.exists(args.output_dir_bestmodel) and os.listdir(args.output_dir_bestmodel) and args.do_train:
        raise ValueError("Best model output directory ({}) already exists and is not empty.".format(args.output_dir_bestmodel))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir_bestmodel):
        os.makedirs(args.output_dir_bestmodel)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Prepare model
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
              from_tf = False,
              config = config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i : label for i, label in enumerate(label_list,1)}
    best_val_f1 = -1
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        for e_no in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                inputs = {'input_ids': input_ids,
                        'attention_mask': input_mask,
                        'labels': label_ids}
                outputs = model(**inputs)
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
            
            # Evaluate the model (validation & test), print results.
            valid_report, valid_f1, _ = evaluate(args, processor, model, tokenizer, device, "dev")
            test_report, test_f1, _ = evaluate(args, processor, model, tokenizer, device, "test")
            logger.info("\n*** epoch {} validation results ***".format(e_no))
            logger.info(valid_report + "\n")
            logger.info("\n*** epoch {} test results ***".format(e_no))
            logger.info(test_report + "\n")

            # Save best model and the associated configuration using validation scores
            if valid_f1 > best_val_f1:
                logger.info("Best validation score, saving best model")
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                model_to_save.save_pretrained(args.output_dir_bestmodel)
                tokenizer.save_pretrained(args.output_dir_bestmodel)
                label_map = {i : label for i, label in enumerate(label_list,1)}
                model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,"num_labels":len(label_list)+1,"label_map":label_map}
                json.dump(model_config,open(os.path.join(args.output_dir_bestmodel,"model_config.json"),"w"))
            else:
                logger.info("Not the best validation score, not overwriting best model")

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        label_map = {i : label for i, label in enumerate(label_list,1)}
        model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,"num_labels":len(label_list)+1,"label_map":label_map}
        json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
        # Load a trained model and config that you have fine-tuned
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Loading best model for final evaluation")
        # Load the best model
        model = BertForSequenceClassification.from_pretrained(args.output_dir_bestmodel)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir_bestmodel, do_lower_case=args.do_lower_case)
        model.to(device)

        # Evaluate the model (validation & test), print results.
        valid_report, valid_f1, preds_tuple_valid = evaluate(args, processor, model, tokenizer, device, "dev", True)
        test_report, test_f1, preds_tuple_test = evaluate(args, processor, model, tokenizer, device, "test", True)
        
        logger.info("\n*** Final validation results ***\n" + valid_report + "\n")
        logger.info("\n*** Final test results ***\n" + test_report + "\n")
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write('valid:\n' + valid_report + '\n')
            writer.write('test:\n' + test_report + '\n')
        
        valid_hypothesis_file = os.path.join(args.output_dir, "valid_hyp.tsv")        
        with open(valid_hypothesis_file, 'w') as writer:
            writer.write("{}\t{}\t{}\t{}\n".format("sentence", "target", "true", "pred"))
            writer.writelines(["{}\t{}\t{}\t{}\n".format(i.text_a, i.text_b, true, pred)
                                                            for i, true, pred in
                                                            zip(preds_tuple_valid[2], preds_tuple_valid[1], preds_tuple_valid[0])])
        
        test_hypothesis_file = os.path.join(args.output_dir, "test_hyp.tsv")
        with open(test_hypothesis_file, 'w') as writer:
            writer.write("{}\t{}\t{}\t{}\n".format("sentence", "target", "true", "pred"))
            writer.writelines(["{}\t{}\t{}\t{}\n".format(i.text_a, i.text_b, true, pred)
                                                            for i, true, pred in
                                                            zip(preds_tuple_test[2], preds_tuple_test[1], preds_tuple_test[0])])


if __name__ == "__main__":
    main()

