import os
import random
import logging
import pickle
from collections import Counter


import numpy as np
import torch
from torch.utils.data import TensorDataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 有卷积 使用cudnn加速存在 不一致的情况
    torch.backends.cudnn.deterministic = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def build_vocab(args):
    vocab_path = os.path.join("data", args.task, "vocab.txt")

    if not os.path.exists(vocab_path):
        vocab = Counter()

        data_dir = os.path.join("data", args.task)
        files = os.listdir(data_dir)
        for file in files:
            if not os.path.isdir(file) and file != '.DS_Store':
                file_path = os.path.join("data", args.task, file)
                with open(file_path, 'r') as f:
                    for line in f.readlines():
                        text_a, text_b, label = line.strip().split('\t')
                        if args.do_lower_case:
                            text_a = text_a.lower()
                            text_b = text_b.lower()
                        vocab.update(text_a.split())
                        vocab.update(text_b.split())

        with open(vocab_path, 'w') as f:
            vocab = vocab.items()
            vocab = sorted(vocab, key=lambda x: x[1], reverse=True)
            f.write('<PAD>\n<UNK>\n')
            for _ in vocab:
                f.write(_[0] + '\n')


def load_vocab(args):
    vocab_path = os.path.join("data", args.task, "vocab.txt")
    if not os.path.exists(vocab_path):
        build_vocab(args)

    with open(vocab_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    word2id = {}
    vocab = []
    for (index, line) in enumerate(lines):
        word = line.strip()
        vocab.append(word)
        word2id[word] = index

    return vocab, word2id


def sentence2ids(args, sentence, word2id):
    if args.do_lower_case:
        sentence = sentence.lower()
    ids = []
    for word in sentence.strip().split():
        if word not in word2id.keys():
            ids.append(word2id['<UNK>'])
        else:
            ids.append(word2id[word])
    return ids


def load_embedding(args):
    embedding_cache_path = os.path.join('data/glove', args.task+'.pkl')

    if os.path.exists(embedding_cache_path):
        with open(embedding_cache_path, 'rb') as f:
            embedding = pickle.load(f)
    else:
        vocab_path = os.path.join('data', args.task, 'vocab.txt')

        if not os.path.exists(vocab_path):
            build_vocab(args)

        vocab, word2id = load_vocab(args)

        args.logger.info('load embedding ... ')

        args.vocab_size = max(args.vocab_size, len(vocab)+1)
        embedding = np.zeros((args.vocab_size, 300))
        tar_count = 0
        glove_vocab = {}
        glove_path = os.path.join("data", "glove", "glove.840B.300d.txt")
        with open(glove_path) as f:
            file_length = 2196017
            index = 0
            for line in f:
                index += 1
                if index % (file_length // 100) == 0:
                    args.logger.info(index // (file_length // 100))
                elems = line.rstrip().split()
                if len(elems) != 300 + 1:
                    continue
                token = elems[0]

                if token in vocab:
                    index = vocab.index(token)
                    vector = [float(x) for x in elems[1:]]
                    embedding[index] = vector
                    if token not in glove_vocab.keys():
                        tar_count += 1
                        glove_vocab[token] = 1
                else:
                    token = token.lower()
                    if token in vocab and token not in glove_vocab.keys():
                        index = vocab.index(token)
                        vector = [float(x) for x in elems[1:]]
                        embedding[index] = vector
                        tar_count += 1
                        glove_vocab[token] = 1

        args.logger.info('Number of word out of glove: %s, partition: %s' % (len(vocab) - tar_count,  (len(vocab) - tar_count) / len(vocab)))

        with open(embedding_cache_path, 'wb') as f:
            pickle.dump(embedding, f)

    return embedding


def load_dataset(args, word2id, data_type):
    data_path = os.path.join(args.data_dir, data_type + '.txt')

    # Read Data
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    examples = []
    for (i, line) in enumerate(lines):
        if len(line.strip().split('\t')) == 3:
            text_a, text_b, label = line.strip().split('\t')
        examples.append((text_a, text_b, label))

    # Convert to features
    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            args.logger.info("Writing example %d/%d" % (ex_index, len_examples))

        input_ids_a = sentence2ids(args, example[0], word2id)
        attention_mask_a = [1] * len(input_ids_a)
        padding_length_a = args.max_seq_length_a - len(input_ids_a)
        input_ids_a = input_ids_a + ([0] * padding_length_a)
        attention_mask_a = attention_mask_a + ([0] * padding_length_a)

        input_ids_b = sentence2ids(args, example[1], word2id)
        attention_mask_b = [1] * len(input_ids_b)
        padding_length_b = args.max_seq_length_b - len(input_ids_b)
        input_ids_b = input_ids_b + ([0] * padding_length_b)
        attention_mask_b = attention_mask_b + ([0] * padding_length_b)

        if example[2] not in args.labels:
            continue
        label = int(example[2])

        input_ids_a = input_ids_a[:args.max_seq_length_a]
        attention_mask_a = attention_mask_a[:args.max_seq_length_a]

        input_ids_b = input_ids_b[:args.max_seq_length_b]
        attention_mask_b = attention_mask_b[:args.max_seq_length_b]
        features.append((input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, label))

    all_input_ids_a = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_attention_mask_a = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_input_ids_b = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_attention_mask_b = torch.tensor([f[3] for f in features], dtype=torch.long)
    all_labels = torch.tensor([f[4] for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_input_ids_b, all_attention_mask_b, all_labels)
    return dataset


def model_init(model, args):
    # Init Model
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

