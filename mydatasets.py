import re
import os
import random
import tarfile
from six.moves import urllib
from torchtext import data
import torchtext
import torch

class QaDataset(data.Dataset):

    def __init__(self, fields, src, label_vocab):
        examples = []
        with open(src, 'r') as p:
            for i, line in enumerate(p):
                l = line.strip().split('\t')
                text, label = l[0], label_vocab[l[1]]
                d = {'text': text, 'label': label}
                examples.append(d)

        keys = examples[0].keys()
        fields = [(k, fields[k]) for k in keys]
        examples = list([torchtext.data.Example.fromlist([ex[k] for k in keys], fields)
                         for ex in examples])

        super(QaDataset, self).__init__(examples, fields)


    @staticmethod
    def get_fields():
        fields = {}
        fields['text'] = torchtext.data.Field(
            lower=True)
        fields['label'] = torchtext.data.Field(
            use_vocab=False,
            tensor_type=torch.LongTensor,
            sequential=False)
        return fields

    @staticmethod
    def build_vocab(train, valid):
        fields = train.fields
        fields['text'].build_vocab(train, valid)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)
