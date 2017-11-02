#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
from torchtext.data import BucketIterator
import data_loader


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.0005, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=200, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=20, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='1,2,3', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()
print(args)


def data_iter(data, device):
    return BucketIterator(
        dataset=data, batch_size=args.batch_size,
        device=device if args.device >= 0 else -1,
        repeat=False)

# load data
print("\nLoading data...")
label2id, id2label = data_loader.get_label_vocab('train.txt', 'valid.txt', 'test.txt')
fields = mydatasets.QaDataset.get_fields()
train_data = mydatasets.QaDataset(fields, 'train.txt', label2id)
valid_data = mydatasets.QaDataset(fields, 'valid.txt', label2id)
test_data = mydatasets.QaDataset(fields, 'test.txt', label2id)

print(train_data[0].label)

print("Building vocabulary")
mydatasets.QaDataset.build_vocab(train_data, valid_data)
text_field = fields['text']


# update args and print
args.embed_num = len(text_field.vocab)
print(len(text_field.vocab))
args.class_num = len(label2id)
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
if args.snapshot is None:
    cnn = model.CNN_Text(args)
else :
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        cnn = torch.load(args.snapshot)
    except :
        print("Sorry, This snapshot doesn't exist."); exit()

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
        
train_iter, valid_iter, test_iter = (data_iter(train_data, args.device),
                                     data_iter(valid_data, args.device),
                                     data_iter(test_data, args.device))

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field)
    print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
elif args.test :
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else :
    print()
    train.train(train_iter, valid_iter, test_iter, cnn, args)
