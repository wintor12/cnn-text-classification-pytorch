
def get_label_vocab(train, valid, test):
    vocab = []
    def _get_v(s):
        v = []
        with open(s, 'r') as p:
            for i, line in enumerate(p):
                l = line.strip().split('\t')[1]
                v.append(l)
        return list(set(v))
    vocab += _get_v(train)
    vocab += _get_v(valid)
    vocab += _get_v(test)
    vocab = list(set(vocab))
    print(len(vocab))
    word2id, id2word = {}, {}
    for i, w in enumerate(vocab):
        word2id[w] = i + 1
        id2word[i + 1] = w
    return word2id, id2word
 

def load_data(train_data, valid_data, test_data):
    label2id, _ = get_label_vocab(train_data, valid_data, test_data)
    def _load_data(src):
        texts, labels = [], []
        with open(src, 'r') as p:
            for line in p:
                l = line.strip().split('\t')
                text, label = l[0], label2id[l[1]]
                texts.append(text)
                labels.append(label)
        return texts, labels
    train, valid, test = {}, {}, {}
    train['text'], train['label'] = _load_data(train_data)
    valid['text'], valid['label'] = _load_data(valid_data)
    test['text'], test['label'] = _load_data(test_data)
    return train, valid, test

    
