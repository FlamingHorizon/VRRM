#coding=utf-8

import sys
import re
import copy

import numpy
from keras import backend as K
from keras.layers import Input, merge, LSTM, Dense, AveragePooling1D, Reshape, Embedding, TimeDistributed, Activation, RepeatVector,Lambda
from keras.models import Model
from keras.engine.topology import Layer
from keras.optimizers import RMSprop, SGD, Adadelta
from theano import config


class Slice(Layer):
    def __init__(self, idx, **kwargs):
        self.idx = idx
        super(Slice, self).__init__(**kwargs)
    def call(self, inputs, mask=None):
        return inputs[:,self.idx,:,:]
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[3])

class Tho(Layer):
    def __init__(self, tho, **kwargs):
        self.tho = tho
        super(Tho, self).__init__(**kwargs)
    def call(self, inputs, mask=None):
        return inputs[:] / self.tho
    def get_output_shape_for(self, input_shape):
        return input_shape

def dot_loss(y_true, y_pred):
    multed = merge([y_true, y_pred], mode='mul')
    dot_sum = K.sum(multed, axis=-1)
    time_sum = K.sum(dot_sum, axis=-1)
    return time_sum

def first_q_acc(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1)[:,0], K.argmax(y_pred, axis=-1)[:,0]))


# sentence to tokens
def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def insert_vocab(word, vocab, inv_vocab):
    '''TODO:this function modifies the vocab'''
    l = len(vocab)
    # if (word not in vocab) and (word not in ['?', '.']):
    if (word not in vocab):
        vocab[word] = l + 1 # word id starts from 1.
        inv_vocab[str(l + 1)] = word

# lines to stories for each sub story. each story may refer to many q's.
def parse_stories(lines, vocab, inv_vocab, cut):
    '''Parse stories provided in the bAbi tasks format
    '''

    data = []
    story = []
    q_a_s = []
    q_flag = 0
    line_count = 0
    for line in lines:
        line = line.decode('utf-8').strip().lower()
        nid = line.split(' ')[0]
        line = ' '.join(line.split(' ')[1:])
        nid = int(nid)
        if q_flag == 1 and line_count != 0:
            substory = [x for x in story if x]
            if cut == 0 or (cut == 1 and len(substory) <= 8):
                data.append((substory, q_a_s))
            q_a_s = []
            q_flag = 0
            if nid == 1:
                story = []
        if '\t' in line:
            q_flag = 1
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            for q_w in q:
                insert_vocab(q_w, vocab, inv_vocab)
            insert_vocab(a, vocab, inv_vocab)
            q_a_s.append((q, a))
            story.append('')
        else:
            sent = tokenize(line)
            for single_w in sent:
                insert_vocab(single_w, vocab, inv_vocab)
            story.append(sent)
        line_count += 1
    # for the last substory
    if q_flag:
        substory = [x for x in story if x]
        if cut == 0 or (cut == 1 and len(substory) <= 8):
            data.append((substory, q_a_s))
    return data

# word token to int index
def int_stories(data, vocab):
    ''' vocab starts from 1 and assume that there are no o-o-v words.'''
    stories = [x[0] for x in data]
    story_lens = [len(x) for x in stories]
    max_story_len = max(story_lens)
    sent_lens = [max([len(y) for y in x]) for x in stories]
    max_sent_len = max(sent_lens)
    
    qs = []
    for pack in data:
        qs.append([y[0] for y in pack[1]])
    max_q_num = max([len(q_group) for q_group in qs])
    q_lens = [max([len(q) for q in q_group]) for q_group in qs]
    max_q_len = max(q_lens)

    # compute story ints

    story_int = numpy.zeros(shape=(len(stories), max_story_len, max_sent_len))
    story_mask = numpy.zeros(shape=story_int.shape)

    for sid, st in enumerate(stories):
        # print st
        for sentid, sent in enumerate(st):
            # print sent
            sent_i = [vocab[w] for w in sent if w in vocab]
            story_int[sid][sentid][:len(sent_i)] = numpy.array(sent_i)
            story_mask[sid][sentid][:len(sent_i)] = numpy.ones(len(sent_i))

    # compute q ints

    q_int = numpy.zeros(shape=(len(qs), max_q_num, max_q_len))
    q_mask = numpy.zeros(shape=q_int.shape)

    for qid, q_group in enumerate(qs):
        for q_n, q in enumerate(q_group):
            q_i = [vocab[w] for w in q if w in vocab]
            q_int[qid][q_n][:len(q_i)] = numpy.array(q_i)
            q_mask[qid][q_n][:len(q_i)] = numpy.ones(len(q_i))

    # compute answers int

    answers = [[y[1] for y in pack[1]] for pack in data]

    ans_int = numpy.zeros(shape=(len(qs), max_q_num, len(vocab)))
    ans_mask = numpy.zeros(shape=(len(qs), max_q_num))
    for qid, a_group in enumerate(answers):
        for a_n, a in enumerate(a_group):
            ans_int[qid][a_n][vocab[a] - 1] = 1
            ans_mask[qid][a_n] = 1
    
    sizes = [max_story_len, max_sent_len, max_q_num, max_q_len]

    return story_int, story_mask, q_int, q_mask, ans_int, ans_mask, sizes

def ortho_weight(ndim):
    W = numpy.random.randn(ndim,ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)
    
class envEmbedding(object):
    def __init__(self, num_relations, num_entities, dim_emb):
        orth_relat = ortho_weight(max(num_relations, dim_emb))
        self.relat_dict = orth_relat[:dim_emb][:num_relations + 1]
        orth_enti = ortho_weight(dim_emb)
        self.enti_dict = orth_enti[:][:num_entities + 1]
        orth_position = ortho_weight(dim_emb)
        self.pos_dict = orth_position[:][:3]
        
        self.dim_emb = dim_emb

    def returnSingleEmb(self, id, mode, pos, time=False):
        ''' mode = 0 for enti, 1 for relat; pos = 0 for arg1, 1 for relat, 2 for arg2'''
        time_vec = self.pos_dict[:][pos]
        if mode == 0:
            if time:
                return self.enti_dict[:][id] * time_vec
            else:
                return self.enti_dict[:][id]
        else:
            if time:
                return self.relat_dict[:][id] * time_vec
            else:
                return self.relat_dict[:][id]

    def returnTupleEmb(self, input_tuple):
        return numpy.tanh(self.enti_dict[:][input_tuple[0]] * self.pos_dict[:][0]
                        + self.relat_dict[:][input_tuple[1]] * self.pos_dict[:][1]
                        + self.enti_dict[:][input_tuple[2]] * self.pos_dict[:][2]) # nonlinearity is needed.
    
    def merge_embedding(self, tupleList):
        total_embedding = numpy.zeros((self.dim_emb, 1))
        for tuple in tupleList:
            total_embedding = total_embedding + self.returnTupleEmb(tuple).reshape(total_embedding.shape)
        return total_embedding

class varTable(object):
    def __init__(self):
        self.var_dict = {}
        self.temp_idx = 1
    def retr_insert(self, key_id, inv_vocab):
        key = inv_vocab[str(key_id)]
        if key not in self.var_dict:
            self.var_dict[key] = self.temp_idx
            self.temp_idx += 1
        return self.var_dict[key]
    def retr(self, key_id, inv_vocab):
        if key_id == 0:
            return 0
        key = inv_vocab[str(key_id)]
        if key in self.var_dict:
            return self.var_dict[key]
        else:
            return 0
    
    def inv_retr(self, id):
        for k,v in self.var_dict.iteritems():
            if v == id:
                return k
        return 'None'
        
    def reset(self):
        self.var_dict = {}
        self.temp_idx = 1
        
        
def select_action(probs, epsilon):
    probs = probs.reshape(probs.shape[-1])
    l = len(probs)
    max_id = numpy.argmax(probs)
    max_v = probs[max_id]
    for i in range(l):
        if i != max_id:
            probs[i] += min(epsilon, max_v) / (l - 1)
        else:
            probs[i] += -min(epsilon, max_v)
    probs = probs / probs.sum() - 1e-6
    one_hot_action = numpy.random.multinomial(1,probs)
    for i in range(l):
        if one_hot_action[i] == 1:
            return i, one_hot_action

def select_action_hard(probs, epsilon):
    probs = probs.reshape(probs.shape[-1])
    l = len(probs)
    max_id = numpy.argmax(probs)
    one_hot_action = numpy.zeros(l)
    one_hot_action[max_id] = 1
    return max_id, one_hot_action     
    
    
def repeatTensor(array_in, rep):
    array_out = numpy.zeros(array_in.shape + (rep,))
    for i in range(rep):
        array_out[:,:,:,i] = array_in
    return array_out
    

        


    
    
