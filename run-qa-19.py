#coding=utf-8



import sys
import re

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

# prepare story data

# sentence to tokens
def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def insert_vocab(word, vocab, inv_vocab):
    l = len(vocab)
    if (word not in vocab):
        vocab[word] = l + 1 # word id starts from 1.
        inv_vocab[str(l + 1)] = word

# lines to stories for each sub story. each story may refer to many q's.
def parse_stories(lines, vocab, inv_vocab):

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
        data.append((substory, q_a_s))
    return data

# word token to int index
def int_stories(data, vocab):
    ''' vocab starts from 1 and assume that there are no o-o-v words.'''
    stories = [x[0] for x in data]
    # print stories
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


    story_int = numpy.zeros(shape=(len(stories), max_story_len, max_sent_len))
    story_mask = numpy.zeros(shape=story_int.shape)

    for sid, st in enumerate(stories):
        for sentid, sent in enumerate(st):
            sent_i = [vocab[w] for w in sent if w in vocab]
            story_int[sid][sentid][:len(sent_i)] = numpy.array(sent_i)
            story_mask[sid][sentid][:len(sent_i)] = numpy.ones(len(sent_i))


    q_int = numpy.zeros(shape=(len(qs), max_q_num, max_q_len))
    q_mask = numpy.zeros(shape=q_int.shape)

    for qid, q_group in enumerate(qs):
        for q_n, q in enumerate(q_group):
            q_i = [vocab[w] for w in q if w in vocab]
            q_int[qid][q_n][:len(q_i)] = numpy.array(q_i)
            q_mask[qid][q_n][:len(q_i)] = numpy.ones(len(q_i))


    answers = [[y[1] for y in pack[1]] for pack in data]

    ans_int = numpy.zeros(shape=(len(qs), max_q_num, len(vocab)))
    ans_mask = numpy.zeros(shape=(len(qs), max_q_num))
    for qid, a_group in enumerate(answers):
        for a_n, a in enumerate(a_group):
            ans_int[qid][a_n][vocab[a] - 1] = 1
            ans_mask[qid][a_n] = 1
    
    sizes = [max_story_len, max_sent_len, max_q_num, max_q_len]

    return story_int, story_mask, q_int, q_mask, ans_int, ans_mask, sizes

# build RL model with keras


def Embed_word(x, vocab_size, dim_emb):
    y = Embedding(vocab_size, dim_emb)(x)
    return Model(input=x, output=y)


# merge all modules together to build the whole model
def DRL_Reasoner(params):
    # hyper params
    max_story_len = params['max_story_len'] 
    max_sent_len = params['max_sent_len'] 
    max_q_num = params['max_story_len']
    
    vocab_size = params['vocab_size']
    dim_emb_story = params['dim_emb_story'] 
    dim_emb_env = params['dim_emb_env'] 
    dim_tracker = params['dim_tracker']
    
    entity_num_act = params['ent_range']
    relation_num_act = params['relation_num']
    
    dim_q_h_ff = params['dim_q_h_ff']
    dim_comp_ff = params['dim_comp_ff']
    
    vocab_size_ans = params['vocab_size_ans'] 
    vocab_size_entity = params['ent_range']
    vocab_size_relation = 1 
    
    tho = params['tho']
    
    
    # Input Tensors
    story_input = Input(shape=(max_story_len, max_sent_len))
    story_mask = Input(shape=(max_story_len, max_sent_len, dim_emb_story))
    q_input = Input(shape=(max_q_num, max_sent_len))
    q_mask = Input(shape=(max_q_num, max_sent_len, dim_emb_story))
    mask_sim = Input(shape=(max_story_len, max_sent_len * dim_emb_story))
    reward_value = Input(shape=(max_story_len, entity_num_act * 1 * relation_num_act))
    reward_value_retr = Input(shape=(max_q_num, vocab_size_entity * 1 * vocab_size_relation))
    
    
    embed_word = Embedding(vocab_size, dim_emb_story, input_length=max_sent_len)
    embed_seq = TimeDistributed(embed_word, input_shape=(max_story_len, max_sent_len), name='embed_seq')
    tb_emb = embed_seq(story_input)
    tb_m_emb = merge([tb_emb, story_mask], mode='mul') 
    encode_single_story = Reshape((max_sent_len * dim_emb_story,), input_shape=(max_sent_len, dim_emb_story))
    encode_story = TimeDistributed(encode_single_story, input_shape=(max_story_len, max_sent_len, dim_emb_story), name='encode_story')
    sent_embs = encode_story(tb_m_emb)
    
    

    tb_emb_q = embed_seq(q_input)
    tb_m_emb_q = merge([tb_emb_q, q_mask], mode='mul') 
    q_embs_raw = encode_story(tb_m_emb_q) 
    q_embs = TimeDistributed(Dense(dim_tracker, activation='sigmoid'), input_shape=(max_q_num, max_sent_len * dim_emb_story))(q_embs_raw)


    # merge and mask
    # sent_env_embs = merge([sent_embs, env_input], mode='concat', concat_axis=2) 
    sent_env_embs = sent_embs
    embs_masked = merge([sent_env_embs, mask_sim], mode='mul') 
    
    # state tracker
    # hiddens = LSTM(dim_tracker, return_sequences=True)(embs_masked)
    hiddens = TimeDistributed(Dense(dim_tracker, activation='sigmoid'), input_shape=(max_story_len, max_sent_len * dim_emb_story))(embs_masked)
    final_rnn_state = Reshape((dim_tracker,))(AveragePooling1D(max_story_len)(hiddens)) 

    # policy distribution
    arg1_bind_raw = Tho(params['tho'])(TimeDistributed(Dense(entity_num_act), input_shape=(max_story_len, dim_tracker), name='arg1_bind_raw')(hiddens))
    arg1_bind_soft = Reshape((max_story_len, entity_num_act, 1))(Activation('softmax')(arg1_bind_raw)) 
    arg2_bind_raw = Tho(params['tho'])(TimeDistributed(Dense(1), input_shape=(max_story_len, dim_tracker), name='arg2_bind_raw')(hiddens))
    arg2_bind_soft = Reshape((max_story_len, 1, 1))(Activation('softmax')(arg2_bind_raw)) 
    
    arg12_bind_list = [Reshape((1, entity_num_act * 1, 1))(merge([Slice(i)(arg1_bind_soft), Slice(i)(arg2_bind_soft)], mode='dot', dot_axes=(2,1))) for i in range(max_story_len)] # q_num * ent_num ^ 2 * 1
    arg12_bind = merge(arg12_bind_list, mode='concat', concat_axis=1)
    
    relate_bind_raw = Tho(params['tho'])(TimeDistributed(Dense(relation_num_act), input_shape=(max_story_len, dim_tracker), name='relate_bind_raw')(hiddens))
    relate_bind_soft = Reshape((max_story_len, 1, relation_num_act))(Activation('softmax')(relate_bind_raw))
    bind_probs_list = [Reshape((1, entity_num_act * 1 * relation_num_act))(merge([Slice(i)(arg12_bind), Slice(i)(relate_bind_soft)], mode='dot', dot_axes=(2,1))) for i in range(max_story_len)]
    bind_probs = merge(bind_probs_list, mode='concat', concat_axis=1)
    bind_probs_log = Lambda(lambda x: (-1) * (K.log(x) + K.epsilon()))(bind_probs)
    bind_probs_re = merge([bind_probs_log, reward_value], mode='mul', name='action_probs_re')
    

    # retrieve probs and answer generation  
    states = RepeatVector(max_q_num)(final_rnn_state)
    q_state = merge([q_embs, states], mode='concat', concat_axis=2)
    retrieve_state = TimeDistributed(Dense(dim_q_h_ff, activation='sigmoid'), input_shape=(max_q_num, q_embs.shape[2]))(q_embs)
    arg1 = Activation('softmax')(Tho(params['tho'])(TimeDistributed(Dense(vocab_size_entity))(retrieve_state)))
    arg2 = Activation('softmax')(Tho(params['tho'])(TimeDistributed(Dense(1))(retrieve_state)))
    relation = Activation('softmax')(Tho(params['tho'])(TimeDistributed(Dense(vocab_size_relation))(retrieve_state)))
    

    # form the retrieve prob vector
    arg1_T = Reshape((max_q_num, vocab_size_entity, 1))(arg1) 
    arg2_T = Reshape((max_q_num, 1, 1))(arg2) 
    arg12_list = [Reshape((1, vocab_size_entity * 1, 1))(merge([Slice(i)(arg1_T), Slice(i)(arg2_T)], mode='dot', dot_axes=(2,1))) for i in range(max_story_len)] 
    arg12 = merge(arg12_list, mode='concat', concat_axis=1)
    relat_T = Reshape((max_q_num, 1, vocab_size_relation))(relation) 
    retrieve_probs_list = [Reshape((1, vocab_size_entity * 1 * vocab_size_relation))(merge([Slice(i)(arg12), Slice(i)(relat_T)], mode='dot', dot_axes=(2,1))) for i in range(max_story_len)]

    retrieve_probs = merge(retrieve_probs_list, mode='concat', concat_axis=1)
    retrieve_probs_log = Lambda(lambda x: (-1) * (K.log(x) + K.epsilon()))(retrieve_probs)
    retrieve_probs_re = merge([retrieve_probs_log, reward_value_retr], mode='mul', name='retrieve_probs_re')

    # a complete model
    DRL_complete = Model(input=[story_input, story_mask, q_input, q_mask, mask_sim, reward_value, reward_value_retr],
                                                output=[bind_probs_re, retrieve_probs_re])
    rmsp = RMSprop(clipnorm=2., lr=0.0001)
    sgd = SGD(clipnorm=100.)
    adad = Adadelta(clipnorm=10.)
    DRL_complete.compile(optimizer=rmsp,
              loss={'action_probs_re': dot_loss, 'retrieve_probs_re': dot_loss},)
    DRL_sim = Model(input=[story_input, story_mask, q_input, q_mask, mask_sim, reward_value, reward_value_retr],
                                                output=[bind_probs, retrieve_probs])
    DRL_debug = Model(input=[story_input, story_mask, q_input, q_mask, mask_sim, reward_value, reward_value_retr],
                                                output=[sent_embs, q_embs, hiddens, arg1_bind_raw, arg1_bind_soft, arg2_bind_raw, arg2_bind_soft,
                                                relate_bind_raw, relate_bind_soft, bind_probs, bind_probs_log, bind_probs_re,
                                                states, arg1, arg2, relation, retrieve_probs, retrieve_probs_log, retrieve_probs_re,
                                                ])
    # a function to check gradients wrt. arbitrary weights.
    layer_name_interested = ['embed_seq']
    weights = []
    for s in layer_name_interested:
        weights.extend(DRL_complete.get_layer(s).trainable_weights)
    print weights
    gradients = DRL_complete.optimizer.get_gradients(DRL_complete.total_loss, weights)
    input_tensors = []
    input_tensors.extend(DRL_complete.inputs)
    input_tensors.extend(DRL_complete.sample_weights)
    input_tensors.extend(DRL_complete.targets)
    input_tensors.append(K.learning_phase())
    get_grad = K.function(inputs=input_tensors, outputs=gradients)

    return DRL_complete, DRL_sim, DRL_debug, get_grad

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
    def reset(self):
        self.var_dict = {}
        self.temp_idx = 1
        
class working_environment(object):
    def __init__(self, num_entities, num_relations):
        self.tupleList = []
        self.tempEntiList = []
        self.adjGraph = numpy.zeros((num_entities, num_entities))
        self.temp_index = 1
        
        self.relationMap = numpy.array([[1,0,5,6], [0,2,7,8], [9,10,3,0], [11,12,0,4]])
        self.oppoList = [(1,2), (2,1), (4,3), (3,4)] # all entities and relations index start from 1 and leave 0 for 'equal' relation
        self.num_entities = num_entities
        self.num_relations = num_relations

    def retrieveRelation(self, arg1, arg2):
        for a, r, b in self.tupleList:
            if a == arg1 and b == arg2:
                return r
        return self.num_relations
    
    def returnEnv(self):
        return self.tupleList, self.adjGraph, self.temp_index
    
    def returnIndex(self):
        return self.temp_index
        
    def resetEnv(self):
        self.tupleList = []
        self.tempEntiList = []
        self.adjGraph = numpy.zeros((self.num_entities, self.num_entities))
        self.temp_index = 1
    
    def getOppo(self, relate):
        for i in self.oppoList:
            if i[0] == relate:
                return i[1]
        return 0
    
    def modifyEnv(self, commend):
        if commend:
            # add the origin relationship into tupleList
            self.tupleList = [t for t in self.tupleList if not((t[0] == commend[0] and t[2] == commend[2]) or (t[2] == commend[0] and t[0] == commend[2]))]
            self.tupleList.append(commend)
            self.adjGraph[commend[0] - 1][commend[2] - 1] = 1
            self.adjGraph[commend[2] - 1][commend[0] - 1] = 1
            # add oppo commend
            oppoRelation = self.getOppo(commend[1])
            self.tupleList.append((commend[2], oppoRelation, commend[0]))

            # inference all the other relations
            # for commend[0]
            for j in range(self.num_entities):
                if j != commend[0] - 1 and self.adjGraph[commend[2] - 1][j] == 1:
                    relation_step2 = self.retrieveRelation(commend[2], j + 1) - 1
                    relation_forward = self.relationMap[commend[1] - 1][relation_step2]
                    relation_backward = self.relationMap[self.getOppo(relation_step2 + 1) - 1][self.getOppo(commend[1]) - 1]
                    self.tupleList = [t for t in self.tupleList if not((t[0] == commend[0] and t[2] == j + 1) or (t[2] == commend[0] and t[0] == j + 1))]
                    self.tupleList.append((commend[0], relation_forward, j + 1))
                    self.tupleList.append((j + 1, relation_backward, commend[0]))
                    self.adjGraph[commend[0] - 1][j] = 2
                    self.adjGraph[j][commend[0] - 1] = 2

            # for commend[2]
            for j in range(self.num_entities):
                if j != commend[2] - 1 and self.adjGraph[commend[0] - 1][j] == 1:
                    relation_step2 = self.retrieveRelation(commend[0], j + 1) - 1
                    relation_forward = self.relationMap[oppoRelation - 1][relation_step2]
                    relation_backward = self.relationMap[self.getOppo(relation_step2 + 1) - 1][self.getOppo(oppoRelation) - 1]
                    self.tupleList = [t for t in self.tupleList if not((t[0] == commend[2] and t[2] == j + 1) or (t[2] == commend[2] and t[0] == j + 1))]
                    self.tupleList.append((commend[2], relation_forward, j + 1))
                    self.tupleList.append((j + 1, relation_backward, commend[2]))
                    self.adjGraph[commend[2] - 1][j] = 2
                    self.adjGraph[j][commend[2] - 1] = 2

            # modify the enti list and new temp_index.
            for i in [0,2]:
                if commend[i] not in self.tempEntiList:
                    self.tempEntiList.append(commend[i])
                    self.temp_index += 1

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

def compute_single_reward(final_ans,label_word):
    reward_scalar = -1.
    if final_ans == label_word:
        reward_scalar += 10
    return reward_scalar, final_ans, label_word

def repeatTensor(array_in, rep):
    array_out = numpy.zeros(array_in.shape + (rep,))
    for i in range(rep):
        array_out[:,:,:,i] = array_in
    return array_out

            
def train_two_phrase(train_file_names, test_file_names, params):
    relate_dict = {'east':1, 'west':2, 'south':3, 'north':4}
    ent_list = {'1':'office', '2':'hallway', '3':'kitchen', '4':'garden', '5':'bedroom', '6':'bathroom'}
    ans_dict = {'e':1, 'w':2, 's':3, 'n':4}
    # form vocabs and data
    numpy.set_printoptions(precision=3)
    print 'loading data.'
    old_std = sys.stdout
    f_print = open('debug_print_phr1.txt','w')
    f_debug = open('debug_print_phr2.txt','w')
    f_test = open('debug_print_test.txt','w')
    
    sys.stdout = f_print
    lines = []
    for f_name in train_file_names:
        f = open(f_name, 'r')
        lines.extend(f.readlines())
        f.close()
    
    lines_test = []
    for f_name in test_file_names:
        f = open(f_name, 'r')
        lines_test.extend(f.readlines())
        f.close()
        
    vocab_text = {}
    inv_vocab_text = {}    
    data = parse_stories(lines, vocab_text, inv_vocab_text)
    data_test = parse_stories(lines_test, vocab_text, inv_vocab_text)
    
    story_int, story_mask, q_int, q_mask, ans_int, ans_mask, sizes = int_stories(data, vocab_text)
    story_mask = repeatTensor(story_mask, params['dim_emb_story'])
    q_mask = repeatTensor(q_mask, params['dim_emb_story'])
    
    story_int_test, story_mask_test, q_int_test, q_mask_test, ans_int_test, ans_mask_test, sizes_test = int_stories(data_test, vocab_text)
    story_mask_test = repeatTensor(story_mask_test, params['dim_emb_story'])
    q_mask_test = repeatTensor(q_mask_test, params['dim_emb_story'])
    
    params['max_story_len'] = max(sizes[2], sizes[0])
    params['max_sent_len'] = max(sizes[1], sizes[3])
    params['max_q_num'] = max(sizes[2], sizes[0])
    params['vocab_size'] = len(vocab_text)
    params['vocab_size_ans'] = len(vocab_text)
    n_samples = len(story_int)
    
    params['ent_range'] = 2 # normal order and inverse order.


    print 'params:'
    print params
    print 'n_samples:'
    print n_samples
    print 'vocab_text:'
    print vocab_text
    sys.stdout = old_std
    
    
    # initialize the env embeddings, actions ,etc.
    story_train = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_story_len'], params['max_sent_len']))
    story_mask_train = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_story_len'], params['max_sent_len'], params['dim_emb_story']))
    q_train = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_q_num'], params['max_sent_len']))
    q_mask_train = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_q_num'], params['max_sent_len'], params['dim_emb_story']))
    
    act_selected_train = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_story_len'], (params['ent_range']) * 1 * params['relation_num'])) # one hot vector
    reward_train = numpy.ones((n_samples * params['story_sims_per_epoch'], params['max_story_len'], (params['ent_range']) * 1 * params['relation_num'])) # real number reward signal
    
    reward_immediate_train = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_story_len']))
    
    ans_train = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_q_num'], params['vocab_size']))
    ans_mask_train = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_q_num']))
    fin_train = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_q_num'], params['dim_emb_env'])) 
    fin_train_pred = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_q_num'], params['dim_emb_env']))
    fin_one_hot_train = numpy.zeros((n_samples * params['story_sims_per_epoch'], params['max_q_num'], (params['ent_range']) * 1 * 1))
    reward_fin_train = numpy.ones((n_samples * params['story_sims_per_epoch'], params['max_q_num'], (params['ent_range']) * 1 * 1))    
    
    
    print 'building model.'
    # simulate and generate final train data, do batch train and policy gd
    # build model
    reasoner, simulator, debugger, gradient_checker = DRL_Reasoner(params)
    
    working_memory = working_environment(params['enti_num'], params['relation_num_expand'])
    working_embed = envEmbedding(params['relation_num_expand'], params['enti_num'], params['dim_emb_env'])
    work_space_table = varTable()
    # for every single story-question group, take simulations, compute the env embeddings, actions taken, and total rewards.
    # take a certain number of simulations for each group, and use the pool as total train data, do sgd and batch training.
    # loop that progress for many epoches.
    mask_sim = numpy.zeros((1, params['max_story_len'], params['dim_emb_story'] * params['max_sent_len']))
    
    all_sim_number = 0
    
    print 'two phrase training started.'
    
    for epoch_id in range(params['epoch_num']):
        train_trace_id = 0
        avg_reward_q_epoch = 0.
        avg_reward_action_epoch = numpy.zeros(params['max_story_len'])
        print 'epoch %d phrase 1' %epoch_id
        # phrase 1: simulate and train data generating
        epoch_precision_rate = 0.
        sample_rate = numpy.zeros(n_samples)
        for story_id in range(n_samples):
            story_precision = 0.
            for sim_round in range(params['story_sims_per_epoch']):           
                sys.stdout = f_print
                story_train[train_trace_id,:len(story_int[0]),:len(story_int[0][0])] = story_int[story_id]
                story_mask_train[train_trace_id,:len(story_int[0]),:len(story_int[0][0]),:] = story_mask[story_id]
                q_train[train_trace_id,:len(q_int[0]),:len(q_int[0][0])] = q_int[story_id]
                q_mask_train[train_trace_id,:len(q_int[0]),:len(q_int[0][0]),:] = q_mask[story_id]
                ans_train[train_trace_id,:len(ans_int[0]),:len(ans_int[0][0])] = ans_int[story_id]

                
                for time_slice in range(params['max_story_len']):
                    mask_sim[0][time_slice][:] = numpy.ones(params['dim_emb_story'] * params['max_sent_len'])
                    

                    # read and embed environment
                    tupleList, adjGraph, temp_index = working_memory.returnEnv()

                    action_probs, retrieve_probs = simulator.predict([story_train[numpy.newaxis,train_trace_id], 
                                                                      story_mask_train[numpy.newaxis,train_trace_id],
                                                                      q_train[numpy.newaxis,train_trace_id],
                                                                      q_mask_train[numpy.newaxis,train_trace_id],
                                                                      mask_sim[:], 
                                                                      reward_train[numpy.newaxis,train_trace_id],
                                                                      reward_fin_train[numpy.newaxis,train_trace_id]])

                    action_selected, action_one_hot = select_action(action_probs[:,time_slice,:], params['epsilon'])


                    act_selected_train[train_trace_id, time_slice,:] = action_one_hot
                    arg_1_ptr = action_selected // ((1) * params['relation_num']) # start from 0 (empty number)
                    arg_2_ptr = (action_selected - arg_1_ptr * (1) * params['relation_num']) // params['relation_num']
                    arg_r_ptr = (action_selected - arg_1_ptr * (1) * params['relation_num']) % params['relation_num'] + 1
                    arg_r = arg_r_ptr


                    
                    flag = 0
                    for w_id in story_int[story_id][time_slice]:
                        if inv_vocab_text[str(int(w_id))] in ent_list.values() and flag == 0:
                            if arg_1_ptr == 0:
                                arg_1 = int(w_id)
                            else:
                                arg_2 = int(w_id)
                            flag = 1
                        elif inv_vocab_text[str(int(w_id))] in ent_list.values() and flag == 1:
                            if arg_1_ptr == 0:
                                arg_2 = int(w_id)
                            else:
                                arg_1 = int(w_id)                    
                    
                    slice_reward = 0
                    for tt in range(time_slice+1):    
                        reward_immediate_train[train_trace_id][tt] += slice_reward * (params['reward_decay'] ** (time_slice - tt))
                    

                    if arg_1 > 0 and arg_2 > 0:
                        # retrieve the table
                        arg_1_int = work_space_table.retr_insert(arg_1, inv_vocab_text)
                        arg_2_int = work_space_table.retr_insert(arg_2, inv_vocab_text)
                        working_memory.modifyEnv((arg_1_int, arg_r, arg_2_int))
                
                retrieved_relation_list = []
                reward_temp_list = []
                for q_idx in range(len(retrieve_probs[0])):                    
                    retr_idx = numpy.argmax(retrieve_probs[0,q_idx,:])
                    arg1_retr_ptr = retr_idx // ((1) * 1)
                    arg2_retr_ptr = (retr_idx - arg1_retr_ptr * (1) * 1) // 1
                    relation_pred = (retr_idx - arg1_retr_ptr * (1) * 1) % 1 + 1
                    
                    flag = 0
                    for q_w in q_int[story_id, 0]:
                        if inv_vocab_text[str(int(q_w))] in ent_list.values() and flag == 0:
                            if arg1_retr_ptr == 0:
                                arg1_retr = int(q_w)
                            else:
                                arg2_retr = int(q_w)
                            flag = 1
                        elif inv_vocab_text[str(int(q_w))] in ent_list.values() and flag == 1:
                            if arg1_retr_ptr == 0:
                                arg2_retr = int(q_w)
                            else:
                                arg1_retr = int(q_w)

                    retrieve_reward_pre = 0
                    reward_temp_list.append(retrieve_reward_pre)

                    
                    arg1_retr_int = work_space_table.retr(arg1_retr, inv_vocab_text)
                    arg2_retr_int = work_space_table.retr(arg2_retr, inv_vocab_text)

                    relation_retr = working_memory.retrieveRelation(arg1_retr_int, arg2_retr_int)
                    retrieved_relation_list.append(relation_retr)

                    
                    one_hot_single = numpy.zeros(((params['ent_range']) * 1 * 1))
                    one_hot_single[retr_idx] = 1
                    fin_one_hot_train[train_trace_id, q_idx, :] = one_hot_single


                reward_q_total = 0.
                ans_shared = 0
                for q_idx in range(len(retrieve_probs[0])):
                    if q_idx == 0:
                        ans_word_int = numpy.argmax(ans_train[train_trace_id][q_idx]) + 1
                        ans_word = inv_vocab_text[str(ans_word_int)]
                        ans_first_relate = ans_word.split(',')[1]
                        ans_second_relate = ans_word.split(',')[0]
                        ans_relate_int = working_memory.relationMap[ans_dict[ans_first_relate] - 1][ans_dict[ans_second_relate] - 1]
                        ans_shared = ans_relate_int
                    else:
                        ans_relate_int = ans_shared
                    reward_scalar_q, ans_q, label_q = compute_single_reward(retrieved_relation_list[q_idx],ans_relate_int)
                    reward_scalar_q *= q_mask_train[train_trace_id, q_idx, 0, 0]
                    
                    if ans_q == label_q and q_idx == 0:
                        epoch_precision_rate += 1.
                        story_precision += 1.

                    reward_q_total += reward_scalar_q
                reward_q_total += reward_temp_list[0]
                
                for time_slice in range(params['max_story_len']):
                    reward_immediate_train[train_trace_id, time_slice] += reward_q_total * (params['reward_decay'] ** (params['max_story_len'] - time_slice))
                for q_idx in range(len(retrieve_probs[0])):
                    reward_fin_train[train_trace_id, q_idx, :] *= reward_q_total # used as input at the last softmax
                    
                avg_reward_q_epoch = avg_reward_q_epoch * (train_trace_id) / (train_trace_id + 1) + reward_q_total / (train_trace_id + 1) # update E[r] for final q.
                
                for time_slice in range(params['max_story_len']):
                    reward_train[train_trace_id, time_slice, :] *= reward_immediate_train[train_trace_id, time_slice] # used as input at the last softmax
                    avg_reward_action_epoch[time_slice] = avg_reward_action_epoch[time_slice] * (train_trace_id) / (train_trace_id + 1) + reward_immediate_train[train_trace_id, time_slice] / (train_trace_id + 1)
                
                train_trace_id += 1
                all_sim_number += 1
                mask_sim *= 0
                working_memory.resetEnv()
                work_space_table.reset()

            sample_rate[story_id] = story_precision / params['story_sims_per_epoch']
        
        for q_idx in range(params['max_q_num']):
            if 0:
                reward_fin_train[:, q_idx, :] -= avg_reward_q_epoch 
        
        for time_slice in range(params['max_story_len']):
            if 0:
                reward_train[:, time_slice, :] -= avg_reward_action_epoch[time_slice] 
                
        epoch_precision_rate = epoch_precision_rate / (n_samples * params['story_sims_per_epoch'])

        
        sys.stdout = old_std
        print 'precision of this epoch: %f' %epoch_precision_rate
        print 'epoch %d phrase 2' %(epoch_id)
        print 'sample_rate:'
        print sample_rate
        # phrase2: go batch train on the trace pool.
        mask_sim_2 = numpy.ones((n_samples * params['story_sims_per_epoch'], params['max_story_len'], params['dim_emb_story'] * params['max_sent_len']))
        

        reasoner.fit([story_train, story_mask_train, q_train, q_mask_train, mask_sim_2, reward_train, reward_fin_train],
                      {'action_probs_re': act_selected_train, 'retrieve_probs_re': fin_one_hot_train},
                      batch_size=params['batch_size_phrase2'], nb_epoch=10, verbose=2)
        
        
        sys.stdout = old_std
                
        # test the model
        if (epoch_id + 1) % params['test_epoch_period'] == 0:
            test_model(simulator, story_int_test, story_mask_test, q_int_test, q_mask_test, ans_int_test, ans_mask_test, params, old_std, f_test, vocab_text, inv_vocab_text)
            
        sys.stdout = old_std
        story_train *= 0
        story_mask_train *= 0
        q_train *= 0
        q_mask_train *= 0    
        act_selected_train *= 0
        reward_train = numpy.ones((n_samples * params['story_sims_per_epoch'], params['max_story_len'], (params['ent_range']) * 1 * params['relation_num'])) # real number reward signal
        ans_train *= 0
        ans_mask_train *= 0
        fin_train *= 0
        fin_train_pred *= 0
        reward_fin_train = numpy.ones((n_samples * params['story_sims_per_epoch'], params['max_q_num'], (params['ent_range']) * 1 * 1))
        fin_one_hot_train *= 0
        reward_immediate_train *= 0
    
    f_print.close()
    f_debug.close()
    f_test.close()


def test_model(simulator, story_int, story_mask, q_int, q_mask, ans_int, ans_mask, params, old_std, f_test, vocab_text, inv_vocab_text):
    relate_dict = {'east':1, 'west':2, 'south':3, 'north':4}
    ent_list = {'1':'office', '2':'hallway', '3':'kitchen', '4':'garden', '5':'bedroom', '6':'bathroom'}
    ans_dict = {'e':1, 'w':2, 's':3, 'n':4}
    
    numpy.set_printoptions(precision=3)
    n_samples = len(story_int)
    acc = 0.
    
    sys.stdout = old_std
    print 'n_samples:'
    print n_samples
    

    story_test = numpy.zeros((1, params['max_story_len'], params['max_sent_len']))
    story_mask_test = numpy.zeros((1, params['max_story_len'], params['max_sent_len'], params['dim_emb_story']))
    q_test = numpy.zeros((1, params['max_q_num'], params['max_sent_len']))
    q_mask_test = numpy.zeros((1, params['max_q_num'], params['max_sent_len'], params['dim_emb_story']))
    

    act_selected_test = numpy.zeros((1, params['max_story_len'], (params['ent_range']) * 1 * params['relation_num'])) 
    reward_test = numpy.ones((1, params['max_story_len'], (params['ent_range']) * 1 * params['relation_num'])) 
    
    reward_immediate_test = numpy.zeros((1, params['max_story_len']))
    
    ans_test = numpy.zeros((1, params['max_q_num'], params['vocab_size']))
    ans_mask_test = numpy.zeros((1, params['max_q_num']))
    fin_test = numpy.zeros((1, params['max_q_num'], params['dim_emb_env']))
    fin_test_pred = numpy.zeros((1, params['max_q_num'], params['dim_emb_env'])) 
    fin_one_hot_test = numpy.zeros((1, params['max_q_num'], (params['ent_range']) * 1 * 1))
    reward_fin_test = numpy.ones((1, params['max_q_num'], (params['ent_range']) * 1 * 1))    
    
    working_memory = working_environment(params['enti_num'], params['relation_num_expand'])
    working_embed = envEmbedding(params['relation_num_expand'], params['enti_num'], params['dim_emb_env'])
    work_space_table = varTable()

    mask_sim = numpy.zeros((1, params['max_story_len'], params['dim_emb_story'] * params['max_sent_len']))
    
    print 'test started.'
    sys.stdout = f_test
    train_trace_id = 0 # always == 0
    for story_id in range(n_samples):
        # write story, q, ans, data     
        print '=======test-phrase====== story_id = %d' %(story_id)
        story_test[train_trace_id,:len(story_int[0]),:len(story_int[0][0])] = story_int[story_id]
        story_mask_test[train_trace_id,:len(story_int[0]),:len(story_int[0][0]),:] = story_mask[story_id]
        q_test[train_trace_id,:len(q_int[0]),:len(q_int[0][0])] = q_int[story_id]
        q_mask_test[train_trace_id,:len(q_int[0]),:len(q_int[0][0]),:] = q_mask[story_id]
        ans_test[train_trace_id,:len(ans_int[0]),:len(ans_int[0][0])] = ans_int[story_id]



        for time_slice in range(params['max_story_len']):
            mask_sim[0][time_slice][:] = numpy.ones(params['dim_emb_story'] * params['max_sent_len'])
            

            tupleList, adjGraph, temp_index = working_memory.returnEnv()



            action_probs, retrieve_probs = simulator.predict([story_test[numpy.newaxis,train_trace_id], 
                                                                      story_mask_test[numpy.newaxis,train_trace_id],
                                                                      q_test[numpy.newaxis,train_trace_id],
                                                                      q_mask_test[numpy.newaxis,train_trace_id],
                                                                      mask_sim[:], 
                                                                      reward_test[numpy.newaxis,train_trace_id],
                                                                      reward_fin_test[numpy.newaxis,train_trace_id]])


            action_selected, action_one_hot = select_action_hard(action_probs[:,time_slice,:], params['epsilon'])
            


            act_selected_test[train_trace_id, time_slice,:] = action_one_hot
            arg_1_ptr = action_selected // ((1) * params['relation_num']) # start from 0 (empty number)
            arg_2_ptr = (action_selected - arg_1_ptr * (1) * params['relation_num']) // params['relation_num']
            arg_r_ptr = (action_selected - arg_1_ptr * (1) * params['relation_num']) % params['relation_num'] + 1
            arg_r = arg_r_ptr
            
            flag = 0
            for w_id in story_int[story_id][time_slice]:
                if inv_vocab_text[str(int(w_id))] in ent_list.values() and flag == 0:
                    if arg_1_ptr == 0:
                        arg_1 = int(w_id)
                    else:
                        arg_2 = int(w_id)
                    flag = 1
                elif inv_vocab_text[str(int(w_id))] in ent_list.values() and flag == 1:
                    if arg_1_ptr == 0:
                        arg_2 = int(w_id)
                    else:
                        arg_1 = int(w_id)
            
            slice_reward = 0
            for tt in range(time_slice+1):    
                reward_immediate_test[train_trace_id][tt] += slice_reward * (params['reward_decay'] ** (time_slice - tt))
            

            
            if arg_1 > 0 and arg_2 > 0:
                # retrieve the table
                arg_1_int = work_space_table.retr_insert(arg_1, inv_vocab_text)
                arg_2_int = work_space_table.retr_insert(arg_2, inv_vocab_text)
                working_memory.modifyEnv((arg_1_int, arg_r, arg_2_int))
                

        retrieved_relation_list = []
        reward_temp_list = [] 
        for q_idx in range(len(retrieve_probs[0])):                    
            retr_idx = numpy.argmax(retrieve_probs[0,q_idx,:])
            arg1_retr_ptr = retr_idx // ((1) * 1)
            arg2_retr_ptr = (retr_idx - arg1_retr_ptr * (1) * 1) // 1
            relation_pred = (retr_idx - arg1_retr_ptr * (1) * 1) % 1 + 1                   
            
            
            flag = 0
            for q_w in q_int[story_id, 0]:
                if inv_vocab_text[str(int(q_w))] in ent_list.values() and flag == 0:
                    if arg1_retr_ptr == 0:
                        arg1_retr = int(q_w)
                    else:
                        arg2_retr = int(q_w)
                    flag = 1
                elif inv_vocab_text[str(int(q_w))] in ent_list.values() and flag == 1:
                    if arg1_retr_ptr == 0:
                        arg2_retr = int(q_w)
                    else:
                        arg1_retr = int(q_w)

            retrieve_reward_pre = 0
            reward_temp_list.append(retrieve_reward_pre)
            

                        
            arg1_retr_int = work_space_table.retr(arg1_retr, inv_vocab_text)
            arg2_retr_int = work_space_table.retr(arg2_retr, inv_vocab_text)
            
            relation_retr = working_memory.retrieveRelation(arg1_retr_int, arg2_retr_int)
            retrieved_relation_list.append(relation_retr)

                    
            one_hot_single = numpy.zeros(((params['ent_range']) * 1 * 1))
            one_hot_single[retr_idx] = 1
            fin_one_hot_test[train_trace_id, q_idx, :] = one_hot_single

            

        reward_q_total = 0.
        ans_shared = 0
        for q_idx in range(len(retrieve_probs[0])):
            if q_idx == 0:
                ans_word_int = numpy.argmax(ans_test[train_trace_id][q_idx]) + 1
                ans_word = inv_vocab_text[str(ans_word_int)]

                ans_first_relate = ans_word.split(',')[1]
                ans_second_relate = ans_word.split(',')[0]
                ans_relate_int = working_memory.relationMap[ans_dict[ans_first_relate] - 1][ans_dict[ans_second_relate] - 1]
                ans_shared = ans_relate_int
            else:
                ans_relate_int = ans_shared
            reward_scalar_q, ans_q, label_q = compute_single_reward(retrieved_relation_list[q_idx],ans_relate_int)
            reward_scalar_q *= q_mask_test[train_trace_id, q_idx, 0, 0]
            
            if ans_q == label_q and q_idx == 0:
                acc += 1.
                

            reward_q_total += reward_scalar_q
        reward_q_total += reward_temp_list[0]
        
        for time_slice in range(params['max_story_len']):
            reward_immediate_test[train_trace_id, time_slice] += reward_q_total * (params['reward_decay'] ** (params['max_story_len'] - time_slice))
        for q_idx in range(len(retrieve_probs[0])):
            # pass
            reward_fin_test[train_trace_id, q_idx, :] *= reward_q_total 
            
     
        
        for time_slice in range(params['max_story_len']):
            # pass
            reward_test[train_trace_id, time_slice, :] *= reward_immediate_test[train_trace_id, time_slice] # used as input at the last softmax
            

        mask_sim *= 0
        working_memory.resetEnv()
        work_space_table.reset()
        
    sys.stdout = old_std
    print 'test result:'
    print acc / n_samples



   

def set_hyper_params():
    params = {}
    params['dim_emb_story'] = 20 
    params['dim_emb_env'] = 20 
    params['dim_tracker'] = 20
    params['enti_num'] = 11 
    params['relation_num'] = 4
    params['relation_num_expand'] = 13
    params['dim_q_h_ff'] = 20
    params['dim_comp_ff'] = 20
    params['story_sims_per_epoch'] = 200
    params['epoch_num'] = 5000 
    params['epsilon'] = 0
    params['tho'] = 20
    params['reward_decay'] = 1
    params['round_error_epsilon'] = 1e-8
    params['long_dist_punish'] = -0.5
    params['batch_size_phrase2'] = 32
    params['test_epoch_period'] = 4
    params['print_epoch_period'] = 5
    return params



if __name__ == '__main__':
    train_file_names = ['../en/qa19_path-finding_train.txt']
    test_file_names = ['../en/qa19_path-finding_test.txt']
    params = set_hyper_params()
    train_two_phrase(train_file_names, test_file_names, params)    
    
    
               
                
                
                
    


    

    
    

    

        


    
    
