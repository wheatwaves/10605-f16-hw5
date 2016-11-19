"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *
from math import sqrt, floor
np.random.seed(0)
def create_target(m,n):
    T = np.zeros((m,n))
    r = np.random.rand(m,)
    for i in xrange(m):
        T[i][int(floor(r[i]*n % n))] = 1
    return T
class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size):
        self.max_len = max_len
        self.in_size = in_size
        self.num_hid = num_hid
        self.out_size = out_size
        self.batch_size = 1
        self.my_xman= self._build() #DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):
        xman = XMan()
        #TODO: define your model here
        X = [f.input(name = 'x' + str(t), default = np.random.rand(self.batch_size, self.in_size))
        for t in xrange(self.max_len)]

        c = f.input(name = 'c', default = np.zeros((self.batch_size, self.num_hid)))
        h = f.input(name = 'h', default = np.zeros((self.batch_size, self.num_hid)))

        Wi = f.param(name = 'Wi', default = sqrt(6.0/(self.in_size+self.num_hid))*np.random.uniform(-1,1,(self.in_size,self.num_hid)))
        Ui = f.param(name = 'Ui', default = sqrt(6.0/(self.num_hid+self.num_hid))*np.random.uniform(-1,1,(self.num_hid,self.num_hid)))
        bi = f.param(name = 'bi', default = 0.1*np.random.uniform(-1,1,(self.num_hid,)))

        Wf = f.param(name = 'Wf', default = sqrt(6.0/(self.in_size+self.num_hid))*np.random.uniform(-1,1,(self.in_size,self.num_hid)))
        Uf = f.param(name = 'Uf', default = sqrt(6.0/(self.num_hid+self.num_hid))*np.random.uniform(-1,1,(self.num_hid,self.num_hid)))
        bf = f.param(name = 'bf', default = 0.1*np.random.uniform(-1,1,(self.num_hid,)))

        Wo = f.param(name = 'Wo', default = sqrt(6.0/(self.in_size+self.num_hid))*np.random.uniform(-1,1,(self.in_size,self.num_hid)))
        Uo = f.param(name = 'Uo', default = sqrt(6.0/(self.num_hid+self.num_hid))*np.random.uniform(-1,1,(self.num_hid,self.num_hid)))
        bo = f.param(name = 'bo', default = 0.1*np.random.uniform(-1,1,(self.num_hid,)))

        Wc = f.param(name = 'Wc', default = sqrt(6.0/(self.in_size+self.num_hid))*np.random.uniform(-1,1,(self.in_size,self.num_hid)))
        Uc = f.param(name = 'Uc', default = sqrt(6.0/(self.num_hid+self.num_hid))*np.random.uniform(-1,1,(self.num_hid,self.num_hid)))
        bc = f.param(name = 'bc', default = 0.1*np.random.uniform(-1,1,(self.num_hid,)))

        for x in X:
            i_t = f.sigmoid(f.mul(x,Wi)+f.mul(h,Ui)+bi)
            f_t = f.sigmoid(f.mul(x,Wf)+f.mul(h,Uf)+bf)
            o_t = f.sigmoid(f.mul(x,Wo)+f.mul(h,Uo)+bo)
            c_t = f.tanh(f.mul(x,Wc)+f.mul(h,Uc)+bc)
            c = f.element_dot(f_t, c) + f.element_dot(i_t,c_t)
            h = f.element_dot(o_t,f.tanh(c))
        W = f.param(name = 'W', default = sqrt(6.0/(self.num_hid+self.out_size))*np.random.uniform(-1,1,(self.num_hid,self.out_size)))
        b = f.param(name = 'b', default = 0.1*np.random.uniform(-1,1,(self.out_size,)))
        xman.outputs = f.softMax(f.mul(h,W)+b)
        target = f.input(name = 'target', default = create_target(self.batch_size, self.out_size))
        xman.loss = f.crossEnt(xman.outputs, target)
        return xman.setup()

def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    # build
    print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    #TODO CHECK GRADIENTS HERE
    opseq = lstm.my_xman.operationSequence(lstm.my_xman.loss)
    autograd = Autograd(lstm.my_xman)
    # initDict = lstm.my_xman.inputDict()
    # print 'Wengart list:'
    # for (dstVarName, operator, inputVarNames) in opseq:
    #     print '  %s = %s(%s)' % (dstVarName,operator,(",".join(inputVarNames))) 
    # valueDict = autograd.eval(opseq,initDict)
    # gradientDict = autograd.bprop(opseq,valueDict,loss=1.0)
    # m, n= len(initDict['Wi']), len(initDict['Wi'][0])
    # for i in xrange(m):
    #     for j in xrange(n):
    #         low_W2, high_W2 = initDict['Wi'].copy(), initDict['Wi'].copy()
    #         low_W2[i][j] = initDict['Wi'][i][j] - 10e-4
    #         high_W2[i][j] = initDict['Wi'][i][j] + 10e-4
    #         lowDict = autograd.eval(opseq, lstm.my_xman.inputDict(Wi=low_W2))
    #         highDict = autograd.eval(opseq, lstm.my_xman.inputDict(Wi=high_W2))
    #         if highDict['loss'] != lowDict['loss']:
    #             print gradientDict['Wi'][i][j]
    #             print (highDict['loss']-lowDict['loss']) / (2*(10e-4))               
    #             print '------------------------'
    # return
    # train
    # get default data and params
    initDict = lstm.my_xman.inputDict()
    lr = init_lr
    bestDict = {}
    best_loss = 10000
    for i in range(epochs):
        training_loss, validation_loss = [], []
        for (idxs,e,l) in mb_train:
            #TODO prepare the input and do a fwd-bckwd pass over it and update the weights
            for t in xrange(max_len):
                initDict['x'+str(t)] = e[:,[t],:].reshape(len(l), mb_train.num_chars)
            initDict['target'] = l
            initDict['c'] = np.zeros((len(l), num_hid))
            initDict['h'] = np.zeros((len(l), num_hid))
            valueDict = autograd.eval(opseq, initDict)
            training_loss.append(valueDict['loss'])
            gradientDict = autograd.bprop(opseq, valueDict, loss=np.float_(1.))
            for rname in gradientDict:
                if lstm.my_xman.isParam(rname):
                    initDict[rname] = initDict[rname] - lr*gradientDict[rname]
        # print str(i) + "th eopch training error = " + str(np.mean(np.array(training_loss)))
        # validate
        for (idxs,e,l) in mb_valid:
            #TODO prepare the input and do a fwd pass over it to compute the loss
            for t in xrange(max_len):
                initDict['x'+str(t)] = e[:,[t],:].reshape(len(l), mb_train.num_chars)
                initDict['target'] = l
            initDict['c'] = np.zeros((len(l), num_hid))
            initDict['h'] = np.zeros((len(l), num_hid))
            valueDict = autograd.eval(opseq, initDict)
            validation_loss.append(valueDict['loss'])
        # print str(i) + "th eopch validation error = " + str(np.mean(np.array(validation_loss)))
        loss = np.mean(validation_loss)
        if loss < best_loss:
            bestDict = initDict.copy()
        #TODO compare current validation loss to minimum validation loss
        # and store params if needed
    print "done"
    output_probabilities = []
    test_loss = []
    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        for t in xrange(max_len):
            bestDict['x'+str(t)] = e[:,[t],:].reshape(len(l), mb_train.num_chars)
            bestDict['target'] = l
        bestDict['c'] = np.zeros((len(l), num_hid))
        bestDict['h'] = np.zeros((len(l), num_hid))
        valueDict = autograd.eval(opseq, initDict)
        for i in valueDict['outputs']:
            for j in xrange(len(i)):
                i[j] = exp(-i[j])
            output_probabilities.append(i)
        test_loss.append(valueDict['loss'])        
    #TODO save probabilities on test set
    # ensure that these are in the same order as the test input
    np.save(output_file, output_probabilities)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='../data/tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    params = vars(parser.parse_args())
    main(params)
