"""
Multilayer Perceptron for character level entity classification
"""
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *
from math import sqrt, floor
from functions import *
np.random.seed(0)
def create_target(m,n):
    T = np.zeros((m,n))
    r = np.random.rand(m,)
    for i in xrange(m):
        T[i][int(floor(r[i]*n % n))] = 1
    return T
class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        self.input_d = layer_sizes[0]
        self.hidden_d = layer_sizes[1]
        self.output_d = layer_sizes[2]
        self.batch_size = 3
        self.my_xman = self._build() # DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable
    def _build(self):
        x = XMan()
        x.X = f.input(name = 'X', default = np.random.rand(self.batch_size, self.input_d))
        a = sqrt(6.0/(self.input_d+self.hidden_d))
        x.W1 = f.param(name = 'W1', default = np.random.uniform(-a,a,(self.input_d, self.hidden_d)))
        x.b1 = f.param(name = 'b1', default = np.random.uniform(-0.1*a,0.1*a,(self.hidden_d,)))
        b = sqrt(6.0/(self.hidden_d+self.output_d))
        x.W2 = f.param(name = 'W2', default = np.random.uniform(-b,b,(self.hidden_d, self.output_d)))
        x.b2 = f.param(name = 'b2', default = np.random.uniform(-0.1*b,0.1*b,(self.output_d,)))


        x.O1 = f.relu(f.mul(x.X, x.W1)+x.b1)
        x.O2 = f.relu(f.mul(x.O1, x.W2)+x.b2)
        x.target = f.input(name = 'target', default = create_target(self.batch_size, self.output_d))
        x.outputs = f.softMax(x.O2)
        x.loss = f.crossEnt(x.outputs, x.target) 


        #TODO define your model here
        return x.setup()

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
    # print "building mlp..."
    mlp = MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
    opseq = mlp.my_xman.operationSequence(mlp.my_xman.loss)
    autograd = Autograd(mlp.my_xman)
    # #TODO CHECK GRADIENTS HERE    
    # print 'Wengart list:'
    # for (dstVarName, operator, inputVarNames) in opseq:
    #     print '  %s = %s(%s)' % (dstVarName,operator,(",".join(inputVarNames))) 
    # print "done"
    # initDict = mlp.my_xman.inputDict()
    # autograd = Autograd(mlp.my_xman)
    # valueDict = autograd.eval(opseq,initDict)
    # gradientDict = autograd.bprop(opseq,valueDict,loss=1.0)
    # m = len(initDict['b1'])
    # for i in xrange(m):
    #     low_W2, high_W2 = initDict['b1'].copy(), initDict['b1'].copy()
    #     low_W2[i] = initDict['b1'][i] - 10e-4
    #     high_W2[i] = initDict['b1'][i] + 10e-4
    #     lowDict = autograd.eval(opseq, mlp.my_xman.inputDict(b1=low_W2))
    #     highDict = autograd.eval(opseq, mlp.my_xman.inputDict(b1=high_W2))
    #     if highDict['loss'] != lowDict['loss']:
    #         print gradientDict['b1'][i]
    #         print (highDict['loss']-lowDict['loss']) / (2*(10e-4))               
    #         print '------------------------'
    # return
    # train
    # get default data and params
    initDict = mlp.my_xman.inputDict()
    lr = init_lr
    bestDict = {}
    best_loss = 10000
    for i in range(epochs):
        training_loss, validation_loss = [], []
        for (idxs,e,l) in mb_train:
            #TODO prepare the input and do a fwd-bckwd pass over it and update the weights
            initDict['X'] = e.reshape(len(l), max_len*mb_train.num_chars)
            initDict['target'] = l
            valueDict = autograd.eval(opseq, initDict)
            training_loss.append(valueDict['loss'])
            gradientDict = autograd.bprop(opseq, valueDict, loss=np.float_(1.))
            for rname in gradientDict:
                if mlp.my_xman.isParam(rname):
                    initDict[rname] = initDict[rname] - lr*gradientDict[rname]

        for (idxs,e,l) in mb_valid:
            #TODO prepare the input and do a fwd pass over it to compute the loss
            initDict['X'] = e.reshape(len(l), max_len*mb_train.num_chars)
            initDict['target'] = l
            valueDict = autograd.eval(opseq, initDict)
            validation_loss.append(valueDict['loss'])
        loss = np.mean(validation_loss)
        if loss < best_loss:
            bestDict = initDict.copy()
        #TODO compare current validation loss to minimum validation loss
        # and store params if needed

    # print "done"
    output_probabilities = []
    for (idxs,e,l) in mb_test:
        bestDict['X'] = e.reshape(len(l), max_len*mb_train.num_chars)
        bestDict['target'] = l
        valueDict = autograd.eval(opseq, initDict)
        for i in valueDict['outputs']:
            for j in xrange(len(i)):
                i[j] = exp(-i[j])
            output_probabilities.append(i)     
        # prepare input and do a fwd pass over it to compute the output probs
        
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
