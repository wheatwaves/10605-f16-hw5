# some useful functions
import numpy as np
from xman import *
from math import exp
from math import log
# some useful functions
# declare all operations here first

def eval_mul(x1,x2):
    R = np.dot(x1, x2)
    return R
def eval_relu(M):
    m, n = len(M), len(M[0])
    R = np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            if M[i][j] > 0:
                R[i][j] = M[i][j]
    return R
def bp_relu(delta, out, x):
    m, n = len(x), len(x[0])
    G = np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            if x[i][j] > 0:
                G[i][j] = delta[i][j]
    return G

def eval_softMax(O):
    m, n = len(O), len(O[0])
    P = np.zeros((m,n))
    for i in xrange(m):
        a = O[i][0]
        for j in xrange(n):
            a = max(a, O[i][j])
        sums = .0
        for j in xrange(n):
            sums += exp(O[i][j]-a)
        for j in xrange(n):
            P[i][j] = -log(exp(O[i][j]-a) / sums)
    return P

def eval_crossEnt(P, T):
    m, n = len(P), len(P[0])
    loss = np.zeros((m))
    for i in xrange(m):
        loss[i] = P[i].dot(np.transpose(T[i]))
    return np.mean(loss)

def bp_crossEnt_softMax_O(delta, out, O, T):
    m, n = len(O), len(O[0])
    P = np.zeros((m,n))
    sums = np.zeros((m,))
    a_val = np.zeros((m,))
    for i in xrange(m):
        a = O[i][0]
        for j in xrange(n):
            a = max(a, O[i][j])
        a_val[i] = a
        sum_v = .0
        for j in xrange(n):
            sum_v += exp(O[i][j]-a)
        sums[i] = sum_v
        for j in xrange(n):
            P[i][j] = -log(exp(O[i][j]-a) / sum_v)
    G = np.zeros((m, n))
    for i in xrange(m):
        flag = -1
        for j in xrange(n):
            if T[i][j] == 1:
                flag = j
                break
        flag_v = exp(O[i][flag]-a_val[i])
        for j in xrange(n):
            if j == flag:
                G[i][j] = (delta / m) * (-1.0/P[i][j]) * ((flag_v*(sums[i]-flag_v))/(sums[i]*sums[i]))  
            else:
                exp_v = exp(O[i][j]-a_val[i])     
                G[i][j] = (delta / m) * (-1.0/P[i][j]) * ((-flag_v*exp_v)/(sums[i]*sums[i]))
    return G

def bp_crossEnt_softMax_T(delta, out, O, T):
    m, n = len(O), len(O[0])
    P = np.zeros((m,n))
    for i in xrange(m):
        a = O[i][0]
        for j in xrange(n):
            a = max(a, O[i][j])
        sum = .0
        for j in xrange(n):
            sum += exp(O[i][j]-a)
        for j in xrange(n):
            P[i][j] = -log(exp(O[i][j]-a) / sum)
    G = np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            if T[i][j] == 1:
                G[i][j] = delta * P[i][j] / m
    return G 


class f(XManFunctions):
    @staticmethod
    def square(a):
        return XManFunctions.registerDefinedByOperator('square',a)
    @staticmethod
    def relu(a):
        return XManFunctions.registerDefinedByOperator('relu',a)
    @staticmethod
    def crossEnt(a, b):
        return XManFunctions.registerDefinedByOperator('crossEnt',a,b)
    @staticmethod
    def softMax(a):
        return XManFunctions.registerDefinedByOperator('softMax',a)
    # TODO add other operation registers

# the functions that autograd.eval will use to evaluate each function,
# to be called with the functions actual inputs as arguments

EVAL_FUNS = {
    'add':      lambda x1,x2: x1+x2,
    'subtract': lambda x1,x2: x1-x2,
    'square':   np.square,
    'mul' : eval_mul, 
    'relu': eval_relu,
    'crossEnt': eval_crossEnt,
    # TODO
    'softMax':  eval_softMax
    # TODO
    # TODO other operations
    }

# the functions that autograd.bprop will use in reverse mode
# differentiation.  BP_FUNS[f] is a list of functions df1,....,dfk
# where dfi is used in propagating errors to the i-th input xi of f.
# Specifically, dfi is called with the ordinary inputs to f, with two
# additions: the incoming error, and the output of the function, which
# was computed by autograd.eval in the eval stage.  dfi will return
# delta * df/dxi [f(x1,...,xk)]
# 
# NOTE: Autograd has an optimization where if it finds a softMax op
# followed by crossEnt op, it combines the backward pass for both. So
# you only need to implement the BP_FUNS for the combined operation 
# crossEnt-softMax below.

def _derivAdd(delta,x1):
    if delta.shape!=x1.shape:
        # broadcast, sum along axis=0
        if delta.shape[1]!=x1.shape[0]:
            raise ValueError("Dimension Mismatch")
        return delta.sum(axis=0) #we sum the gradients over the batch
    else: return delta

BP_FUNS = {
    'add':              [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: _derivAdd(delta,x2)],
    'subtract':         [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: -_derivAdd(delta,x2)],
    'square':           [lambda delta,out,x : delta * 2.0 * x],
    'relu': [bp_relu],
    'mul': [lambda delta,out,x1,x2:delta.dot(np.transpose(x2)), lambda delta,out,x1,x2:np.transpose(x1).dot(delta)],
    'crossEnt-softMax': [bp_crossEnt_softMax_O, bp_crossEnt_softMax_T]
     # TODO
    # TODO other operations
    }
