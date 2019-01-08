"""
Created on Thu Dec 21 16:22:56 2017

@author: Jiajie
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn.functional as F
from torch.autograd import Variable


def bootstrap_cross_entropy(input, target, ishard=False, beta=0.95, weight=None, size_average=True):
    r"""Function that measures Cross Entropy between target and output
    logits with prediction consistency(bootstrap)

    Args:
        input: Variable of arbitrary shape
        target: Variable :math:`(N)` where each value is
            `0 <= targets[i] <= C-1
        ishard: Choose soft/hard bootstrap mode
        beta: Weight between ``gt`` label and prediction. In paper, 0.8 for hard and 0.95 for soft
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Default: ``True``

    Examples::

         >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
         >>> target = autograd.Variable(torch.LongTensor(3).random_(5))
         >>> loss = bootstrap_cross_entropy(input, target)
         >>> loss.backward()
    """
    input_prob = F.softmax(input)
    target_onehot = Variable(input.data.new(input.data.size()).zero_())
    target_onehot.scatter_(1, target.view(-1,1), 1)
#    print(target_onehot)
    if ishard:
        _,idx = input_prob.max(1)
        target_onehot = target_onehot * beta + \
                    Variable(input.data.new(input.data.size()).zero_()).scatter_(1, idx.view(-1,1), 1) * (1-beta)
    else:
        target_onehot = target_onehot * beta + input_prob * (1-beta)
    loss = - target_onehot * F.log_softmax(input)
    #print(loss.size())
    #print(weight.size())
    #if weight is not None:
    #    loss = loss.sum(1) * weight

    if size_average:
        if weight is not None:
            return (loss.sum(1) * weight).mean()
        return loss.sum(1).mean()
    else:
        return loss.sum()
    
    


def BCE_bootstrap_with_logits(input, target, ishard=False, beta=0.95, weight=None, size_average=True):
    r"""Function that measures Binary Cross Entropy between target and output
    logits with prediction consistency(bootstrap)

    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        ishard: Choose soft/hard bootstrap mode
        beta: Weight between ``gt`` label and prediction. In paper, 0.8 for hard and 0.95 for soft
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Default: ``True``

    Examples::

         >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
         >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
         >>> loss = BCE_bootstrap_with_logits(input, target)
         >>> loss.backward()
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
    input_prob = torch.sigmoid(input)
    if ishard:
        target = target * beta + (input_prob>0.5) * (1-beta)
    else:
        target = target * beta + input_prob * (1-beta)
    print(target)
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if size_average:
        return loss.mean()
    else:
        return loss.sum()