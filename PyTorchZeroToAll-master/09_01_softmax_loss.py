import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# Cross entropy example
import numpy as np
# One hot
# 0: 1 0 0
# 1: 0 1 0
# 2: 0 0 1
Y = np.array([1, 0, 0])

Y_pred1 = np.array([0.7, 0.2, 0.1]) #잘 예측된 값
Y_pred2 = np.array([0.1, 0.3, 0.6]) #잘 예측되지않은 값
print("loss1 = ", np.sum(-Y * np.log(Y_pred1)))
print("loss2 = ", np.sum(-Y * np.log(Y_pred2))) #여기 결과값으로써 cross-entrophy loss가 예측의 정도를 측정할수 있다

# Softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss() #pytorch에서는 이 함수안에 softmax와 cross-entrophy가 내장되어있어서 value들이 one-hot으로 되어있지않아도 된다.

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot => (one-hot)이라는거는 제일 큰놈만 1로 한다 는건가?
Y = Variable(torch.LongTensor([0]), requires_grad=False)  #=>여기서 torch.LongTensor([a,b,c])에서 a,b,c에 들어가는 거은 어떤애가 제일큰지를 나타냄

# input is of size nBatch x nClasses = 1 x 4
# Y_pred are logits (not softmax) => logit은 그냥 행렬곱만했을때의 결과
Y_pred1 = Variable(torch.Tensor([[2.0, 1.0, 0.1]]))
Y_pred2 = Variable(torch.Tensor([[0.5, 2.0, 0.3]]))

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("PyTorch Loss1 = ", l1.data, "\nPyTorch Loss2=", l2.data)

print("Y_pred1=", torch.max(Y_pred1.data, 1)[1])
print("Y_pred2=", torch.max(Y_pred2.data, 1)[1])

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = Variable(torch.LongTensor([2, 0, 1]), requires_grad=False)

# input is of size nBatch x nClasses = 2 x 4
# Y_pred are logits (not softmax)
Y_pred1 = Variable(torch.Tensor([[0.1, 0.2, 0.9],
                                 [1.1, 0.1, 0.2],
                                 [0.2, 2.1, 0.1]]))


Y_pred2 = Variable(torch.Tensor([[0.8, 0.2, 0.3],
                                 [0.2, 0.3, 0.5],
                                 [0.2, 0.2, 0.5]]))

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("Batch Loss1 = ", l1.data, "\nBatch Loss2=", l2.data)
