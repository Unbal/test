import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [4.0, 6.0, 8.0]

w = Variable(torch.Tensor([1.0]),  requires_grad=True)  # Any random value #여기서부터는 파이토치 쓴다!
v = Variable(torch.Tensor([1.0]),  requires_grad=True)  # Any random value #여기서부터는 파이토치 쓴다!

# our model forward pass

def forward(x):
    return x * w 

def forward2(x):
    return x + v
# Loss function
def loss(x, y):
    temp = forward(x)
    y_pred = forward2(temp)
    return (y_pred - y) * (y_pred - y)

# Before training
print("predict (before training)",  4, forward(4).data[0])

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward() #backward라는 애를 쓰면 모든 그래프내의 변수가 해당 loss에 대해서 얼마나 바뀌는 지를 계산해준다.
        print("\tgrad: ", x_val, y_val, w.grad.data[0], v.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data
        v.data = v.data - 0.01 * v.grad.data

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()
        v.grad.data.zero_()

    print("progress:", epoch, l.data[0])

# After training
print("predict (after training)",  4, forward(4).data[0])
