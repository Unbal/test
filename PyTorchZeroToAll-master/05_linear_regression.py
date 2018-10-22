
import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]])) #금방까지한것들은 w라는 변수를 둔거였고, 이제부터는 pytorch API를 적극활용해보자
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


class Model(torch.nn.Module): #이제부터는 이렇게 클래스를 만들어서 실행해보자

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # input과 output의 차원을 적어준다.

    def forward(self, x):#forward,즉 우리가 원하는 결과값을 얻을수 있는 방법?route?를 적어준다.
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x) #위에서 정의한 함수들을 시작! 근데 얘네가 무슨 함수인지는 신경안쓰는건가? 걍 weighted_sum?
        return y_pred

# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False) # 얘는 loss계산해주는애 (MSE랑 또 무슨게 있는지 모르겠다)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) ##얘는 가중치 최신화(SGD랑 또 뭐가 있을까), lr = learning rate

# Training loop
for epoch in range(100):
        # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() #model안에 있는 모든 variable에 대해서 0으로 초기화 시켜주는 함수
    loss.backward()
    optimizer.step() #이거는 loss.backward()에서 얻은 loss들로 optimizer선언시에 지정해준 model내의 변수들을 update하는 과정


# After training
hour_var = Variable(torch.Tensor([[4.0]]))
y_pred = model(hour_var)
print("predict (after training)",  4, model(hour_var).data[0][0])
