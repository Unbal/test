x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value

# our model forward pass

def forward(x):
    return x * w

# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# compute gradient  #직접 gradient를 계산해서 update시켜주는 꼴인것 같다.(따로 API안쓰고)
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)

# Before training
print("predict (before training)",  4, forward(4))

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data): #zip함수를 쓰면 [1,2,3],[2,4,6]의 배열이 [1,2],[2,4],[3,6]꼴로 나뉘어서 x_val,y_val에 저장된다.
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad #위에서 정의한 식으로 gradient계산한 다음에 가중치 update
        print("\tgrad: ", x_val, y_val, round(grad, 2)) #round함수는 소수점 n번째 자리에서 인자 반올림
        l = loss(x_val, y_val)

    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("predict (after training)",  "4 hours", forward(4))
