import numpy as np

# f = w * x

#f = 2 * x
X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8],dtype=np.float32)

w = 0.0
print(w)

# model prediction
def forward(x):
    return w*x

#loss = MSE
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

#training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    #Prediction
    y_pred = forward(X)
    #Loss
    l = loss(Y,y_pred)
    #Gradient
    dw = gradient(X,Y,y_pred)
    #Update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')