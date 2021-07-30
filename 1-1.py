import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv

# dataset 4
n = 200
x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:,1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y_d4 = 2 * y_d4 -1

#model
w = np.array([1,1,1,1])

#sigmoid_vector
def sigmoid_vector(y, w_dot_x):
    return 1/(1+np.exp(-y*w_dot_x))

lmd = 0.01

#loss function
def J(x, y, w):
    w_dot_x = np.dot(x,w.T)
    sig_vec = sigmoid_vector(y, w_dot_x)
    entropy_vec = -np.log(sig_vec)
    return np.sum(entropy_vec) + lmd*np.dot(w,w.T)

#gradient
def J_grad(x, y, w):
    w_dot_x = np.dot(x,w)
    sig_vec = sigmoid_vector(y, w_dot_x)
    #coefficient vector
    co_vec = -y*(-sig_vec+1)
    return np.sum(np.dot(np.diag(co_vec), x), axis=0) + 2*lmd*w

def vectorize_mats(x1,x2):
    shape = x1.shape
    n = shape[0]
    d = shape[1]
    xx1 = np.tile(x1,(1,d)).reshape(1,n*d*d)
    xx2 = np.tile(x2.reshape(n*d,1),(1,d)).reshape(1,n*d*d)
    return (xx1*xx2).reshape(n,d,d)

#hessian
def J_hessian(x, y, w):
    shape = x.shape
    d = shape[1]
    w_dot_x = np.dot(x,w)
    sig_vec = sigmoid_vector(y, w_dot_x)
    #coefficient vector
    co_vec = sig_vec*(-sig_vec+1)
    return np.sum(vectorize_mats(np.dot(np.diag(co_vec), x), x), axis=0) + 2*lmd*np.diag(np.ones(d))

#print(J_hessian(x_d4, y_d4, w))

#train w under this comment
epoch = 50
#history of lost in each method
batch_hist = []
newton_hist = []

#upper bound of Liptitz constant of the gradient
max_hessian = np.sum(0.25*vectorize_mats(x_d4, x_d4), axis=0) + 2*lmd*np.diag(np.ones((x_d4.shape[1])))
lip = np.linalg.norm(max_hessian, 2)
print(lip)
rate = 1/(lip)

def train_batch(x,y,w,epoch):
    global batch_hist
    batch_hist = []
    for i in np.arange(epoch):
        w = w - rate * J_grad(x, y, w)
        batch_hist.append(J(x, y, w))
    return w

def train_newton(x,y,w,epoch):
    global newton_hist
    newton_hist = []
    for i in np.arange(epoch):
        w = w - (np.linalg.inv(J_hessian(x, y, w))).dot(J_grad(x, y, w))
        newton_hist.append(J(x, y, w))
    return w

w1 = train_batch(x_d4, y_d4, w, epoch)
w2 = train_newton(x_d4, y_d4, w, epoch)

J_min = min(np.array(batch_hist).min(), np.array(newton_hist).min())
batch_hist = batch_hist - J_min 
newton_hist = newton_hist - J_min 
print(w1)
print(w2)

#graph

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(epoch), batch_hist, label='batch')
ax.plot(np.arange(epoch), newton_hist, label='newton')
ax.set_yscale('log')
ax.set_ylim(pow(10,-15),pow(10,2))

fig.savefig('1-1.png')