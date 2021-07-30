import numpy as np
import matplotlib.pyplot as plt

# dataset 4
n = 200
x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:,1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y_d4 = 2 * y_d4 -1

#hyper parameter
lmd = 0.01

#parameter
a = np.ones(n) * 0.5

def calc_w(a):
    temp = np.zeros(x_d4[0].shape[0])
    for i in range(n):
        temp += a[i]*y_d4[i]*x_d4[i]
    return temp/(2*lmd)

w = calc_w(a)

d = x_d4[0].shape[0]
K = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        K[i,j] = y_d4[i]*y_d4[j]*np.dot(x_d4[i],x_d4[j])

def obj_w(w):
    temp = 0
    for i in range(n):
        temp += max(0, 1-y_d4[i]*np.dot(w,x_d4[i]))
    temp += lmd * np.dot(w,w)
    return temp

def obj_a(a):
    return -np.dot(a, np.dot(K,a.T)) / (4*lmd) + np.dot(a,np.ones(n))

gap_hist = []
rate = 0.01

def update(epoch):
    global a,w,gap_hist
    for e in range(epoch):
        for i in range(n):
            a[i] -= rate*(np.dot(K[i,:],a)/(2*lmd) - 1)
            a[i] = max(0, min(a[i], 1))
        w = calc_w(a)
        gap_hist.append(obj_w(w) - obj_a(a))

epoch = 1000
update(epoch)

#graph

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(epoch), gap_hist, label='batch')
ax.set_yscale('log')
ax.set_ylim(pow(10,-1),pow(10,4))

fig.savefig('3-3.png')