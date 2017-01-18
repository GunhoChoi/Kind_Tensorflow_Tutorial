import numpy as np
import matplotlib.pyplot as plt

N = 100 
D = 2 
K = 3 
X = np.zeros((N*K,D)) 
y = np.zeros(N*K, dtype='uint8') 

# data generation
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N)  
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)] 
  y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.Spectral)
plt.show()

# hpyerparameter setting

H=100 # Hidden layer node 
w1=0.01*np.random.rand(D,H)/np.sqrt(D/2)
b1=np.zeros([1,H])
w2=0.01*np.random.rand(H,3)/np.sqrt(H/2)
b2=np.zeros([1,K])

learning_rate=1e-2
reg_lambda=1e-2
iteration=10000
num_examples=X.shape[0]

for i in range(10000):
    
    # forward propagation
    hidden_layer=np.maximum(0,np.dot(X,w1)+b1)
    output_layer= np.dot(hidden_layer,w2)+b2

    # softmax classifier
    exp_scores=np.exp(output_layer)
    probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    
    # loss calculation
    cross_entropy_loss= -np.sum(1.0*np.log(probs[range(num_examples),y]))
    data_loss=cross_entropy_loss/num_examples
    reg_loss=1/2*reg_lambda*(np.sum(w1*w1)+np.sum(w2*w2))
    
    loss=data_loss+reg_loss
    
    # error calculation
    errors=probs
    errors[range(num_examples),y]-=1
    
    # backpropagation
    # output layer -> hidden layer
    dw2=np.dot(hidden_layer.T,errors)
    db2=np.sum(errors,axis=0,keepdims=True)

    #ReLU backprop
    dhidden_error=np.dot(errors,w2.T)
    dhidden_error[hidden_layer<=0]=0

    # hidden layer -> input layer
    dw1=np.dot(X.T,dhidden_error)
    db1=np.sum(dhidden_error,axis=0,keepdims=True)

    # derivative (1/2*w^2) = w
    dw2+=reg_lambda*w2
    dw1+=reg_lambda*w1

    # model update
    w1+= -learning_rate*dw1
    b1+= -learning_rate*db1
    w2+= -learning_rate*dw2
    b2+= -learning_rate*db2

hidden_layer = np.maximum(0, np.dot(X, w1) + b1)
scores = np.dot(hidden_layer, w2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], w1) + b1), w2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()