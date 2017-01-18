import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
H=100 
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix]= j

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.Spectral)
plt.show()

# reshape y
new_y=np.zeros([N*K,K])
for i in range(N*K):
	new_y[i,y[i]]=1


learning_rate=1e-2
reg_lambda=1e-2
iteration=10000
num_examples=X.shape[0]


x =tf.placeholder(shape=[300,2],dtype=tf.float32,name='x') # feed X
y_=tf.placeholder(shape=[300,3],dtype=tf.float32,name='y_') # feed new_y

w1=tf.Variable(0.01*np.random.rand(D,H)/np.sqrt(D/2),dtype=tf.float32,name='w1')
b1=tf.Variable(np.zeros([1,H]),dtype=tf.float32,name='b1' )
w2=tf.Variable(0.01*np.random.rand(H,3)/np.sqrt(H/2),dtype=tf.float32,name='w2')
b2=tf.Variable(np.zeros([1,K]),dtype=tf.float32,name='b2' )

hidden_layer= tf.nn.relu(tf.matmul(x,w1)+b1)
output_layer=tf.matmul(hidden_layer,w2)+b2
cross_entropy_loss=tf.nn.softmax_cross_entropy_with_logits(output_layer,y_)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_layer,1),tf.argmax(y_,1)),dtype=tf.float32))

feed_dict={x:X,y_:new_y}
init= tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for x in range(iteration):
		_,acc=sess.run([train_step,accuracy],feed_dict=feed_dict)
		if x%1000==0:
			print(acc)

	w1=w1.eval()
	b1=b1.eval()
	w2=w2.eval()
	b2=b2.eval()

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