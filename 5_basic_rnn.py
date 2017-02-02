import numpy as np
import tensorflow as tf

input_string="barackobama"

string_to_char=list(input_string)
set_of_char=set(string_to_char)
look_up_table=list(set_of_char)

dict_char={ c:idx for idx,c in enumerate(set_of_char)}

x1=[dict_char[i] for i in input_string[:-1]]
y1=[dict_char[i] for i in input_string[1:]]

length=len(x1)

def index_to_onehot(x,length):
	arr=[]
	for i in x:
		new_arr=[int(i==j) for j in range(length)]
		arr.append(new_arr)
	return arr	

input_x  =np.array(index_to_onehot(x1,length),dtype=np.float32)
#output_y_=np.array(index_to_onehot(y1,length),dtype=np.float32)
output_y_=y1

num_units=length
batch_size=1
learning_rate=1e-2

rnn_cell=tf.nn.rnn_cell.BasicRNNCell(num_units=num_units)
hidden_state_initial=rnn_cell.zero_state(batch_size,dtype=tf.float32)
input_x_split=tf.split(0,length,input_x)

# https://github.com/ahangchen/TensorFlowDoc/blob/master/api_docs/python/functions_and_classes/shard0/tf.nn.rnn.md
outputs, state = tf.nn.rnn(rnn_cell, input_x_split, hidden_state_initial)

logits = tf.reshape(tf.concat(1, outputs), [-1, num_units])
targets = tf.reshape(output_y_, [-1])
weights = tf.ones([length* batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

# https://github.com/tensorflow/tensorflow/blob/287db3a9b0701021f302e7bb58af5cf89fdcd424/tensorflow/g3doc/api_docs/python/functions_and_classes/shard5/tf.contrib.legacy_seq2seq.sequence_loss_by_example.md
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(dict_char)
	print(len(sess.run(outputs)))
	print(sess.run([logits]))
	print(sess.run([targets]))

	for i in range(100):
		sess.run(train_op)
		result = sess.run(tf.argmax(logits, 1))
		print(result)
	print(result, [look_up_table[t] for t in result])	