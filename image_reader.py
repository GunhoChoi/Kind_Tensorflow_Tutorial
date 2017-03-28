import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size =16
epoch = 10 

# Read image files by name

shoes_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./shoes/*.jpg"),num_epochs=epoch)
bags_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./bags/*.jpg"),num_epochs=epoch)

# Define Reader

image_reader = tf.WholeFileReader()

# reader returns filename & image data

_, shoes_file = image_reader.read(shoes_filename_queue)
_, bags_file = image_reader.read(bags_filename_queue)

# decode data with decode_jpg function

shoes_image = tf.image.decode_jpeg(shoes_file)
bags_image = tf.image.decode_jpeg(bags_file)

# change shape and data type according to usage

shoes_image = tf.cast(tf.reshape(shoes_image,shape=[64,64,3]),dtype=tf.float32)
bags_image = tf.cast(tf.reshape(bags_image,shape=[64,64,3]),dtype=tf.float32)

# make shuffled batch with tf.train.shuffle_batch

num_preprocess_threads = 1
min_queue_examples = 256
batch_shoes = tf.train.shuffle_batch([shoes_image],
				    batch_size=batch_size,
				    num_threads=num_preprocess_threads,
				    capacity=min_queue_examples + 3 * batch_size,
				    min_after_dequeue=min_queue_examples)

batch_bags = tf.train.shuffle_batch([bags_image],
				    batch_size=batch_size,
				    num_threads=num_preprocess_threads,
				    capacity=min_queue_examples + 3 * batch_size,
				    min_after_dequeue=min_queue_examples)

with tf.Session() as sess:
	for i in range(epoch):
		print(batch_shoes,batch_bags)
