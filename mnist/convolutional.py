# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:07:05 2018

@author: jiang
"""

import os
import tensorflow as tf
import numpy as np
import gzip

from six.moves import urllib

SOURSE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = 'mnist_data/'
IMAGE_SIZE = 28
PIXEL_DEPTH = 255
NUM_CHANNELS = 1
NUM_LABELS = 10
BATCH_SIZE = 64
NUM_EPOCHS = 10
VALIDATION_SIZE = 5000
SEED = None
EVAL_FREQUENCY = 100

def maybe_download(filename):
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY + filename)
    print(filepath)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURSE_URL+filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            print('Successfully download', filename, f.size)
    return filepath

def extract_data(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as byte_stream:
        byte_stream.read(16)
        buff = byte_stream.read(num_images*IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS)
        data = np.frombuffer(buff, np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH) / 2.0) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data
    
def extract_labels(filename, num_labels):
    print('Extracting', filename)
    with gzip.open(filename) as byte_stream:
        byte_stream.read(8)
        buff = byte_stream.read(num_labels)
        labels = np.frombuffer(buff, np.uint8).astype(np.int8)
    return labels

def fake_data(num_images):
    data = np.ndarray([num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], np.float32)
    labels = np.zeros([num_images, 1], np.int64)
    for image in range(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image, 0] = label
    return data, labels

def error_rate(predictions, labels):
    return (100.0 - 100.0*(np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))) / predictions.shape[0])

def one_hot(labels):
    return (np.arange(10) == np.array(labels).reshape([-1, 1]))

train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

train_data = extract_data(train_data_filename, 60000)
train_labels = one_hot(extract_labels(train_labels_filename, 60000))
print(train_labels)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

validation_data = train_data[:VALIDATION_SIZE, ...]
validation_labels = train_labels[:VALIDATION_SIZE, ...]
train_data = train_data[VALIDATION_SIZE:, ...]
train_labels = train_labels[VALIDATION_SIZE:, ...]

graph = tf.Graph()
dropout = True
with graph.as_default():    
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    y = tf.placeholder(tf.int64, [None, NUM_LABELS])
    conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 *IMAGE_SIZE // 4 * 64, 512], stddev=0.1))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
        
    
    conv = tf.nn.conv2d(x, conv1_weights, [1, 1, 1, 1], 'SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(pool, conv2_weights, [1, 1, 1, 1], 'SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [-1, shape[1]*shape[2]*shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    if dropout:
        hidden = tf.nn.dropout(hidden, keep_prob=0.5)
    logits =  tf.matmul(hidden, fc2_weights) + fc2_biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    loss += 5e-4 * regularizers
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.01, batch*BATCH_SIZE, train_labels.shape[0], 0.95, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
    
    train_prediction = tf.nn.softmax(logits)

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print('initialized')
    iteration = train_data.shape[0] // BATCH_SIZE
    start = 0
    end = 0
    for epoch in range(NUM_EPOCHS):
        for i in range(iteration):
            end = start + BATCH_SIZE
            batch_x = train_data[start:end, ...]
            batch_y = train_labels[start:end, ...]
            feed_dict = {x:batch_x, y:batch_y}
            _, losses = sess.run([optimizer, loss], feed_dict=feed_dict)
            print('step : %d  cost : %.5f' %(i, losses))
            







    
        


if __name__ == '__main__':
    maybe_download('train-images-idx3-ubyte.gz')
    data = extract_data('train-images-idx3-ubyte.gz', 1000)
    d, l = fake_data(1000)
    prediction = np.array([[1], [2], [3]])
    a = error_rate(prediction, [1, 2, 0])
    
