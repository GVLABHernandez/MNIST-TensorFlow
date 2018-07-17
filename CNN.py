from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data
import random

n_sample_train = 10000
n_sample_test = 1000

def get_MNIST_data():

    mnist = input_data.read_data_sets('./Data', one_hot=True)
    train_x, one_hots_train = mnist.train.next_batch(n_sample_train)
    test_x, one_hots_test = mnist.train.next_batch(n_sample_test)

    train_x = train_x.reshape(-1, 28, 28)
    test_x = test_x.reshape(-1, 28, 28)

    train_x = train_x[:, :, :, np.newaxis]
    test_x = test_x[:, :, :, np.newaxis]

    return train_x, one_hots_train, test_x, one_hots_test

def plot_MNIST(x, one_hot):

    row = 4
    column = 4
    p = random.sample(range(1, 100), row * column)

    plt.figure()

    for i in range(row * column):

        image = x[p[i]].reshape(28, 28)
        plt.subplot(row, column, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title('label = {}'.format(np.argmax(one_hot[p[i]]).astype(int)))
        plt.axis('off')

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                    wspace=0.05, hspace=0.3)
    plt.show()

def dense(input, name, in_size, out_size, activation="relu"):

    with tf.variable_scope(name, reuse=False):
        w = tf.get_variable("w", shape=[in_size, out_size],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        b = tf.get_variable("b", shape=[out_size], initializer=tf.constant_initializer(0.0))

        l = tf.add(tf.matmul(input, w), b)

        if activation == "relu":
            l = tf.nn.relu(l)
        elif activation == "sigmoid":
            l = tf.nn.sigmoid(l)
        elif activation == "tanh":
            l = tf.nn.tanh(l)
        else:
            l = l
        print(l)
    return l

def scope(y, y_, learning_rate=0.1):

    #Learning rate
    learning_rate = tf.Variable(learning_rate,  trainable=False)

    # Loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=y_), name="loss")

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       name="optimizer").minimize(loss)

    # Evaluate the model
    correct = tf.equal(tf.cast(tf.argmax(y_, 1), tf.int32),
                       tf.cast(tf.argmax(y, 1), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    #  Tensorboard
    writer = tf.summary.FileWriter('./Tensorboard/')
    # run this command in the terminal to launch tensorboard:
    # tensorboard --logdir=./Tensorboard/
    writer.add_graph(graph=sess.graph)

    return loss, accuracy, optimizer, writer

def confusion_matrix(cm, accuracy):

    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size=15)

train_x, one_hots_train, test_x, one_hots_test = get_MNIST_data()
number_test = [one_hots_test[i, :].argmax() for i in range(0, one_hots_test.shape[0])]

plot_MNIST(x=train_x, one_hot=one_hots_train)

n_label = len(np.unique(number_test))   # Number of class
height = train_x.shape[1]
width = train_x.shape[2]

# Session and context manager
tf.reset_default_graph()
sess = tf.Session()

with tf.variable_scope(tf.get_variable_scope()):

    # Placeholders
    x = tf.placeholder(tf.float32, [None, height, width, 1], name='X')
    y = tf.placeholder(tf.float32, [None, n_label], name='Y')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Neural network
    l1 = tf.layers.conv2d(inputs=x, kernel_size=[5, 5], strides=[1, 1], filters=16, padding='SAME',
                         activation=tf.nn.relu, name="Conv_1")
    print(l1)
    l1 = tf.layers.max_pooling2d(inputs=l1, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    print(l1)

    l2 = tf.layers.conv2d(inputs=l1, kernel_size=[5, 5], strides=[1, 1], filters=32, padding='SAME',
                         activation=tf.nn.relu, name="Conv_2")
    print(l2)
    l2 = tf.layers.max_pooling2d(inputs=l2, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    print(l2)

    # Reshape to a fully connected layers
    size = l2.get_shape().as_list()
    l3 = tf.reshape(l2, [-1, size[1] * size[2] * size[3]], name="reshape_to_dense")
    print(l3)

    l4 = dense(input=l3, name="output_layer", in_size=size[1] * size[2] * size[3], out_size=n_label, activation="None")
    print(l4)

    # Softmax layer
    y_ = tf.nn.softmax(l4, name='softmax')
    print(y_)

    # Scope
    loss, accuracy, optimizer, writer = scope(y, y_, learning_rate=0.01)

    # Initialize the Neural Network
    sess.run(tf.global_variables_initializer())

    # Train the Neural Network
    loss_history = []
    acc_history = []
    epoch = 100
    train_data = {x: train_x, y: one_hots_train}

    for e in range(epoch):

        _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict=train_data)

        loss_history.append(l)
        acc_history.append(acc)

        print("Epoch " + str(e) + " - Loss: " + str(l) + " - " + str(acc))

plt.plot(acc_history)

# Test the trained Neural Network

# Confusion matrix
