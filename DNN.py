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

def dense(inputs, in_size, out_size, activation='sigmoid', name='layer'):

    with tf.variable_scope(name, reuse=False):

        w = tf.get_variable("w", shape=[in_size, out_size], initializer=tf.random_normal_initializer(mean=0., stddev=0.1))
        b = tf.get_variable("b", shape=[out_size], initializer=tf.constant_initializer(0.0))

        l = tf.add(tf.matmul(inputs, w), b)

        if activation == 'relu':
            l = tf.nn.relu(l)
        elif activation == 'sigmoid':
            l = tf.nn.sigmoid(l)
        elif activation == 'tanh':
            l = tf.nn.tanh(l)
        else:
            l = l

        l = tf.nn.dropout(l, keep_prob)

    return l

def scope(sess, hyperparameters):

    # Learning rate
    learning_rate = tf.Variable(hyperparameters['learning_rate'], trainable=False)

    # Loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_), name="loss")

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(loss)

    # Evaluate the model
    correct = tf.equal(tf.cast(tf.argmax(y_, 1), tf.int32), tf.cast(tf.argmax(y, 1), tf.int32), name='correct')
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    # Tensorboard summary
    writer = tf.summary.FileWriter('./Tensorboard/')  # run this command in the terminal to launch tensorboard: tensorboard --logdir=./Tensorboard/
    writer.add_graph(graph=sess.graph)

    return optimizer, loss, accuracy

def train(x, y, keep_prob, train_x, one_hots_train):

    loss_history, acc_history = [], []

    for epoch in range(hyperparameters['maxEpoch']):

        i = 0
        loss_batch, acc_batch = [], []

        while i < n_sample_train:

            start = i
            end = i + hyperparameters['batch_size']

            train_data = {x: train_x[start:end], y: one_hots_train[start:end], keep_prob: 0.8}

            _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict=train_data)

            loss_batch.append(l)
            acc_batch.append(acc)

            i += hyperparameters['batch_size']

        epoch_loss = np.mean(loss_batch)
        epoch_acc = np.mean(acc_batch)

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        print('Epoch', epoch, '/', hyperparameters['maxEpoch'], '. : Loss:', epoch_loss, ' - Accuracy', epoch_acc)

    return loss_history, acc_history

def test(x, y, keep_prob, test_x, one_hots_test):

    # Test the Neural Network
    test_data = {x: test_x, y: one_hots_test, keep_prob: 1}
    acc = sess.run(accuracy, feed_dict=test_data)
    predictions = y_.eval(feed_dict=test_data, session=sess)
    one_hots_predictions = (predictions == predictions.max(axis=1, keepdims=True)).astype(int)
    number_predicted = [one_hots_predictions[i, :].argmax() for i in range(0, one_hots_predictions.shape[0])]

    print('Test accuracy ' + str(acc))

    return acc, number_predicted

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
height = train_x.shape[1]               # All the pixels are represented as a vector (dim: 784)

# Hyperparameters
hyperparameters = {'learning_rate': 0.1, 'maxEpoch': 50, 'batch_size': 200,
                   'dense_size': [height, 10, 10, n_label], 'dense_activation': ['sigmoid', 'relu', 'none'], 'names': ['layer_1', 'layer_2', 'layer_out']}

# Session and context manager
tf.reset_default_graph()
sess = tf.Session()

with tf.variable_scope(tf.get_variable_scope()):

    # Placeholders
    x = tf.placeholder(tf.float32, [None, height], name='X')
    y = tf.placeholder(tf.float32, [None, n_label], name='Y')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Neural network
    print(x)
    l1 = dense(x, in_size=hyperparameters['dense_size'][0], out_size=hyperparameters['dense_size'][1], activation=hyperparameters['dense_activation'][0], name=hyperparameters['names'][0])
    print(l1)
    l2 = dense(l1, in_size=hyperparameters['dense_size'][1], out_size=hyperparameters['dense_size'][2], activation=hyperparameters['dense_activation'][1], name=hyperparameters['names'][1])
    print(l2)
    l3 = dense(l2, in_size=hyperparameters['dense_size'][2], out_size=hyperparameters['dense_size'][3], activation=hyperparameters['dense_activation'][2], name=hyperparameters['names'][2])
    print(l3)

    # Softmax layer
    y_ = tf.nn.softmax(l3, name='softmax')

    # Scope
    optimizer, loss, accuracy = scope(sess, hyperparameters)

    # Initialize the Neural Network
    sess.run(tf.global_variables_initializer())

    # Train the Neural Network
    loss_history, acc_history = train(x, y, keep_prob, train_x, one_hots_train)

# Test the trained Neural Network
accuracy, number_predicted = test(x, y, keep_prob, test_x, one_hots_test)

# Confusion matrix
cm = metrics.confusion_matrix(number_test, number_predicted)
print(cm)
cmxN = cm / cm.astype(np.float).sum(axis=0)
print(cmxN)

confusion_matrix(cm=cmxN, accuracy=accuracy)