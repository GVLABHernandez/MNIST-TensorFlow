from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import os
from mpl_toolkits.mplot3d import Axes3D

n_sample_train = 10000
n_sample_test = 1000
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
        elif activation == 'leaky_relu':
            l = tf.nn.leaky_relu(l)
        else:
            l = l

    return l

def scope(sess, hyperparameters):

    # Learning rate
    learning_rate = tf.Variable(hyperparameters['learning_rate'], trainable=False)

    # Loss function
    recons_loss = tf.reduce_mean(tf.square(x - x_hat))
    #recons_loss = tf.sqrt(tf.reduce_mean(tf.square(x - x_hat)))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(recons_loss)

    # Tensorboard summary
    writer = tf.summary.FileWriter('./Tensorboard/')  # run this command in the terminal to launch tensorboard: tensorboard --logdir=./Tensorboard/
    writer.add_graph(graph=sess.graph)

    return optimizer, recons_loss

def confusion_matrix(cm, accuracy):

    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size=15)

def encoder(z, train_x, sess):

    encoded = sess.run(z, feed_dict={x: train_x, dropout_rate: 0})

    return encoded

def reconstruct(x_hat, train_x, sess):

    reconstructed = sess.run(x_hat, feed_dict={x: train_x, dropout_rate: 0.2})

    return reconstructed

train_x, one_hots_train, test_x, one_hots_test = get_MNIST_data()
number_test = [one_hots_test[i, :].argmax() for i in range(0, one_hots_test.shape[0])]

plot_MNIST(x=train_x, one_hot=one_hots_train)

n_label = len(np.unique(number_test))   # Number of class
height = train_x.shape[1]               # All the pixels are represented as a vector (dim: 784)

# Hyperparameters
z_dimension = 2

hyperparameters_encoder = {'en_size': [height, 20, 20, z_dimension],
                           'en_activation': ['relu', 'relu', 'linear'],
                           'names': ['en_layer_1', 'en_layer_2', 'latent_space']}
hyperparameters_decoder = {'de_size': [z_dimension, 20, 20, height],
                           'de_activation': ['relu', 'relu', 'linear'],
                           'names': ['de_layer_1', 'de_layer_2', 'de_layer_out']}
hyperparameters_scope = {'learning_rate': 0.01, 'maxEpoch': 50, 'batch_size': 500}

# Session and context manager
tf.reset_default_graph()
sess = tf.Session()

with tf.variable_scope(tf.get_variable_scope()):
    # Placeholders
    x = tf.placeholder(tf.float32, [None, height], name='X')
    # Encoder
    print('')
    print("ENCODER")
    print(x)
    l1 = dense(x, in_size=hyperparameters_encoder['en_size'][0],out_size=hyperparameters_encoder['en_size'][1],
               activation=hyperparameters_encoder['en_activation'][0], name=hyperparameters_encoder['names'][0])
    print(l1)
    l2 = dense(l1, in_size=hyperparameters_encoder['en_size'][1],out_size=hyperparameters_encoder['en_size'][2],
               activation=hyperparameters_encoder['en_activation'][1], name=hyperparameters_encoder['names'][1])
    print(l2)
    z = dense(l2, in_size=hyperparameters_encoder['en_size'][2], out_size=hyperparameters_encoder['en_size'][3],
              activation=hyperparameters_encoder['en_activation'][2], name=hyperparameters_encoder['names'][2])
    print(z)

    print('')
    print("DECODER")
    # Decoder
    print(z)
    l4 = dense(z, in_size=hyperparameters_decoder['de_size'][0],out_size=hyperparameters_decoder['de_size'][1],
               activation=hyperparameters_decoder['de_activation'][0], name=hyperparameters_decoder['names'][0])
    print(l4)
    l5 = dense(l4, in_size=hyperparameters_decoder['de_size'][1],out_size=hyperparameters_decoder['de_size'][2],
               activation=hyperparameters_decoder['de_activation'][1], name=hyperparameters_decoder['names'][1])
    print(l5)
    x_hat = dense(l5, in_size=hyperparameters_decoder['de_size'][2], out_size=hyperparameters_decoder['de_size'][3],
              activation=hyperparameters_decoder['de_activation'][2], name=hyperparameters_decoder['names'][2])
    print(x_hat)

    # Scope
    learning_rate = tf.Variable(hyperparameters_scope['learning_rate'],trainable=False)

    # Loss function
    loss = tf.reduce_mean(tf.square(x - x_hat))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss)

    # Tensorboard
    writer = tf.summary.FileWriter('.\Tensorboard')  # tensorboard --logdir=./Tensorboard/
    writer.add_graph(graph=sess.graph)

    # Initialize the Neural Network
    sess.run(tf.global_variables_initializer())

    # Train the Neural Network
    loss_history = []

    for epoch in range(hyperparameters_scope['maxEpoch']):

        i = 0
        loss_batch = []
        while i < n_sample_train:

            start = i
            end = i + hyperparameters_scope['batch_size']

            train_data = {x: train_x[start:end]}

            _, l = sess.run([optimizer, loss], feed_dict=train_data)
            loss_batch.append(l)
            i = i + hyperparameters_scope['batch_size']

        epoch_loss = np.mean(loss_batch)
        loss_history.append(epoch_loss)

        print('Epoch', epoch, '/', hyperparameters_scope['maxEpoch'],
              '. : Loss:', epoch_loss)

    # Encode the training data
    train_data = {x: train_x}

    encoded = sess.run(z, feed_dict=train_data)

    plt.scatter(encoded[:, 0], encoded[:, 1])

    # Reconstruct the data at the output of the decoder
    reconstructed = sess.run(x_hat, feed_dict=train_data)

# Plot the latent space



# Plot reconstruction

# PCA
