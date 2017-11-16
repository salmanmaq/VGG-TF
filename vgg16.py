from __future__ import division
import tensorflow as tf
import numpy as np
#import cv2
#import matplotlib.pyplot as plt
import os
import cPickle
import functools

LOGDIR = './graphs'

class CIFAR10Loader:
    '''Loads the CIFAR-10 Dataset from disk'''

    def __init__(self, data_path):
        '''Loads the CIFAR-10 dataset'''

        # Specify the list of files in the data path
        train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_batch = 'test_batch'

        # Load the training data
        self.tr_data = np.ndarray([50000, 3072])
        self.tr_labels = np.ndarray([50000], dtype=int)

        for i in range(len(train_batches)):
            f = train_batches[i]
            path = data_path + f
            with open(path, 'rb') as fo:
                dict = cPickle.load(fo)
                data = dict['data']
                labels = dict['labels']
                self.tr_data[i*10000:(i+1)*10000,:] = data
                self.tr_labels[i*10000:(i+1)*10000] = labels

        # Load the test data
        path = data_path + test_batch
        with open(path, 'rb') as fo:
            dict = cPickle.load(fo)
            self.te_data = dict['data']
            self.te_labels = dict['labels']

        # Convert labels to one hot encoding
        self.train_labels = np.zeros([50000, 10], dtype=int)
        self.train_labels[np.arange(50000), self.tr_labels] = 1

        self.test_labels = np.zeros([10000, 10], dtype = int)
        self.test_labels[np.arange(10000), self.te_labels] = 1

        # Reshape training and test images from vectors to matrices
        self.train_data = np.reshape(self.tr_data, newshape=[50000, 32, 32, 3])
        self.test_data = np.reshape(self.te_data, newshape=[10000, 32, 32, 3])

    def getTrainData(self):
        '''Fetches the training images and labels'''
        return self.train_data, self.train_labels

    def getTestData(self):
        '''Fetches the test images and labels'''
        return self.test_data, self.test_labels

class VGG16:

    def __init__(self, dataLoader):
        ''' Initialize the model with the hyperparameters'''

        # Initialize the network hyperparameters
        self.image_height = 32
        self.image_width = 32
        self.learning_rate = 0.001
        self.keep_prob = tf.placeholder(tf.float32) # Dropout layers keep probability
        self.batch_size = 256
        self.EPOCHS = 50
        self.from_scratch = True
        self.num_classes = 10

        # Create placeholders for data and labels
        self.X = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, 3]) # Input image
        self.y = tf.placeholder(tf.float32, [None, self.num_classes]) # Label

        self.weights = self._initialize_weights()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self._initialze_network()

        self._output = None
        self._optimizer = None
        self._loss = None
        self._accuracy = None

        # Load the dataset
        self.train_data, self.train_labels = dataLoader.getTrainData()
        self.test_data, self.test_labels = dataLoader.getTestData()

        self.batches_per_epoch = self.train_data.shape[0] / self.batch_size
        if (self.batches_per_epoch % self.batch_size != 0):
            self.batches_per_epoch = int(self.batches_per_epoch) + 1

        self.batch_counter = 0 # Iterator to load data batch by batch

    def _initialize_weights(self):
        ''' Initializes the network weights'''
        weights = dict()
        weights['W_conv1'] = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev= 0.1))
        weights['b_conv1'] = tf.Variable(tf.constant(0.1, shape=[64]))
        weights['W_conv2'] = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev= 0.1))
        weights['b_conv2'] = tf.Variable(tf.constant(0.1, shape=[64]))
        weights['W_conv3'] = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev= 0.1))
        weights['b_conv3'] = tf.Variable(tf.constant(0.1, shape=[128]))
        weights['W_conv4'] = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev= 0.1))
        weights['b_conv4'] = tf.Variable(tf.constant(0.1, shape=[128]))
        weights['W_conv5'] = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev= 0.1))
        weights['b_conv5'] = tf.Variable(tf.constant(0.1, shape=[256]))
        weights['W_conv6']= tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev= 0.1))
        weights['b_conv6']= tf.Variable(tf.constant(0.1, shape=[256]))
        weights['W_conv7']= tf.Variable(tf.truncated_normal([1, 1, 256, 256], stddev= 0.1))
        weights['b_conv7']= tf.Variable(tf.constant(0.1, shape=[256]))
        weights['W_conv8']= tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev= 0.1))
        weights['b_conv8']= tf.Variable(tf.constant(0.1, shape=[512]))
        weights['W_conv9']= tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev= 0.1))
        weights['b_conv9']= tf.Variable(tf.constant(0.1, shape=[512]))
        weights['W_conv10'] = tf.Variable(tf.truncated_normal([1, 1, 512, 512], stddev= 0.1))
        weights['b_conv10'] = tf.Variable(tf.constant(0.1, shape=[512]))
        weights['W_conv11'] = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev= 0.1))
        weights['b_conv11'] = tf.Variable(tf.constant(0.1, shape=[512]))
        weights['W_conv12'] = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev= 0.1))
        weights['b_conv12'] = tf.Variable(tf.constant(0.1, shape=[512]))
        weights['W_conv13'] = tf.Variable(tf.truncated_normal([1, 1, 512, 512], stddev= 0.1))
        weights['b_conv13'] = tf.Variable(tf.constant(0.1, shape=[512]))
        weights['W_fc1'] = tf.Variable(tf.truncated_normal([1*1*512, 512], stddev= 0.1))
        weights['b_fc1'] = tf.Variable(tf.constant(0.1, shape=[512]))
        weights['W_fc2'] = tf.Variable(tf.truncated_normal([512, 512], stddev= 0.1))
        weights['b_fc2'] = tf.Variable(tf.constant(0.1, shape=[512]))
        weights['W_fc3'] = tf.Variable(tf.truncated_normal([512, self.num_classes], stddev= 0.1))
        weights['b_fc3'] = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))
        return weights

    def _initialze_network(self):
        ''' Runs the forward pass'''

        # Layer 1
        convolve1 = tf.nn.conv2d(self.X, self.weights['W_conv1'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv1']
        conv1 = tf.nn.relu(convolve1)
        # Layer 2
        convolve2 = tf.nn.conv2d(conv1, self.weights['W_conv2'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv2']
        h_conv2 = tf.nn.relu(convolve2)
        conv2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        # Layer 3
        convolve3 = tf.nn.conv2d(conv2, self.weights['W_conv3'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv3']
        conv3 = tf.nn.relu(convolve3)
        # Layer 4
        convolve4 = tf.nn.conv2d(conv3, self.weights['W_conv4'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv4']
        h_conv4 = tf.nn.relu(convolve4)
        conv4 = tf.nn.max_pool(h_conv4, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        # Layer 5
        convolve5 = tf.nn.conv2d(conv4, self.weights['W_conv5'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv5']
        conv5 = tf.nn.relu(convolve5)
        # Layer 6
        convolve6 = tf.nn.conv2d(conv5, self.weights['W_conv6'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv6']
        conv6 = tf.nn.relu(convolve6)
        # Layer 7
        convolve7 = tf.nn.conv2d(conv6, self.weights['W_conv7'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv7']
        h_conv7 = tf.nn.relu(convolve7)
        conv7 = tf.nn.max_pool(h_conv7, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        # Layer 8
        convolve8 = tf.nn.conv2d(conv7, self.weights['W_conv8'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv8']
        conv8 = tf.nn.relu(convolve8)
        # Layer 9
        convolve9 = tf.nn.conv2d(conv8, self.weights['W_conv9'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv9']
        conv9 = tf.nn.relu(convolve9)
        # Layer 10
        convolve10 = tf.nn.conv2d(conv9, self.weights['W_conv10'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv10']
        h_conv10 = tf.nn.relu(convolve10)
        conv10 = tf.nn.max_pool(h_conv10, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        # Layer 11
        convolve11 = tf.nn.conv2d(conv10, self.weights['W_conv11'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv11']
        conv11 = tf.nn.relu(convolve11)
        # Layer 12
        convolve12 = tf.nn.conv2d(conv11, self.weights['W_conv12'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv12']
        conv12 = tf.nn.relu(convolve12)
        # Layer 13
        convolve13 = tf.nn.conv2d(conv12, self.weights['W_conv13'], strides=[1,1,1,1], padding='SAME') + self.weights['b_conv13']
        h_conv13 = tf.nn.relu(convolve13)
        conv13 = tf.nn.max_pool(h_conv13, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        # Layer 14 (FC)
        layer13_flat = tf.reshape(conv13, [-1, 1*1*512])
        _fc1 = tf.add(tf.matmul(layer13_flat, self.weights['W_fc1']), self.weights['b_fc1'])
        fc1 = tf.nn.relu(_fc1)
        # Dropout 14
        fc1_drop = tf.nn.dropout(fc1, self.keep_prob)
        # Layer 15 (FC)
        _fc2 = tf.add(tf.matmul(fc1_drop, self.weights['W_fc2']), self.weights['b_fc2'])
        fc2 = tf.nn.relu(_fc2)
        # Dropout 15
        fc2_drop = tf.nn.dropout(fc2, self.keep_prob)
        # Layer 16 (FC)
        _fc3 = tf.add(tf.matmul(fc2_drop, self.weights['W_fc3']), self.weights['b_fc3'])
        self.fc3 = tf.nn.relu(_fc3)

    def train(self):
        ''' Trains the CNN and updates the parameters

        Inputs: Training Images, Training Image labels
        Outputs: None
        '''
        print '\n################# Started Training! #################\n'

        for _ in range(self.EPOCHS):
            # Get the next batch for training
            for bn in range(self.batches_per_epoch):
                X_batch, y_batch = self.getNextBatch()
                opti, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: X_batch, self.y: y_batch, self.keep_prob: 1.0})
                if (self.batch_counter % 10 == 0) or (self.batch_counter == 0):
                    print 'Epoch: %d, Batch: %d of %d, Loss: %f' % (_, self.batch_counter, self.batches_per_epoch, loss)

            print '\n###### Epoch %d of %d completed! ######' % (_, self.EPOCHS)

        print '\n################# Finished Training! #################\n'

    def test(self):
        '''Tests the performance of the test data and returns the loss'''
        loss = self.sess.run(self.loss, feed_dict={self.X: self.test_data, self.y : self.test_labels, self.keep_prob: 1.0})
        accuracy = self.sess.run(self.accuracy, feed_dict={self.y : self.test_labels})
        return loss, accuracy

    def getNextBatch(self):
        '''Gets the next batch of the training data'''
        start = self.batch_counter * self.batch_size
        end = (self.batch_counter + 1) * self.batch_size
        self.batch_counter += 1

        # Condition to check if it's the last batch. Reset counter if it's the case
        if end > self.train_data.shape[0]:
            end = self.train_data.shape[0]
            self.batch_counter = 0 #TODO: This results in the last batch being printed as batch 0 as well

        batch_data = self.train_data[start:end,:,:,:]
        batch_labels = self.train_labels[start:end,:]

        return batch_data, batch_labels

    def shuffleDataset(self):
        '''Shuffles the entire dataset'''
        pass

    def getWeights(self):
        '''Returns the network weights'''
        return self.sess.run(self.weights)

    @property
    def output(self):
        #if not self._output:
        y_pred = tf.nn.softmax(self.fc3)
        self._output = y_pred
        return self._output

    @property
    def loss(self):
        #if not self._loss:
        cross_entroy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.output), reduction_indices=[1]))
        self._loss = cross_entroy
        return self._loss

    @property
    def optimizer(self):
        if not self._optimizer:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            opt = opt.minimize(self.loss)
            self._optimizer = opt
        return self._optimizer

    @property
    def accuracy(self):
        if not self._accuracy:
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self._accuracy = accuracy
        return self._accuracy
