import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization

import time

TF_DTYPE = tf.float32
class FeedForwardModel(object):

    def __init__(self, sess, bsde, config):
        self._sess = sess
        self.config = config
        self.bsde = bsde
        
        self.dim = bsde._dim
        self.total_time = bsde._total_time
        self.num_time_interval = bsde._num_time_interval
        self.delta_t = bsde._delta_t
        
        # Store all z_network in a list
        self.z_network=[]
        

    # Function that calculate the output of a z_network at certain time step, given the input
    def calculate_zt(self, _network, _input):
        z = _network[0](_input)
        for i in range(1, len(_network)):
            z = _network[i](z)
        return z
    
    # Function that builds up the z_network at each time step
    def relu_adding(self, network, unit_list, activation = None):
        #num >=2
        num = len(unit_list)
        network.append(Dense(units = unit_list[0], input_shape = (self.dim,), activation = 'relu'))
        for i in range(1, num):
            network.append(Dense(units = unit_list[i], input_shape = (unit_list[i-1],), activation = 'relu'))
        network.append(Dense(units = self.dim, input_shape = (unit_list[-1],), activation = activation))
    
    # Build the fully connected neural network
    def build(self):
        start_time = time.time()
        time_stamp = np.arange(0, self.num_time_interval) * self.delta_t
        
        # add N-1 z_network
        for i in range(self.num_time_interval-1):
            temp = []
            temp.append(BatchNormalization())
            self.relu_adding(temp, self.config.z_units)
            self.z_network.append(temp) 
        
        self._x = tf.placeholder(TF_DTYPE, [None,self.dim, self.num_time_interval+1], name='X')
        self._dw = tf.placeholder(TF_DTYPE, [None, self.dim, self.num_time_interval], name='dW')
        self._is_training = tf.placeholder(tf.bool)
        
        # Price at t=0, y_init
        self._y_init = tf.Variable(tf.random_uniform([1],
                                                     minval=self.config.y_init_range[0],
                                                     maxval=self.config.y_init_range[1],
                                                     dtype=TF_DTYPE))

        # The first delta at t=0
        self._z_init = tf.Variable(tf.random_uniform([1, self.dim],
                                                    minval=-.1, maxval=.1,
                                                    dtype=TF_DTYPE))
        
        # Connect the network 
        with tf.variable_scope('forward'):
            all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE)
            y = all_one_vec * self._y_init
            z = tf.matmul(all_one_vec, self._z_init)
            
            # Going forward through BSDE:
            for t in range(0, self.num_time_interval - 1):
                y = y - self.delta_t * (
                    self.bsde.f_tf(time_stamp[t], self._x[:, :, t], y, z))
                
                y = y + tf.reduce_sum(z * self._dw[:, :, t], 1, keepdims=True)

                z = self.calculate_zt(self.z_network[t], self._x[:, :, t + 1])

            y = y - self.bsde.delta_t * self.bsde.f_tf(
                time_stamp[-1], self._x[:, :, -2], y, z)
            y = y + tf.reduce_sum(z * self._dw[:, :, -1], 1, keepdims=True)
            
            # Mean square loss:
            loss = y - self.bsde.g_tf(self.total_time, self._x[:, :, -1])
            self._loss = tf.reduce_mean(tf.square(loss))
        
        # Stochastic gradient descent using Adam, all settings are as default
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._loss, trainable_variables)
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self.config.lr_boundaries,
                                                    self.config.lr_values)
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step,name='train_step')
        all_ops = [apply_op]
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time()-start_time
        
    def train(self):
        self.start_time = time.time()
        
        # Generate validation set sample paths
        dw_valid,x_valid = self.bsde.sample(self.config.valid_size)
        
        # Validation set
        feed_dict_valid = {self._x: x_valid, self._dw:dw_valid,
                           self._is_training: False}
        
        # Initialize all parameters
        self._sess.run(tf.global_variables_initializer())
        
        # Begin training 
        for step in range(self.config.num_iterations + 1):
            
            # Print training result according to the logging frequency
            if step % self.config.logging_frequency == 0:
                
                # Calculate mean square loss, and get current y_ini
                loss, init= self._sess.run([self._loss, self._y_init], 
                                                   feed_dict=feed_dict_valid)
                
                # Calculate running time
                elapsed_time = time.time() - self.start_time + self._t_build
                print("step: %5u, loss: %.4e, y_init: %.4e, elapsed time %3u" % (step, loss, init,elapsed_time))
            
            # Generate training set sample paths
            dw_train,x_train = self.bsde.sample(self.config.batch_size)
            
            # Train the model using the training set
            loss=self._sess.run([self._loss ,self._train_ops], 
                                feed_dict={self._x: x_train,self._dw:dw_train,self._is_training: True})[0]