import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization

import time

TF_DTYPE = tf.float32
class FeedForwardModel(object):

    def __init__(self,sess,bsde,config):
        self._sess = sess
        self.config = config
        self.bsde = bsde
        
        self.f_network = []
        self.z_network=[]
        
        self.dim = bsde._dim
        self.total_time = bsde._total_time
        self.num_time_interval = bsde._num_time_interval
        self.delta_t = bsde._delta_t

    def train(self):
        start_time = time.time()
        
        # Generate validation set sample paths
        dw_valid,x_valid = self.bsde.sample(self.config.valid_size)
        feed_dict_valid = {self._x: x_valid,self._dw:dw_valid,self.lambda1:1,self.lambda2:0, self._is_training: False}
        
        # During the pre-train, stop if loss<10, set the initial loss to be 1000
        loss=1000
        counter=1
        while loss>10:
            print('the '+str(counter)+' time of pre-train')
            
            # Initialize parameters, train the initial layer to fit the g_t payoff function
            self._sess.run(tf.global_variables_initializer())
            for step in range(self.config.pre_train_num_iteration+1):
                
                # Print validation loss and result during training:
                if step % self.config.logging_frequency == 0:
                    loss= self._sess.run(self._loss, feed_dict=feed_dict_valid)
                    elapsed_time = time.time()-start_time+self._t_build
                    print("step: %5u,loss: %.4e,  elapsed time %3u" % (step, loss, elapsed_time))
                
                # Generate training samples and train
                dw_train,x_train = self.bsde.sample(self.config.batch_size)
                loss=self._sess.run([self._loss ,self._train_ops], feed_dict={self._x: x_train,self._dw:dw_train,self.lambda1:1,
                                               self.lambda2:0,self._is_training: True})[0]
            counter+=1
        
        # When pre-train is done, switch between lambda1 and lambda2
        print('Finish pre train')
        feed_dict_valid = {self._x: x_valid,self._dw:dw_valid,self.lambda1:0,self.lambda2:1, self._is_training: False}
        
        # Start the usual training:
        for step in range(self.config.num_iterations+1):
            
            # Print validation loss and result during training:
            if step % self.config.logging_frequency == 0:
                loss= self._sess.run(self._loss, feed_dict=feed_dict_valid)
                elapsed_time = time.time()-start_time+self._t_build
                print("step: %5u,loss: %.4e,  elapsed time %3u" % (step, loss, elapsed_time))
            
            # Generate training samples and train
            dw_train,x_train = self.bsde.sample(self.config.batch_size)
            loss=self._sess.run([self._loss ,self._train_ops], feed_dict={self._x: x_train,self._dw:dw_train,self.lambda1:0,
                                            self.lambda2:1,self._is_training: True})[0]
        
        # Calculate and return the arrays used to do the plot of nn-network result:
        f_graphs=self._sess.run(self.f_graphs, feed_dict=feed_dict_valid)
        z_graphs=self._sess.run(self.z_graphs, feed_dict=feed_dict_valid)
        return f_graphs,z_graphs
    
    # Function that builds up the nn_network at each time step
    def relu_adding(self, network, unit_list, activation = None):
        #num >=2
        num = len(unit_list)
        network.append(Dense(units = unit_list[0], input_shape = (self.dim,), activation = 'relu'))
        for i in range(1, num):
            network.append(Dense(units = unit_list[i], input_shape = (unit_list[i-1],), activation = 'relu'))
        network.append(Dense(units = self.dim, input_shape = (unit_list[-1],), activation = activation))
    
    # Function that calculate the output of a nn_network at certain time step, given the input
    def calculate_nn(self, _network, _input):
        result = _network[0](_input)
        for i in range(1, len(_network)):
            result = _network[i](result)
        return result
    
    # Build the fully connected neural network
    def build(self):
        start_time = time.time()
        time_stamp = np.arange(0, self.num_time_interval) * self.delta_t
        
        # add N z_network
        for i in range(self.num_time_interval):
            temp = []
            temp.append(BatchNormalization())
            self.relu_adding(temp, self.config.z_units)
            self.z_network.append(temp)
        
        # Build the initial network for price
        temp = []
        temp.append(BatchNormalization())
        self.relu_adding(temp, self.config.f_units,activation = 'relu')
        self.f_network.append(temp)
        
        # lambda1 and lambda2 are the switch for pre-train and train
        self._x = tf.placeholder(TF_DTYPE, [None,self.dim, self.num_time_interval+1], name='X')
        self._dw = tf.placeholder(TF_DTYPE, [None, self.dim, self.num_time_interval], name='dW')
        self.lambda1=tf.placeholder(TF_DTYPE, name='lambda1')
        self.lambda2=tf.placeholder(TF_DTYPE, name='lambda2')
        self._is_training = tf.placeholder(tf.bool)
        
        # this x is used for drawing delta and price curve
        x = np.linspace(self.bsde._x0_range[0],self.bsde._x0_range[1], 82)
        self.x = tf.constant(x, dtype=TF_DTYPE, shape=[82,1])

        with tf.variable_scope('forward'):
            y = self.calculate_nn(self.f_network[0], self._x[:,:,0])
            z = self.calculate_nn(self.z_network[0], self._x[:,:,0])
            
            # Make a copy of initial y
            y_init=y+0.0
            
            # Going forward through BSDE:
            for t in range(self.num_time_interval-1):
                y = y - self.delta_t * (
                    self.bsde.f_tf(time_stamp[t], self._x[:, :, t], y, z))
                
                y = y + tf.reduce_sum(z *self.bsde._sigma*self._x[:, :, t] * self._dw[:, :, t], 1, keepdims=True)

                z = self.calculate_nn(self.z_network[t+1], self._x[:,:,t+1])
                
            y = y - self.delta_t * (
                    self.bsde.f_tf(time_stamp[-1], self._x[:, :, -2], y, z))
            y=y+tf.reduce_sum(z * self.bsde._sigma *self._x[:, :, -2]* self._dw[:, :, -1], 1, keepdims=True)
            
            # Mean square loss:
            # When lambda1 = 1 and lambda2 = 0, the loss is the loss of training the initial to fit payoff function
            # When lambda1 = 0 and lambda2 = 1, the loss is the usual mean square loss
            loss1 = y_init - self.bsde.g_tf(self.total_time, self._x[:, :, 0])
            loss2 = y - self.bsde.g_tf(self.total_time, self._x[:, :, -1])
            self._loss = self.lambda1 * tf.reduce_mean(tf.square(loss1)) + self.lambda2 * tf.reduce_mean(tf.square(loss2))
            
            # Input x the linspace to the neural networks to plot the curve
            self.f_graphs = []
            l = self.x+0.0
            l = self.calculate_nn(self.f_network[0], l)
            self.f_graphs=l
            
            self.z_graphs=[]
            for t in range(self.num_time_interval):
                l = self.x+0.0
                l = self.calculate_nn(self.z_network[t], l)
                self.z_graphs.append(l)
        
        # Stochastic gradient descent using Adam, all settings are as default
        trainable_variables = tf.trainable_variables()
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self.config.lr_boundaries,
                                                    self.config.lr_values)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._train_ops = optimizer.minimize(self._loss,global_step=global_step,var_list=tf.trainable_variables())
        self._t_build = time.time()-start_time