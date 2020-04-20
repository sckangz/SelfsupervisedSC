import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1=1.0, re_constant2=1.0, re_constant3=1.0,re_constant4=1.0,
                 batch_size=200, reg=None,ds = None, \
                 denoise=False, model_path=None, restore_path=None, \
                 logs_path='./logs',rawImg=None):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        usereg = 2
        # input required to be fed
        self.x = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])
        t_bs = tf.shape(self.x)[0]
        weights = self._initialize_weights()

        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)

        hid_dim =latent.shape[1].value * latent.shape[2].value * latent.shape[3].value
        z = tf.reshape(latent, [t_bs, hid_dim])
        # classifier  module
        if ds is not None:
            #pslb0 = tf.layers.dense(z, 4*ds, kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.sigmoid,name='ss_d0')

            pslb = tf.layers.dense(z,ds,kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.softmax,name = 'ss_d')
            cluster_assignment = tf.argmax(pslb, -1)
            eq = tf.to_float(tf.equal(cluster_assignment,tf.transpose(cluster_assignment)))

        ze = z
        Coef = weights['Coef']
        z_ce = tf.matmul(Coef, ze)
        # if ds is not None:
        #     z_c = tf.layers.dense(z_ce, hid_dim,activation=tf.nn.relu, name='demb')

        z_c = z_ce
        self.Coef = Coef

        latent_c = tf.reshape(z_c, tf.shape(latent))
        self.z = ze

        self.x_r = self.decoder(latent_c, weights, shape)
        self.x_r2 = self.decoder(latent, weights, shape)
        # l_2 reconstruction loss
        self.reconst_cost = tf.reduce_sum(tf.square(tf.subtract(self.x_r, self.x)))
        self.reconst_cost_pre = tf.reduce_sum(tf.square(tf.subtract(self.x_r2, self.x)))
        tf.summary.scalar("recons_loss", self.reconst_cost)

        if usereg == 2:
            self.reg_losses = tf.reduce_sum(tf.square(self.Coef))+tf.trace(tf.square(self.Coef))
        else:
            self.reg_losses = tf.reduce_sum(tf.abs(self.Coef))+tf.trace(tf.abs(self.Coef))

        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.square(tf.subtract(z_ce, ze)))

        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses)

        x_flattten = tf.reshape(x_input, [t_bs, -1])
        x_flattten2 = tf.reshape(self.x_r, [t_bs, -1])
        XZ = tf.matmul(Coef, x_flattten)
        self.selfexpress_losses2 = 0.5 * tf.reduce_sum(tf.square(tf.subtract(XZ, x_flattten)))

        normL = True
        #graph(C)
        absC = tf.abs(Coef)
        C = (absC + tf.transpose(
            absC)) * 0.5  # * (tf.ones([Coef.shape[0].value,Coef.shape[0].value])-tf.eye(Coef.shape[0].value))
        C = C + tf.eye(Coef.shape[0].value)

        self.cc=C- tf.eye(Coef.shape[0].value)
        # DD = tf.diag(tf.sqrt(1.0/tf.reduce_sum(C, axis=1)))
        # C = tf.matmul(DD,C)
        # C = tf.matmul(C,DD)
        # D = tf.eye(Coef.shape[0].value)
        # L = D - C
        if normL == True:
            D = tf.diag(tf.sqrt((1.0 / tf.reduce_sum(C, axis=1))))
            I = tf.eye(D.shape[0].value)
            L = I - tf.matmul(tf.matmul(D, C), D)
            D = I
        else:
            D = tf.diag(tf.reduce_sum(C, axis=1))
            L = D - C
        # self.reg_losses += 1.0*tf.reduce_sum(tf.square(tf.reduce_sum(Coef,axis=1)-tf.ones_like(tf.reduce_sum(Coef,axis=1))))
        # XLX = tf.matmul(tf.matmul(tf.transpose(x_flattten), L), x_flattten)
        XLX2 = tf.matmul(tf.matmul(tf.transpose(x_flattten), L), x_flattten2)
        # YLY = tf.matmul(tf.matmul(tf.transpose(z), L), z)
        XX = x_flattten - x_flattten2
        # XXDXX = tf.matmul(tf.matmul(tf.transpose(XX),D),XX)
        self.tracelossx = tf.reduce_sum(tf.square(XX)) + 2.0 * tf.trace(XLX2)  # /self.batch_size
        # self.d = tf.reduce_sum(C, axis=1)

        self.d = cluster_assignment
        self.l = tf.trace(XLX2)
        regass = tf.to_float(tf.reduce_sum(pslb,axis=0))

        onesl=np.ones(batch_size)
        zerosl=np.zeros(batch_size)
        #thershold
        weight_label=tf.where(tf.reduce_max(pslb,axis=1)>0.8,onesl,zerosl)
        cluster_assignment1=tf.one_hot(cluster_assignment,ds)
        self.w_weight=weight_label
        self.labelloss=tf.losses.softmax_cross_entropy(onehot_labels=cluster_assignment1,logits=pslb,weights=weight_label)


        self.graphloss = tf.reduce_sum(tf.nn.relu((1-eq) * C)+tf.nn.relu(eq * (0.001-C)))+ tf.reduce_sum(tf.square(regass))




        #self.loss = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses
        #self.loss2 = (self.reconst_cost+self.tracelossx + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses  + re_constant4 * self.selfexpress_losses2)
        self.loss3 = ( self.reconst_cost+self.tracelossx +re_constant2 * self.selfexpress_losses  + re_constant3 * self.labelloss+re_constant4 * self.graphloss)
            # self.reconst_cost + reg_constant1 * self.reg_losses + re_constant3 * self.selfexpress_losses2 + re_constant2 * self.selfexpress_losses
        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss3)  # GradientDescentOptimizer #AdamOptimizer
        self.optimizer3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss3)
        self.optimizer = self.optimizer2#tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)  # GradientDescentOptimizer #AdamOptimizer
        self.optimizer_pre = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.reconst_cost_pre)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef") or v.name.startswith("ss"))])
        # [v for v in tf.trainable_variables() if not (v.name.startswith("Coef")or v.name.startswith("ss"))]
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden)
        all_weights['Coef'] = tf.Variable(
            # 1 * tf.eye(self.batch_size, dtype=tf.float32), name='Coef')
            1.0e-5 * (tf.ones([self.batch_size, self.batch_size], dtype=tf.float32)), name='Coef')

        all_weights['enc_w0'] = tf.get_variable("enc_w0",
                                                shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))  # , name = 'enc_b0'

        iter_i = 1
        while iter_i < n_layers:
            enc_name_wi = 'enc_w' + str(iter_i)
            all_weights[enc_name_wi] = tf.get_variable(enc_name_wi,
                                                       shape=[self.kernel_size[iter_i], self.kernel_size[iter_i],
                                                              self.n_hidden[iter_i - 1], \
                                                              self.n_hidden[iter_i]],
                                                       initializer=layers.xavier_initializer_conv2d(),
                                                       regularizer=self.reg)
            enc_name_bi = 'enc_b' + str(iter_i)
            all_weights[enc_name_bi] = tf.Variable(
                tf.zeros([self.n_hidden[iter_i]], dtype=tf.float32))  # , name = enc_name_bi
            iter_i = iter_i + 1

        iter_i = 1
        while iter_i < n_layers:
            dec_name_wi = 'dec_w' + str(iter_i - 1)
            all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers - iter_i],
                                                                           self.kernel_size[n_layers - iter_i],
                                                                           self.n_hidden[n_layers - iter_i - 1],
                                                                           self.n_hidden[n_layers - iter_i]],
                                                       initializer=layers.xavier_initializer_conv2d(),
                                                       regularizer=self.reg)
            dec_name_bi = 'dec_b' + str(iter_i - 1)
            all_weights[dec_name_bi] = tf.Variable(
                tf.zeros([self.n_hidden[n_layers - iter_i - 1]], dtype=tf.float32))  # , name = dec_name_bi
            iter_i = iter_i + 1

        dec_name_wi = 'dec_w' + str(iter_i - 1)
        all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                                       self.n_hidden[0]],
                                                   initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        dec_name_bi = 'dec_b' + str(iter_i - 1)
        all_weights[dec_name_bi] = tf.Variable(tf.zeros([1], dtype=tf.float32))  # , name = dec_name_bi

        return all_weights

    # Building the encoder
    def encoder(self, x, weights):
        shapes = []
        shapes.append(x.get_shape().as_list())
        layeri = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1, 2, 2, 1], padding='SAME'),
                                weights['enc_b0'])
        layeri = tf.nn.relu(layeri)
        shapes.append(layeri.get_shape().as_list())

        n_layers = len(self.n_hidden)
        iter_i = 1
        while iter_i < n_layers:
            layeri = tf.nn.bias_add(
                tf.nn.conv2d(layeri, weights['enc_w' + str(iter_i)], strides=[1, 2, 2, 1], padding='SAME'),
                weights['enc_b' + str(iter_i)])
            layeri = tf.nn.relu(layeri)
            shapes.append(layeri.get_shape().as_list())
            iter_i = iter_i + 1

        layer3 = layeri
        return layer3, shapes

    # Building the decoder
    def decoder(self, z, weights, shapes):
        n_layers = len(self.n_hidden)
        layer3 = z
        iter_i = 0
        while iter_i < n_layers:
            # if iter_i == n_layers-1:
            #    strides_i = [1,2,2,1]
            # else:
            #    strides_i = [1,1,1,1]
            shape_de = shapes[n_layers - iter_i - 1]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack(
                [tf.shape(self.x)[0], shape_de[1], shape_de[2], shape_de[3]]), strides=[1, 2, 2, 1], padding='SAME'),
                            weights['dec_b' + str(iter_i)])
            layer3 = tf.nn.relu(layer3)
            iter_i = iter_i + 1
        return layer3

    def partial_fit(self, X, lr, mode=0):  #
        cost0, cost1, cost2,cost3, summary, _, Coef,d,dt,l = self.sess.run((self.reconst_cost, self.selfexpress_losses,
                                                                   self.selfexpress_losses2,self.tracelossx, self.merged_summary_op,
                                                                   self.optimizer, self.Coef,self.w_weight,self.d,self.l),
                                                                  feed_dict={self.x: X, self.learning_rate: lr})  #
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return [cost0, cost1, cost2,cost3], Coef, d,dt



    def partial_pre(self, X, lr, mode=0):  #
        cost0, _, = self.sess.run((self.reconst_cost_pre,self.optimizer_pre),
                                                                  feed_dict={self.x: X, self.learning_rate: lr})  #
        self.iter = self.iter + 1
        return [cost0]

    def initlization(self):
        tf.reset_default_graph()
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print("model restored")
