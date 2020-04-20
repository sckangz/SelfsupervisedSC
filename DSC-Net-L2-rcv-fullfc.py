import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from tensorflow.examples.tutorials.mnist import input_data

from AEutils import  *
import  traceback
# SELECT GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sc = 1250
import tensorflow as tf
from tensorflow.contrib import layers

class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1=1.0, re_constant2=1.0, re_constant3=1.0,re_constant4=1.0,
                 batch_size=200, reg=None,ds=None, \
                 denoise=False, model_path=None, restore_path=None, \
                 logs_path='./logs'):
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
        self.x = tf.placeholder(tf.float32, [None, n_input[0]*n_input[1]])
        self.learning_rate = tf.placeholder(tf.float32, [])
        c_dim = batch_size * batch_size
        weights = self._initialize_weights()

        # if denoise == False:
        #     x_input = self.x
        #     latent, shape = self.encoder(x_input, weights)
        # else:
        #     x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
        #                                               mean=0,
        #                                               stddev=0.2,
        #                                               dtype=tf.float32))
        #     latent, shape = self.encoder(x_input, weights)
        x_input = self.x
        latent = tf.layers.dense(x_input,500,activation=tf.nn.relu)
        latent = tf.layers.dense(latent, 500,activation=tf.nn.relu)
        latent = tf.layers.dense(latent, 2000,activation=tf.nn.relu)
        latent = tf.layers.dense(latent, 4, activation=None)
        z = latent

        # query = tf.layers.dense(x_input,500,activation=tf.nn.relu)
        # query = tf.layers.dense(query, 1000,activation=tf.nn.relu)
        # query = tf.layers.dense(query, 750,activation=None)
        # query = tf.reshape(query, [batch_size, 750])
        #
        # key = tf.layers.dense(x_input, 500, activation=tf.nn.relu)
        # key = tf.layers.dense(key, 1000, activation=tf.nn.relu)
        # key = tf.layers.dense(key, 750, activation=None)
        # key = tf.reshape(key, [batch_size, 750])

        # Coef = tf.matmul(query,tf.transpose(key))
        # Coef = tf.nn.softmax(Coef,axis=1)
        # Coef = Coef / tf.reduce_sum(Coef,axis=1,keepdims=True)
        Coef = weights['Coef']
        z_c = tf.matmul(Coef, z)
        self.Coef = Coef

        self.z = z



        if ds is not None:
            #pslb0 = tf.layers.dense(z, 4*ds, kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.sigmoid,name='ss_d0')

            pslb = tf.layers.dense(z,ds,kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.softmax,name = 'ss_d')
            cluster_assignment = tf.argmax(pslb, -1)
            eq = tf.to_float(tf.equal(cluster_assignment,tf.transpose(cluster_assignment)))




        # self.x_r = self.decoder(latent_c, weights, shape)
        # self.x_r2 = self.decoder(latent, weights, shape)
        self.x_r = tf.layers.dense(z_c, 2000,activation=tf.nn.relu,name='d1')
        self.x_r = tf.layers.dense(self.x_r, 500,activation=tf.nn.relu,name='d2')
        self.x_r = tf.layers.dense(self.x_r, 500,activation=tf.nn.relu,name='d3')
        self.x_r = tf.layers.dense(self.x_r, 2000, activation=None, name='d4')

        self.x_r2 = tf.layers.dense(z, 2000, activation=tf.nn.relu,name='d1',reuse=True)
        self.x_r2 = tf.layers.dense(self.x_r2, 500, activation=tf.nn.relu,name='d2',reuse=True)
        self.x_r2 = tf.layers.dense(self.x_r2, 500, activation=tf.nn.relu,name='d3',reuse=True)
        self.x_r2 = tf.layers.dense(self.x_r2, 2000, activation=None, name='d4', reuse=True)
        # self.x_r = self.x_r2
        # l_2 reconstruction loss
        self.reconst_cost = tf.reduce_sum(tf.square(tf.subtract(self.x_r, self.x)))
        self.reconst_cost_pre = tf.reduce_sum(tf.square(tf.subtract(self.x_r2, self.x)))
        tf.summary.scalar("recons_loss", self.reconst_cost)

        if usereg == 2:
            self.reg_losses = tf.reduce_sum(tf.square(self.Coef))+tf.trace(tf.square(self.Coef))
        else:
            self.reg_losses = tf.reduce_sum(tf.abs(self.Coef))+tf.trace(tf.abs(self.Coef))

        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.square(tf.subtract(z_c, z)))

        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses)

        x_flattten = tf.reshape(x_input, [batch_size, -1])
        x_flattten2 = tf.reshape(self.x_r, [batch_size, -1])
        XZ = tf.matmul(Coef, x_flattten)
        self.selfexpress_losses2 = 0.5 * tf.reduce_sum(tf.square(tf.subtract(XZ, x_flattten)))

        normL = False
        absC = tf.abs(Coef)
        C = (absC + tf.transpose(
            absC)) * 0.5  # * (tf.ones([Coef.shape[0].value,Coef.shape[0].value])-tf.eye(Coef.shape[0].value))
        C = C + tf.eye(Coef.shape[0].value)
        DD = tf.diag(tf.sqrt(1.0 / tf.reduce_sum(C, axis=1)))
        C = tf.matmul(DD, C)
        C = tf.matmul(C, DD)
        # C = C + tf.eye(Coef.shape[0].value)
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
        XXDXX = tf.matmul(tf.matmul(tf.transpose(XX), D), XX)
        self.tracelossx = tf.trace(XXDXX + 2.0 * XLX2)
        self.d = tf.reduce_sum(C, axis=1)
        self.l = tf.trace(XLX2)




        self.lala=cluster_assignment



        regass = tf.to_float(tf.reduce_sum(pslb,axis=0))

        onesl=np.ones(batch_size)
        zerosl=np.zeros(batch_size)
        weight_label=tf.where(tf.reduce_max(pslb,axis=1)>0.8,onesl,zerosl)
        cluster_assignment1=tf.one_hot(cluster_assignment,ds)
        #label1=tf.one_hot(self.label,ds)
        self.w_weight=weight_label
        self.labelloss=tf.losses.softmax_cross_entropy(onehot_labels=cluster_assignment1,logits=pslb,weights=weight_label)

        self.graphloss = tf.reduce_sum(tf.nn.relu((1 - eq) * C) + tf.nn.relu(eq * (0.001 - C))) + tf.reduce_sum(tf.square(regass))



        self.loss = self.reconst_cost + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses
        self.loss2 = (self.reconst_cost + self.tracelossx + + re_constant2 * self.selfexpress_losses + + re_constant3 * self.labelloss + re_constant4 * self.graphloss)
        # self.reconst_cost + reg_constant1 * self.reg_losses + re_constant3 * self.selfexpress_losses2 + re_constant2 * self.selfexpress_losses
        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss2)  # GradientDescentOptimizer #AdamOptimizer
        self.optimizer = self.optimizer2#tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)  # GradientDescentOptimizer #AdamOptimizer
        self.optimizer_pre = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.reconst_cost_pre)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])
        # [v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]
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
        cost0, cost1, cost2,cost3, summary, _, Coef,d,l = self.sess.run((self.reconst_cost, self.selfexpress_losses,
                                                                   self.selfexpress_losses2,self.tracelossx, self.merged_summary_op,
                                                                   self.optimizer, self.Coef,self.d,self.l),
                                                                  feed_dict={self.x: X, self.learning_rate: lr})  #
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return [cost0, cost1, cost2,cost3], Coef

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
#for the first time training the model, set the pred to False
pred = False
def test_face(Img, Label, CAE, num_class,lr2=5e-4):

    d = 4
    alpha = 5
    ro = 0.12
    # d = 5
    # alpha = 6
    # ro = 0.14
    acc_= []
    for i in range(0,5-num_class):
        face_10_subjs = np.array(Img[sc*i:sc*(i+num_class),:])
        face_10_subjs = face_10_subjs.astype(float)        
        label_10_subjs = np.array(Label[sc*i:sc*(i+num_class)])
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs) 
                     
        CAE.initlization()        
        # CAE.restore() # restore from pre-trained model
        global pred
        if pred==True:
            CAE.restore()
        else:
            pre_step = 2000
            epoch = 0
            pbatch_size = 256
            while epoch < pre_step:
                indices = np.arange(0, rawImg.shape[0])
                np.random.shuffle(indices)
                indices = indices[:pbatch_size]
                face_10_subjs_pre = np.array(rawImg[indices, :])
                cost = CAE.partial_pre(face_10_subjs_pre, 1e-3)
                epoch = epoch + 1
                if epoch % 300 == 0:
                    print("pre epoch: %.1d" % epoch, "cost: %.8f" % (cost[0]/float(batch_size)))
            CAE.save_model()
            pred = True



        max_step = 15#50 + num_class*25# 100+num_class*20
        display_step = 2000#max_step/20#10
        # fine-tune network
        epoch = 0
        COLD = None
        lastr = 1.0
        # visualize(rawImg, rawLabel, CAE, 'rcv-tsne-ae.png')
        while epoch < max_step:
            epoch = epoch + 1
            cost, Coef = CAE.partial_fit(face_10_subjs, lr2, mode = 'fine')  #
            if epoch % display_step == 0:
                print("epoch: %.1d" % epoch, "cost: %.8f" % (cost[0]/float(batch_size))   )
                print(cost)
                for posti in range(1):
                    display(Coef, label_10_subjs, d, alpha, ro)


            if COLD is not None:
                normc = np.linalg.norm(COLD, ord='fro')
                normcd = np.linalg.norm(Coef - COLD, ord='fro')
                r = normcd/normc
                #print(epoch,r)
                if r < 1.0e-6 and lastr < 1.0e-6:
                    print("early stop")
                    print("epoch: %.1d" % epoch, "cost: %.8f" % (cost[0] / float(batch_size)))
                    print(cost)
                    for posti in range(1):
                        display(Coef, label_10_subjs, d, alpha, ro)
                    break
                lastr = r
            COLD = Coef

        for posti in range(1):
            # drawC(Coef)
            acc_x,_,y_pre = display(Coef, label_10_subjs, d, alpha, ro)
            # acc_.append(acc_x)
            acckm = KMtest(face_10_subjs,label_10_subjs,CAE)
            accnn,acckm2 = NNtest(face_10_subjs,y_pre,rawImg,rawLabel,CAE,1,True)
            acc_.append(accnn)
            # visualize(rawImg, rawLabel, CAE, 'rcv-tsne-sae.png')
        # for sd in [5,6,7]:
        #     for sa in [7,8,9]:
        #         for sr in [0.12,0.14,0.16]:
        #             print(sd, sa, sr)
        #             display(Coef, label_10_subjs, sd, sa, sr)
    acc_ = np.array(acc_)
    mm = np.max(acc_)

    print("%d subjects:" % num_class)    
    print("Max: %.4f%%" % ((1-mm)*100))
    print(acc_) 
    
    return (1-mm)

def read_list(file_name, type='int'):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    if type == 'str':
        array = np.asarray([l.strip() for l in lines])
        return array
    elif type == 'int':
        array = np.asarray([int(l.strip()) for l in lines])
        return array
    else:
        print("Unknown type")
        return None

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_rcv1
import scipy.sparse as sp
from sklearn.datasets import fetch_openml
if __name__ == '__main__':

    dataset = fetch_rcv1(subset="all")
    print("Dataset RCV1 loaded...")
    data = dataset.data
    target = dataset.target

    # Get the split between training/test set and validation set
    test_indices = read_list("./deep-k-means-master/split/rcv1/test")
    n_test = test_indices.shape[0]
    validation_indices = read_list("./deep-k-means-master/split/rcv1/validation")
    n_validation = validation_indices.shape[0]

    # Filter the dataset
    ## Keep only the data points in the test and validation sets
    test_data = data[test_indices]
    test_target = target[test_indices]
    validation_data = data[validation_indices]
    validation_target = target[validation_indices]
    data = sp.vstack([test_data, validation_data])
    target = sp.vstack([test_target, validation_target])
    ## Update test_indices and validation_indices to fit the new data indexing
    test_indices = np.asarray(range(0, n_test))  # Test points come first in filtered dataset
    validation_indices = np.asarray(
        range(n_test, n_test + n_validation))  # Validation points come after in filtered dataset

    # Pre-process the dataset
    ## Filter words based on tf-idf
    sum_tfidf = np.asarray(sp.spmatrix.sum(data, axis=0))[
        0]  # Sum of tf-idf for all words based on the filtered dataset
    word_indices = np.argpartition(-sum_tfidf, 2000)[:2000]  # Keep only the 2000 top words in the vocabulary
    data = data[:, word_indices].toarray()  # Switch from sparse matrix to full matrix
    ## Retrieve the unique label (corresponding to one of the specified categories) from target's label vector
    names = dataset.target_names
    category_names = ['CCAT', 'ECAT', 'GCAT', 'MCAT']
    category_indices = [i for i in range(len(names)) if names[i] in category_names]
    dict_category_indices = {j: i for i, j in
                             enumerate(category_indices)}  # To rescale the indices between 0 and some K
    filtered_target = []
    for i in range(target.shape[0]):  # Loop over data points
        target_coo = target[i].tocoo().col
        filtered_target_coo = [t for t in target_coo if t in category_indices]
        assert len(filtered_target_coo) == 1  # Only one relevant label per document because of pre-filtering
        filtered_target.append(dict_category_indices[filtered_target_coo[0]])
    target = np.asarray(filtered_target)
    n_samples = data.shape[0]  # Number of samples in the dataset


    rawLabel = np.reshape(target,(-1,1))
    num = rawLabel.shape[0]
    rawImg = np.reshape(data,[num,-1])

    # dataset = fetch_mldata("MNIST original")
    # print("Dataset MNIST loaded...")
    # data = dataset.data
    # target = dataset.target
    # num = data.shape[0]  # Number of samples in the dataset
    # n_clusters = 10  # Number of clusters to obtain
    # data = data / 255.0
    # rawImg = np.reshape(data,[num,-1])
    # rawLabel = target

    for i in range(4):
        ind = [ii for ii in range(num) if rawLabel[ii] == i]
        ind = ind[0:sc]
        if i == 0:
            Img = rawImg[ind]
            Label = rawLabel[ind]
        else:
            Img = np.concatenate([Img,rawImg[ind]])
            Label =  np.concatenate([Label,rawLabel[ind]])
    Label = np.reshape(Label,(-1,1))
    rawLabel = np.reshape(rawLabel,(-1,1))
    # model_path = './models/model-mnist.ckpt'
    # restore_path = './models/model-mnist.ckpt'
    # logs_path = './logs'
    model_path = './models/model-rcv-tempfullfc.ckpt'
    restore_path = './models/model-rcv-tempfullfc.ckpt'
    logs_path = './logs'

    # face image clustering
    n_input = [2000,1]
    n_hidden = [20,10,5]
    kernel_size = [5,5,5]
    
    Img = np.reshape(Img,[Img.shape[0],-1])
    rawImg = np.reshape(rawImg, [rawImg.shape[0],-1])
    all_subjects = [4]
    reg1 = 1.0e-4
    reg02 = 2
    reg03 = 1e-1

    mm = 0
    mreg = [0,0,0,0]

    startfrom = [0, 0, 0]
    mm = 0

    for reg2 in [1e-3,1e-2,1e-1,1,10,100,1e3]:
        for reg3 in [1e-3,1e-2,1e-1,1,10,100,1e3]:#[0.01,0.1,1,5,10]:
            for reg4 in [1e-3,1e-2,1e-1,1,10,100,1e3]:
    # for reg2 in [10,20,50]:
    #     for reg3 in [10,20,50]:
    #         for reg4 in [0.01,0.1,0.5,1,5,10,20,50]:
                for lr2 in [1e-4]:
                    try:
                        print("reg:", reg2, reg3,reg4, lr2)
                        avg = []
                        med = []
                        iter_loop = 0
                        while iter_loop < len(all_subjects):
                            num_class = all_subjects[iter_loop]
                            batch_size = num_class * sc

                            tf.reset_default_graph()
                            CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, re_constant3=reg3,re_constant4=reg4,
                                         kernel_size=kernel_size, batch_size=batch_size, ds=num_class,model_path=model_path, restore_path=restore_path, logs_path=logs_path)

                            avg_i = test_face(Img, Label, CAE, num_class,lr2)
                            avg.append(avg_i)
                            iter_loop = iter_loop + 1
                            #visualize(Img, Label, CAE)
                        iter_loop = 0

                        if 1-avg[0] > mm:
                            mreg= [reg2,reg3,reg4,lr2]
                            mm = 1-avg[0]
                        print("max:", mreg, mm)
                    except:
                        print("error in ", reg2, reg3, lr2)
                        traceback.print_exc()
                    finally:
                        try:
                            CAE.sess.close()
                        except:
                            ''

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
