import matplotlib.pyplot as plt
import scipy.io as sio
import os
from tensorflow.examples.tutorials.mnist import input_data

from AEnet_13 import ConvAE
from AEutils import  *
import  traceback
# SELECT GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sc = 100
pred = True
def test_face(Img, Label, CAE, num_class,lr2=5e-4):

    d = 4
    alpha = 5
    ro = 0.12
    # d = 5
    # alpha = 6
    # ro = 0.14
    acc_= []
    for i in range(0,11-num_class):
        face_10_subjs = np.array(Img[sc*i:sc*(i+num_class),:])
        face_10_subjs = face_10_subjs.astype(float)        
        label_10_subjs = np.array(Label[sc*i:sc*(i+num_class)])
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs) 
                     
        CAE.initlization()        
        CAE.restore() # restore from pre-trained model
        # global pred
        # if pred==True:
        #     CAE.restore()
        # else:
        #     pre_step = 1000
        #     epoch = 0
        #     while epoch < pre_step:
        #         epoch = epoch + 1
        #         cost = CAE.partial_pre(face_10_subjs, lr2)
        #         if epoch % 300 == 0:
        #             print("pre epoch: %.1d" % epoch, "cost: %.8f" % (cost[0]/float(batch_size)))
        #     CAE.save_model()
        #     pred = True



        max_step = 300#50 + num_class*25# 100+num_class*20
        display_step = 2000#max_step/20#10
        # fine-tune network
        epoch = 0
        COLD = None
        lastr = 1.0
        # visualize(Img, Label, CAE, 'mnist1-tsne-ae.png')
        while epoch < max_step:
            epoch = epoch + 1
            #if epoch < 2:
            #    cost, Coef, dd,dt = CAE.partial_fit(face_10_subjs, lr2, mode = 'fine')
            #else:
            #    if dd.min()>0:
            cost, Coef, dd,dt= CAE.partial_fit(face_10_subjs, lr2, mode='fine')
            #    else:



            #if epoch==1:
                #cost, Coef, dd, dt = CAE.partial_fit(face_10_subjs, lr2, mode='fine')
            #else:
                #L,y_pre=display1(Coef, label_10_subjs, d, alpha, ro)
                #y_pre=y_pre-1
                #cost, Coef, dd, dt = CAE.partial_fit1(face_10_subjs, lr2,y_pre, mode='fine')

            if epoch % display_step == 0:
                print("epoch: %.1d" % epoch, "cost: %.8f" % (cost[0]/float(batch_size))   )
                print(cost)
                #dd=tf.reduce_sum(dd)
                print(dd)
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
            dd=np.sum(dd)
            print(dd,'hhhhhh')
            acc_x,L,y_pre = display(Coef, label_10_subjs, d, alpha, ro)
            # drawC(L, 'L-MNIST1-L1.png')
            # acc_.append(acc_x)
            # acckm = KMtest(face_10_subjs,label_10_subjs,CAE)
            # acckm = KMtest(face_10_subjs,label_10_subjs,CAE)
            # accnn = NNtest(face_10_subjs,y_pre,rawImg,rawLabel,CAE,5)
            acc_.append(acc_x)
            # visualize(Img, Label, CAE, 'mnist1-tsne-sae.png')
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
    
   
        
    
if __name__ == '__main__':
    
    # load face images and labels
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    Img = []
    Label = []
    num = mnist.train.num_examples
    rawImg = mnist.train._images
    rawLabel = mnist.train._labels
    for i in range(10):
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
    model_path = './models/model-mnist.ckpt'
    restore_path = './models/model-mnist.ckpt'
    logs_path = './logs'

    # face image clustering
    n_input = [28, 28]
    n_hidden = [20,10,5]
    kernel_size = [5,3,3]
    
    Img = np.reshape(Img,[Img.shape[0],n_input[0],n_input[1],1])
    rawImg = np.reshape(rawImg, [rawImg.shape[0], n_input[0], n_input[1], 1])
    all_subjects = [10]
    reg1 = 1.0e-4
    reg02 = 2
    reg03 = 1e-1

    mm = 0
    mreg = [0,0,0,0]

    startfrom = [0, 0, 0]
    mm = 0
    results = []
    # for reg2 in [0.1,1,5,10]:
    #     for reg3 in [1]:#[0.1,1,5,10,50]:
    #         for reg4 in [0.1,1,5,10]:
    #for reg2 in [1e-1,1,10]:
    for reg2 in [1e-3,1e-2,1e-1,1,10,100,1e3]:
        for reg3 in [1e-3,1e-2,0.1,0.6,1,10,100]:
            for reg4 in [1e-3,1e-2,1e-1,1,10,100,1e3]:
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
                            CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, re_constant3=reg3,re_constant4=reg4,ds=num_class,
                                         kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, restore_path=restore_path, logs_path=logs_path,rawImg=rawImg)

                            avg_i = test_face(Img, Label, CAE, num_class,lr2)
                            avg.append(avg_i)
                            iter_loop = iter_loop + 1
                            #visualize(Img, Label, CAE)
                        iter_loop = 0
                        results.append(1 - avg[0])
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

    print(results)
    rs = '['
    for i in range(len(results)):
        rs += ('%.3f' % results[i])
        rs += ' '
        if i%5==4:
            rs+=';'
    rs += ']'
    print(rs)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
