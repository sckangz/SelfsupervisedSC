import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import traceback
from AEnet_13 import ConvAE
from AEutils import  *

# SELECT GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
pred = False
def test_face(Img, Label, CAE, num_class,lr2=5e-4):

    d = 3
    alpha = 1.2
    ro = 0.18
    
    acc_= []
    for i in range(0,41-num_class): 
        face_10_subjs = np.array(Img[10*i:10*(i+num_class),:])
        face_10_subjs = face_10_subjs.astype(float)        
        label_10_subjs = np.array(Label[10*i:10*(i+num_class)]) 
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs) 
                     
        CAE.initlization()        
        CAE.restore() # restore from pre-trained model
        # global pred
        # if pred==True:
        #     CAE.restore()
        # else:
        #     pre_step = 9000
        #     epoch = 0
        #     while epoch < pre_step:
        #         epoch = epoch + 1
        #         cost = CAE.partial_pre(face_10_subjs, 1e-4)
        #         if epoch % 300 == 0:
        #             print("pre epoch: %.1d" % epoch, "cost: %.8f" % (cost[0]/float(batch_size)))
        #     CAE.save_model()
        #     pred = True
        max_step = 4500#50 + num_class*25# 100+num_class*20
        display_step = 8000#max_step/20#10
        lr = 1.0
        fine_step = -1
        # fine-tune network
        epoch = 0
        COLD = None
        lastr = 1.0
        while epoch < max_step:
            epoch = epoch + 1
            if epoch <= fine_step:
                cost, Coef = CAE.partial_fit(face_10_subjs, lr)#
            else:
                lr = lr2
                cost, Coef,dd,dt = CAE.partial_fit(face_10_subjs, lr, mode = 'fine')  #
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
            drawC(Coef)
            acc_x,L,_ = display(Coef, label_10_subjs, d, alpha, ro)
            drawC(L, 'L-L2.png')
            acc_.append(acc_x)
        acc_.append(acc_x)

        #for sd in [3,4]:
            #for sa in [0.8,1,1.2]:
                #for sr in [0.20,0.18]:
                    #print(sd, sa, sr)
                    #display(Coef, label_10_subjs, sd, sa, sr)
    
    acc_ = np.array(acc_)
    mm = np.max(acc_)

    print("%d subjects:" % num_class)    
    print("Max: %.4f%%" % ((1-mm)*100))
    print(acc_) 
    
    return (1-mm)
    
   
        
    
if __name__ == '__main__':
    
    # load face images and labels
    data = sio.loadmat('./Data/ORL_32x32.mat')
    Img = data['fea']
    Label = data['gnd']

    model_path = './models/model-335-32x32-orl.ckpt'
    restore_path = './models/model-335-32x32-orl.ckpt'
    logs_path = './logs'

    # face image clustering
    n_input = [32, 32]
    kernel_size = [3,3,3]
    n_hidden=[3,3,5]
    Img = Img / 255.0
    Img = np.reshape(Img,[Img.shape[0],n_input[0],n_input[1],1]) 
    
    all_subjects = [40]
    reg1 = 1.0e-4
    reg02 = 2
    reg03 = 1e-1

    mm = 0
    mreg2 = 0
    mreg3 = 0
    mlr2 = 0

    startfrom = [0, 0, 0]
    mm = 0
    result=[]

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
                            batch_size = num_class * 10

                            tf.reset_default_graph()
                            CAE = ConvAE(n_input=n_input,n_hidden=n_hidden,reg_constant1=reg1,re_constant2=reg2,re_constant3=reg3,re_constant4=reg4,ds=num_class,\
                                         kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, restore_path=restore_path, logs_path=logs_path)

                            avg_i = test_face(Img, Label, CAE, num_class,lr2)
                            avg.append(avg_i)
                            iter_loop = iter_loop + 1

                        iter_loop = 0
                        result.append(avg[0])

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
    print(result)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
