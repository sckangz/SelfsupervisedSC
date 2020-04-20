import scipy.io as sio
import os,traceback

from AEnet_13 import ConvAE
from AEutils import  *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


d = 16
alpha = 6
ro = 0.03

data = sio.loadmat('./Data//COIL20.mat')
Img = data['fea']
Label = data['gnd']
Img = np.reshape(Img,(Img.shape[0],32,32,1))

n_input = [32,32]
kernel_size = [3]
n_hidden = [15]
batch_size = 20*72
model_path = './models/model-32x32-coil.ckpt'
ft_path = './models/model-32x32-coil.ckpt'
logs_path = './logs'

num_class = 20 #how many class we sample
num_sa = 72

batch_size_test = num_sa * num_class


iter_ft = 0
ft_times = 120
display_step = 300

fine_step = -1

reg1 = 1.0e-4
reg02 = 100
reg03 = 1e-5

mm = 0
mreg=[0,0,0,0]
mlr2 = 0
startfrom = [0,0,0]

# for reg2 in [50,60,80]:
# 	for reg3 in [6,8,10]:
# 		for reg4 in [0.5,1,2,5]:
for reg2 in [1e-3,1e-2,1e-1,1,10,100,1e3]:
	for reg3 in [1e-3,1e-2,1e-1,1,10,100,1e3]:#[12,15,20]:
		for reg4 in [1e-3,1e-2,1e-1,1,10,100,1e3]:
			for learning_rate in [1e-4]:
				try:
					if reg2<startfrom[0] or (reg2==startfrom[0] and reg3<startfrom[1]) or (reg2==startfrom[0] and reg3==startfrom[1] and learning_rate<startfrom[2]):
						continue
					print("reg:", reg2, reg3,reg4,learning_rate)
					tf.reset_default_graph()
					CAE = ConvAE(n_input = n_input, n_hidden = n_hidden, reg_constant1 = reg1, re_constant2 = reg2, re_constant3 = reg3, re_constant4=reg4,ds=num_class, kernel_size = kernel_size,
								batch_size = batch_size_test, model_path=model_path, restore_path=model_path, logs_path= logs_path)

					acc_= []
					for i in range(0,1):
						coil20_all_subjs = Img
						coil20_all_subjs = coil20_all_subjs.astype(float)
						label_all_subjs = Label
						label_all_subjs = label_all_subjs - label_all_subjs.min() + 1
						label_all_subjs = np.squeeze(label_all_subjs)

						CAE.initlization()
						CAE.restore()
						COLD = None
						lastr = 1.0
						losslist = []
						for iter_ft  in range(ft_times):
							cost, C,dd,dt = CAE.partial_fit(coil20_all_subjs, learning_rate, mode='fine')  #
							losslist.append(cost[-1])
							if iter_ft % display_step == 0 and iter_ft > 10:
								print ("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0]/float(batch_size_test)))
								print(cost)
								for posti in range(2):
									display(C, coil20_all_subjs, d, alpha, ro, num_class, label_all_subjs)

							if COLD is not None:
								normc = np.linalg.norm(COLD,ord='fro')
								normcd =np.linalg.norm(C-COLD,ord='fro')
								r = normcd / normc
								# print(epoch,r)
								if r < 1.0e-6 and lastr < 1.0e-6:
									print("early stop")
									print("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0] / float(batch_size_test)))
									print(cost)
									for posti in range(2):
										display(C, coil20_all_subjs, d, alpha, ro, num_class, label_all_subjs)
									break
								lastr = r
							COLD = C

						print("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0] / float(batch_size_test)))
						print(cost)

						# drawC(C)
						# print(C)
						for posti in range(1):
							acc,L,_ = display(C, coil20_all_subjs, d, alpha, ro, num_class, label_all_subjs)
							acc_.append(acc)
						acc_.append(acc)

						# for sd in [12,16]:
						# 	for sa in [6,8]:
						# 		for sr in [0.02,0.03,0.04]:
						# 			print(sd, sa, sr)
						# 			display(C, coil20_all_subjs, sd, sa, sr, num_class, label_all_subjs)

					acc_ = np.array(acc_)
					print(acc_)
					lossnp = np.asarray(losslist)
					#np.savetxt("loss-l2.csv", lossnp, delimiter=',')
					if max(acc_) > mm:
						drawC(L, 'L-COIL20-L2.png')
						mreg= [reg2,reg3,reg4,learning_rate]
						mm = max(acc_)
					print("max:", mreg, mm)
				except:
					traceback.print_exc()
				finally:
					try:
						CAE.sess.close()
					except:
						''



