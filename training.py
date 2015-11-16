import numpy as np
import random
import math
class Training():
	def __init__(self):
		self.train_data=None
		self.label=None
		self.train_data,self.label=self.read_from_file('dataset/dataset1-a8a-training.txt')
	def get_example_num(self):
		return len(self.train_data)
	def read_from_file(self,filename):
		train_data=[]
		label=[]
		with open(filename,'r') as f:
			for line in f:
				t = [int(x) for x in line.split(',')[:-1]]
				l=int(line.rsplit(',',1)[-1].strip())
				train_data.append(t)
				label.append(l)
		train_data=np.array(train_data)
		label=np.array(label)
		return train_data,label
	def pegasos_hinge_loss(self,data,label,lam,T,ts):
		max_row=len(data)-1
		w=np.zeros(len(data[0]))
		index_t=0
		ws=[]
		for t in range(1,T+1):
			rand=random.randint(0,max_row)
			x=data[rand]
			y=label[rand]
			temp=y*np.inner(w,x)
			w*=(1-1/t)
			if temp<1:
				w+=(1/(lam*t))*y*x
			if t==ts[index_t]:
				ws.append(w)
				index_t+=1
		return ws
	def pegasos_log_loss(self,data,label,lam,T,ts):
		max_row=len(data)-1
		w=np.zeros(len(data[0]))
		index_t=0
		ws=[]
		for t in range(1,T+1):
			rand=random.randint(0,max_row)
			x=data[rand]
			y=label[rand]
			w=(1-1/t)*w+(1/(lam*t))*(y/(1+math.e**(y*np.inner(w,x))))*x
			if t==ts[index_t]:
				ws.append(w)
				index_t+=1
		return ws
if __name__=='__main__':
	filepath='dataset/dataset1-a8a-training.txt'
	t=Training()
	T=5*t.get_example_num()
	ts=[0.1*T,0.2*T,0.3*T,0.4*T,0.5*T,0.6*T,0.7*T,0.8*T,0.9*T,T]
	ws=t.pegasos_log_loss(t.train_data,t.label,1e-5,T,ts)
	print(ws)
