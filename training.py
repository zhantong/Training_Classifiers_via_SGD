import numpy as np
import random
import math
import matplotlib.pyplot as plt
class Training():
	def __init__(self):
		self.train_data=None
		self.label=None
		self.train_data,self.label=self.read_from_file('dataset/dataset1-a8a-training.txt')
	def get_example_num(self):
		return len(self.train_data)
	def read_from_file(self,filename):
		data=[]
		label=[]
		with open(filename,'r') as f:
			for line in f:
				t = [int(x) for x in line.split(',')[:-1]]
				l=int(line.rsplit(',',1)[-1].strip())
				data.append(t)
				label.append(l)
		data=np.array(data)
		label=np.array(label)
		return data,label
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
			w=(1-1/t)*w
			if temp<1:
				w=w+(1/(lam*t))*y*x
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
				print(t)
				ws.append(w)
				index_t+=1
		return ws
	def test_error(self,w,test,label):
		res=np.inner(w,test)
		wrong=0
		for (t,l) in zip(res,label):
			if (t>0 and l==-1) or (t<0 and l==1):
				wrong+=1
		return wrong/len(res)
	def test_error_s(self,ws):
		result=[]
		test,label=self.read_from_file('dataset/dataset1-a8a-testing.txt')
		for w in ws:
			test_err=self.test_error(w,test,label)
			result.append(test_err)
		return result
	def to_img(self,xs,ys):
		plt.plot(xs,ys)
		plt.axis([0.1,1,0,1])
		plt.show()
if __name__=='__main__':
	filepath='dataset/dataset1-a8a-training.txt'
	t=Training()
	T=5*t.get_example_num()
	ts=[0.1*T,0.2*T,0.3*T,0.4*T,0.5*T,0.6*T,0.7*T,0.8*T,0.9*T,T]
	ws=t.pegasos_hinge_loss(t.train_data,t.label,1e-4,T,ts)
	errors=t.test_error_s(ws)
	t.to_img([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],errors)