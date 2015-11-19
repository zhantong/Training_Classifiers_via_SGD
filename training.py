import numpy as np
import random
import math
import matplotlib.pyplot as plt
class Training():
	def __init__(self,training_file,test_file):
		self.train_data=None
		self.label=None
		self.train_data,self.train_label=self.read_from_file(training_file)
		self.test_data,self.test_label=self.read_from_file(test_file)
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
	def pegasos_hinge_loss(self,data,label,lam,ts,T):
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
	def pegasos_log_loss(self,data,label,lam,ts,T):
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
	def test_error(self,w,test,label):
		res=np.inner(w,test)
		wrong=0
		for (t,l) in zip(res,label):
			if (t>0 and l==-1) or (t<0 and l==1):
				wrong+=1
		return wrong/len(res)
	def test_error_s(self,ws):
		result=[]
		for w in ws:
			test_err=self.test_error(w,self.test_data,self.test_label)
			result.append(test_err)
		return result
	def to_img(self,ys,title):
		xs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
		plt.plot(xs,ys)
		plt.title(title)
		plt.ylabel('test error')
		plt.xlabel('number of iterations')
		plt.axis([0.1,1,0,1])
		plt.xticks(xs,[r'$0.1T$',r'$0.2T$',r'$0.3T$',r'$0.4T$',r'$0.5T$',r'$0.6T$',r'$0.7T$',r'$0.8T$',r'$0.9T$',r'$1T$'])
		plt.savefig(title+".png")
		plt.close()
	def get_ts(self,T):
		result=[]
		for i in range(1,11,1):
			result.append(int(T*i*0.1))
		return result
	def start(self,lam,title):
		T=5*len(self.train_data)
		ts=self.get_ts(T)
		test_error=self.test_error_s(self.pegasos_hinge_loss(self.train_data,self.train_label,lam,ts,T))
		name=title+" hinge loss"
		self.to_img(test_error,name)
		print(name+"\t",test_error)
		test_error=self.test_error_s(self.pegasos_log_loss(self.train_data,self.train_label,lam,ts,T))
		name=title+" log loss"
		self.to_img(test_error,name)
		print(name+"\t",test_error)
if __name__=='__main__':
	t=Training('dataset/dataset1-a8a-training.txt','dataset/dataset1-a8a-testing.txt')
	t.start(1e-4,"dataset 1")
	t=Training('dataset/dataset1-a9a-training.txt','dataset/dataset1-a9a-testing.txt')
	t.start(5e-5,"dataset 2")

