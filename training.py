import numpy as np
class Training():
	def __init__(self):
		self.train_data=None
		self.label=None
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
	def pegasos(data,lam,t,T):
		
if __name__=='__main__':
	filepath='dataset/dataset1-a8a-training.txt'
	t=Training()
	t.read_from_file(filepath)
