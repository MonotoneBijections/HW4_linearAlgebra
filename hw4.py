import sys
import numpy as np
import pandas as pd

def load(fname):
	f = open(fname, 'r').readlines()
	n = len(f)
	ret = {}
	for l in f:
		l = l.split('\n')[0].split(',')
		i = int(l[0])
		ret[i] = {}
		for j in range(n):
			if str(j) in l[1:]:
				ret[i][j] = 1
			else:
				ret[i][j] = 0
	ret = pd.DataFrame(ret).values
	return ret

def get_tran(g):
	# TODO
	# print(g)
	# print(len(g))
	weight = []
	for i in range(len(g)):
		tmp = 0 
		for j in range(len(g)):
			if g[j][i] == 1: 
				tmp+=1.0 
		weight.append(tmp)
	# print(weight)
	
	np_g=np.zeros((len(g),len(g)))
	for i in range(len(g)) :
		for j in range(len(g)):
			np_g[i][j]= float(g[i][j])/weight[j]

	# print(np_g)
	g = np_g
	# print(g)
			
			
	
	return g

def cal_rank(t, d = 0.85, max_iterations = 1000, alpha = 0.001):
	# TODO
	#init vector
	# print(len(t))
	r_ori =np.zeros((len(t),1))
	r_0 =np.zeros((len(t),1))

	tmp = 1.0/ len(t)
	tmp2= (1.0 -d)/ len(t)
	for  i in range(len(t)) :
		r_ori[i] = tmp
		r_0[i]= tmp2
	r_next = r_0+ np.matmul(t, r_ori) *d
	num =1 
	# print(dist(r_ori,r_next))
	while(num< max_iterations and dist(r_ori,r_next)>alpha) :
		
		num+=1 
		# print(num)
		r_ori = r_next 
		r_next = r_0+ np.matmul(t, r_ori) *d
	# print(dist(r_ori,r_next))
	# print(r_next)
	#returns the rank 
	rev_rank = np.argsort(r_next , axis = 0 )
	# print(rev_rank)
	rank = np.zeros((10,2)) 
	for i in range(10) :
		rank[i][0] = i+1 
		rank[i][1] =  rev_rank[len(rev_rank)-1-i][0]
	
	

	return rank 
def save(t, r):
	np.savetxt('1.txt', t, delimiter=' ',fmt='%.3f')
	np.savetxt('2.txt', r, delimiter=' ',fmt='%d')


	
	return

def dist(a, b):
	return np.sum(np.abs(a-b))

def main():
	graph = load(sys.argv[1])
	transition_matrix = get_tran(graph)
	rank = cal_rank(transition_matrix)
	save(transition_matrix, rank)

if __name__ == '__main__':
	main()

