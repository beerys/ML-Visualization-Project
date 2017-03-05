import numpy as np
from data_handler import DataHandler

def calc_error(data,U,V,_lambda=0.0):
	error = np.linalg.norm(U)**2 + np.linalg.norm(V)**2
	error *= _lambda/2

	for sample in data:
		i = int(sample[0])-1
		j = int(sample[1])-1
		y = int(sample[2])
		error += (y - np.dot(U[:,i],V[:,j]))**2

	return error


def run_epoch(data,U,V,_lambda,step):
	np.random.shuffle(data)
	for sample in data:
		i = int(sample[0])-1
		j = int(sample[1])-1
		y = int(sample[2])
		error = y-np.dot(U[:,i],V[:,j])
		U[:,i] -= step*(_lambda*U[:,i] - error*V[:,j])
		V[:,j] -= step*(_lambda*V[:,j] - error*U[:,i])
	return (U,V)


######################################################################
################################ MAIN ################################
######################################################################

dh = DataHandler()
M = dh.num_users
N = dh.num_movies
D = dh.num_ratings
data = np.array(dh.rating_data)

# parameters of our model
threshold = 0.0001
step = 0.01
k = 20
_lambda = 0.0


U = np.random.rand(k,M)-0.5
V = np.random.rand(k,N)-0.5
errors = [calc_error(data,U,V,_lambda)]

# calculate the first epoch
U,V = run_epoch(data,U,V,_lambda,step)
errors = np.append(errors, [calc_error(data,U,V,_lambda)])

while (errors[-2]-errors[-1]) / (errors[0]-errors[1]) >	threshold:
	U,V = run_epoch(data,U,V,_lambda,step)
	errors = np.append(errors, [calc_error(data,U,V,_lambda)])
	print errors[-1]

