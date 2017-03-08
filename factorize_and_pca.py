import numpy as np
#import scipy.linalg.svd as svd
import matplotlib.pyplot as plt
import pickle
import random
from data_handler import DataHandler
from sklearn.decomposition import PCA
from numpy.linalg import svd


##################### TA code ##################
def grad_U(Ui, Yij, Vj, reg, eta):
    return (1-reg*eta)*Ui + eta * Vj * (Yij - np.dot(Ui,Vj))     

def grad_V(Vj, Yij, Ui, reg, eta):  
    return (1-reg*eta)*Vj + eta * Ui * (Yij - np.dot(Ui,Vj))

def get_err(U, V, Y):
    err = 0.0
    for (i,j,Yij) in Y:
        err += 0.5 *(Yij - np.dot(U[i-1], V[:,j-1]))**2
    return err / float(len(Y))

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    U = np.random.random((M,K)) - 0.5
    V = np.random.random((K,N)) - 0.5
    size = Y.shape[0]
    delta = None
    print("training reg = %s, k = %s, M = %s, N = %s"%(reg, K, M, N))
    indices = range(size)    
    for epoch in range(max_epochs):
        # Run an epoch of SGD
        before_E_in = get_err(U, V, Y)
        np.random.shuffle(indices)
        for ind in indices:
            (i,j, Yij) = Y[ind]
            # Update U[i], V[j]
            U[i-1] = grad_U(U[i-1], Yij, V[:,j-1], reg, eta)
            V[:,j-1] = grad_V(V[:,j-1], Yij, U[i-1], reg, eta);
        # At end of epoch, print E_in
        E_in = get_err(U,V,Y)
        print("Epoch %s, E_in (MSE): %s"%(epoch + 1, E_in))

        # Compute change in E_in for first epoch
        if epoch == 0:
            delta = before_E_in - E_in

        # If E_in doesn't decrease by some fraction <eps>
        # of the initial decrease in E_in, stop early            
        elif before_E_in - E_in < eps * delta:
            break

    return (U, V, get_err(U,V,Y))


##################### Our code ##################
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

def train_latent_vectors():
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

	return (U,V)

def visualize(movies,title):
	movies_proj = V_proj[:,movies]
	#plt.plot(V_proj[0],V_proj[1], 'o')
	# if annotate:
	# 	for i in range(len(V_proj[0])):
	# 		# skip movie names with weird ascii characters
	# 		if i in [542, 1004, 1103, 1232, 1251, 1321, 1365, 1622, 1632]:
	# 			continue
	# 		ax.annotate(dh.movie_names[i+1], xy=(V_proj[0][i],V_proj[1][i]))
	plt.plot(movies_proj[0],movies_proj[1], 'ro')
	plt.axhline(0, color='black')
	plt.axvline(0, color='black')
	plt.title(title)
	plt.show()


######################################################################
################################ MAIN ################################
######################################################################


dh = DataHandler()
M = dh.num_users
N = dh.num_movies
data = np.array(dh.rating_data)

train = False
K = 20
eta = 0.01
reg = 0.5
regularize = True
if reg == 0.0:
	regularize = False

if train:
	U,V, err = train_model(M, N, K, eta, reg, data)
	#U,V = train_latent_vectors()
	if regularize:
		pickle.dump(U, open('toSave/U_TA_reg.p', 'wb'))
		pickle.dump(V, open('toSave/V_TA_reg.p', 'wb'))
	else:
		pickle.dump(U, open('toSave/U_TA.p', 'wb'))
		pickle.dump(V, open('toSave/V_TA.p', 'wb'))
else:
	if regularize:
		U = pickle.load(open('toLoad/U_TA_reg.p', 'rb'))
		V = pickle.load(open('toLoad/V_TA_reg.p', 'rb'))
	else:
		U = pickle.load(open('toLoad/U_TA.p', 'rb'))
		V = pickle.load(open('toLoad/V_TA.p', 'rb'))



A,E,B = svd(V)
V_svdComp = A[:,:2]
V_proj = np.matmul(V_svdComp.transpose(),V)
#U_pca = np.matmul(V_pcaComp,U)

# Star Wars movies
starwars_movies = [50, 181, 172]
visualize(starwars_movies,'Star Wars Movies')

# # 10 random movies
# random10 = random.sample(range(N),10)
# visualize(random10,'10 Random Movies')
#
# # 10 most popular movies
# most_popular = dh.get_most_popular()
# visualize(most_popular,'Most Popular Movies')
#
# # 10 best movies
# best_movies = dh.get_best()
# visualize(best_movies,'Best Movies')
#
# # for movie in best_movies:
# # 	print dh.movie_names[movie]
# # 	print dh.movie_ratings[movie]['rating_sum']
# # 	print dh.movie_ratings[movie]['total']
#
# # genres
# genre_list = ['Film-Noir', 'Horror', 'Western']
# for genre in genre_list:
# 	movies_by_genre = dh.get_movies_by_genre(genre)
# 	title = genre + ' Movies'
# 	visualize(movies_by_genre,title)




