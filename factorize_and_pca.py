import numpy as np
#import scipy.linalg.svd as svd
import matplotlib.pyplot as plt
import pickle
import random
from data_handler import DataHandler
from sklearn.decomposition import PCA
from numpy.linalg import svd


##################### TA code ##################
def grad_U(Ui, Yij, Vj, ai, bj, reg, bi, eta):
    return (1-reg*eta)*Ui + eta * Vj * (Yij - np.dot(Ui,Vj) + bi*ai + bi*bj)     

def grad_V(Vj, Yij, Ui, ai, bj, reg, bi, eta):  
    return (1-reg*eta)*Vj + eta * Ui * (Yij - np.dot(Ui,Vj) + bi*ai + bi*bj)

def grad_a(Ui, Yij, Vj, ai, bj, reg, bi, eta):
    return (1-reg*eta)*ai + eta * (Yij - np.dot(Ui,Vj) + bi*ai + bi*bj)

def grad_b(Vj, Yij, Ui, ai, bj, reg, bi, eta):
    return (1-reg*eta)*bj + eta * (Yij - np.dot(Ui,Vj) + bi*ai + bi*bj)

def get_err(U, V, a, b, Y, reg, bi):
    err = 0.0
    for (i,j,Yij) in Y:
        err += 0.5 *(Yij - np.dot(U[i-1], V[:,j-1]) + bi*a[i-1] + bi*b[j-1])**2
    return err / float(len(Y)) + reg*reg_err(U,V,a,b,bi)

def reg_err(U, V, a, b, bi):
    ls = (np.linalg.norm(U, ord='fro')**2 + np.linalg.norm(V, ord='fro')**2)
    return ls + bi*( np.linalg.norm(a) + np.linalg.norm(b) )

def train_model(M, N, K, eta, reg, bi, Y, eps=0.0001, max_epochs=300):
    U = np.random.random((M,K)) - 0.5
    V = np.random.random((K,N)) - 0.5
    if bi == 0:
        a = np.zeros(M)
        b = np.zeros(N)
    else:
        a = np.random.random(M) - 0.5
        b = np.random.random(N) - 0.5
    size = Y.shape[0]
    delta = None
    print("training reg = %s, k = %s, M = %s, N = %s"%(reg, K, M, N))
    indices = range(size)    
    for epoch in range(max_epochs):
        # Run an epoch of SGD
        before_E_in = get_err(U, V, a, b, Y, reg, bi)
        np.random.shuffle(indices)
        for ind in indices:
            (i,j, Yij) = Y[ind]
            # Update U[i], V[j]
            U[i-1] = grad_U(U[i-1], Yij, V[:,j-1], a[i-1], b[j-1], reg, bi, eta)
            V[:,j-1] = grad_V(V[:,j-1], Yij, U[i-1], a[i-1], b[j-1], reg, bi, eta)
            # Update a, b
            a[i-1] = grad_a(U[i-1], Yij, V[:,j-1], a[i-1], b[j-1], reg, bi, eta)
            b[j-1] = grad_b(V[:,j-1], Yij, U[i-1], a[i-1], b[j-1], reg, bi, eta)
        # At end of epoch, print E_in
        E_in = get_err(U,V,a,b,Y,reg,bi)
        print("Epoch %s, E_in (MSE): %s"%(epoch + 1, E_in))

        # Compute change in E_in for first epoch
        if epoch == 0:
            delta = before_E_in - E_in

        # If E_in doesn't decrease by some fraction <eps>
        # of the initial decrease in E_in, stop early            
        elif before_E_in - E_in < eps * delta:
            break

    return (U, V, get_err(U,V,a,b,Y,0,0))


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

def visualize(movies,title,annotate=True,filename=''):
	badset = [542, 1004, 1103, 1232, 1251, 1321, 1365, 1622, 1632]
	badset = [m+1 for m in badset]
	movies = [m-1 for m in movies]
	movies_proj = V_proj[:, movies]

	fig = plt.figure()
	ax  = fig.add_subplot(111)
	#plt.plot(V_proj[0],V_proj[1], 'o')
	if annotate:
		for i in movies:
			# skip movie names with weird ascii characters
			if i in badset:
				continue
			ax.annotate(dh.movie_names[i+1], xy=(V_proj[0][i],V_proj[1][i]))
	plt.plot(movies_proj[0],movies_proj[1], 'ro')
	#if annotate:
	#	for i in range(len(movies_proj[0])):
	#		ax.annotate(dh.movie_names[movies[i]], xy=(movies_proj[0][i], movies_proj[1][i]))
	plt.axhline(0, color='black')
	plt.axvline(0, color='black')
	# plt.xlim(-.5,2.5) $ boundaries for regularized
	# plt.ylim(-1,1)
	plt.xlim(-2.7,0.7) # boundaries for unregularized
	plt.ylim(-1.7,1.7)
	plt.title(title)
	#fig.set_size_inches(9, 6)
	if len(filename) > 0:
		plt.savefig('Plots/factorized/'+filename+'.png')
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
regularize = False
bi = 0

if train:
	U,V,err = train_model(M, N, K, eta, reg, bi, data)
	#U,V = train_latent_vectors()
	if not reg == 0.0:
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

# # Star Wars movies
# starwars_movies = [50, 181, 172]
# visualize(starwars_movies,'Star Wars Movies')

# # select movies
# select_movies = [50, 181, 172, 69, 22, 550, 144]
# visualize(select_movies,'Select Movies',annotate=True)

# 10 random movies
random10 = random.sample(range(N),10)
visualize(random10,'10 Random Movies',filename='randommovies')

# 10 most popular movies
most_popular = dh.get_most_popular()
visualize(most_popular,'Most Popular Movies',filename='popularmovies')

# 10 best movies
best_movies = dh.get_best()
visualize(best_movies,'Best Movies',filename='bestmovies')

# genres
genre_list = ['Action', 'Childrens', 'Comedy', 'Documentary', 'Film-Noir']
for genre in genre_list:
	movies_by_genre = dh.get_movies_by_genre(genre)
	title = genre + ' Movies'
	visualize(movies_by_genre,title,annotate=False,filename=genre)




