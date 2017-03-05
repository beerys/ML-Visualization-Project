import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from data_handler import DataHandler
from sklearn.decomposition import PCA

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

######################################################################
################################ MAIN ################################
######################################################################

dh = DataHandler()
M = dh.num_users
N = dh.num_movies
D = dh.num_ratings
data = np.array(dh.rating_data)

train = False

if train:
    U,V = train_latent_vectors()
    pickle.dump(U, open('toSave/U.p', 'wb'))
    pickle.dump(V, open('toSave/V.p', 'wb'))
else:
    U = pickle.load(open('toLoad/U.p', 'rb'))
    V = pickle.load(open('toLoad/V.p', 'rb'))

pca = PCA(n_components=2)
pca.fit(V)
V_pca = pca.components_

# 10 random movies
random10 = random.sample(range(N),10)
random10_pca = V_pca[:,random10]
plt.plot(V_pca[0],V_pca[1], 'o')
plt.plot(random10_pca[0],random10_pca[1], 'ro')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title('10 Random Movies PCA')
plt.show()

# 10 most popular movies
most_popular = dh.get_most_popular()
most_popular_pca = V_pca[:,dh.get_most_popular()]
plt.plot(V_pca[0],V_pca[1], 'o')
plt.plot(most_popular_pca[0],most_popular_pca[1], 'ro')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title('Most Popular Movies PCA')
plt.show()

# 10 best movies
best_movies = dh.get_best()
best_movies_pca = V_pca[:,best_movies]
plt.plot(V_pca[0],V_pca[1], 'o')
plt.plot(best_movies_pca[0],best_movies_pca[1], 'ro')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title('Best Movies PCA')
plt.show()

# for movie in best_movies:
# 	print dh.movie_names[movie]
# 	print dh.movie_ratings[movie]['rating_sum']
# 	print dh.movie_ratings[movie]['total']

# genres
genre_list = ['Action', 'Horror', 'Western']
for genre in genre_list:
	movies_by_genre = dh.get_movies_by_genre(genre)
	genre_movies_pca = V_pca[:,movies_by_genre]
	plt.plot(V_pca[0],V_pca[1], 'o')
	plt.plot(genre_movies_pca[0],genre_movies_pca[1], 'ro')
	plt.axhline(0, color='black')
	plt.axvline(0, color='black')
	plt.title(genre + ' Movies PCA')
	plt.show()


