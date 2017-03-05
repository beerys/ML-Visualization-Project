#CMS 155
#Project 3
#Python 2.7
#reads data. visualization types can be added as fns

import numpy as np
import matplotlib.pyplot as plt

class DataHandler(object):

    def __init__(self):
        self.data_file = 'data.txt'
        self.movie_file = 'movies.txt'
        self.rating_data = []
        self.movie_data = {}
        self.num_users = 943
        self.num_movies = 1682
        self.num_ratings = 100000

        self.read_data()

    def read_data(self):
        file = open(self.data_file,'r')
        for line in file:
            line = line.strip().split()
            for i in range(self.num_ratings):
                self.rating_data.append([int(line[3 * i]),int(line[3 * i + 1]),int(line[3 * i + 2])])

        file = open(self.movie_file, 'r')
        for line in file:
            line = line.replace('\r','\t')
            line = line.strip().split('\t')
            #print(line)
            for i in range(self.num_movies):
                self.movie_data[i+1] = line[21 * i+1:21 * i + 21]

    def ratings_hist(self):
        ratings = {}
        for i in range(6):
            ratings[i] = 0
        for elem in self.rating_data:
            ratings[elem[2]] +=1

        plt.bar(ratings.keys(),ratings.values())
        plt.title('All MovieLens ratings')
        plt.xlabel('Rating')
        plt.ylabel('Number of instances')
        plt.savefig('Plots/Basic_Visualization_1')
