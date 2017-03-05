#CMS 155
#Project 3
#Python 2.7
#reads data. visualization types can be added as fns


import numpy as np

class DataHandler(object):

    def __init__(self):
        self.data_file = 'data.txt'
        self.movie_file = 'movies.txt'
        self.rating_data = []
        self.movie_data = {}
        self.num_movies = 1682

        self.read_data()

    def read_data(self):
        file = open(self.data_file,'r')
        for line in file:
            line = line.strip().split()
            self.rating_data.append([int(line[0]),int(line[1]),int(line[2])])

        file = open(self.movie_file, 'r')
        for line in file:
            line = line.replace('\r','\t')
            line = line.strip().split('\t')
            #print(line)
            for i in range(self.num_movies):
                self.movie_data[i+1] = line[21 * i+1:21 * i + 21]
