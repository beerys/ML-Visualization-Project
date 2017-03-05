#CMS 155
#Project 3
#Python 2.7
#reads data. visualization types can be added as fns

import numpy as np
import matplotlib.pyplot as plt
import heapq

class DataHandler(object):

    def __init__(self):
        self.data_file = 'data.txt'
        self.movie_file = 'movies.txt'
        self.rating_data = []
        self.movie_data = {}
        self.num_users = 943
        self.movie_names = {}
        self.movie_ratings = {}
        self.num_movies = 1682
        self.num_ratings = 100000

        for i in range(self.num_movies):
            self.movie_ratings[i+1] = {}
            self.movie_ratings[i + 1]['total'] = 0
            for rating in range(1,6):
                self.movie_ratings[i+1][rating] = 0


        self.read_data()

    def read_data(self):
        file = open(self.data_file,'r')
        for line in file:
            line = line.strip().split()
            for i in range(self.num_ratings):
                user = int(line[3 * i ])
                movie = int(line[3 * i + 1])
                rating = int(line[3 * i + 2])
                self.rating_data.append([user,movie,rating])
                self.movie_ratings[movie][rating] += 1
                self.movie_ratings[movie]['total'] += 1

        file = open(self.movie_file, 'r')
        for line in file:
            line = line.replace('\r','\t')
            line = line.strip().split('\t')
            for i in range(self.num_movies):
                self.movie_names[i+1] = line[21 * i+1]
                self.movie_data[i+1] = line[21 * i+2:21 * i + 21]


    def ratings_hist(self):
        ratings = {}
        for i in range(1,6):
            ratings[i] = 0
        for elem in self.rating_data:
            ratings[elem[2]] += 1

        plt.bar(ratings.keys(),ratings.values())
        plt.title('All MovieLens ratings')
        plt.xlabel('Rating')
        plt.ylabel('Number of instances')
        plt.savefig('Plots/Basic_Visualization_1')
        plt.clf()

    def most_popular_hist(self):
        #this is the first type of visualization tried
        # ratings = {}
        # for i in range(1,6):
        #     ratings[i] = 0
        # for elem in top10:
        #     for i in range(1,6):
        #         ratings[i] += self.movie_ratings[elem[1]][i]

        #this is the second type
        top10 = self.get_most_popular()
        plt.title('Ratings of most popular movies')

        fig, axes = plt.subplots(nrows=2, ncols=5)
        fig.tight_layout()
        for i in range(1,11):
            plt.subplot(2,5,i)
            ratings = self.movie_ratings[top10[i]]
            del ratings['total']
            plt.bar(ratings.keys(),ratings.values())
            # plt.xlabel('Rating')
            # plt.ylabel('Number of instances')
            plt.title(self.movie_names[top10[i]])
        plt.savefig('Plots/Basic_Visualization_2_type2')
        plt.clf()

    def get_most_popular(self):
        top10 = []
        for i in range(self.num_movies):
            if len(top10) <= 10:
                heapq.heappush(top10,(self.movie_ratings[i+1]['total'],i+1))
            else:
                heapq.heappushpop(top10,(self.movie_ratings[i+1]['total'],i+1))
        return [i[1] for i in top10]