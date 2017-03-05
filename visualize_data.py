#CMS 155
#Project 3
#Python 2.7
#run to get visualization graphs

from data_handler import DataHandler

dh = DataHandler()

#dh.ratings_hist()
#dh.most_popular_hist()
#dh.best_hist()
genreList = ['Action', 'Musical', 'Documentary']
for genre in genreList:
    dh.genre_hist(genre)