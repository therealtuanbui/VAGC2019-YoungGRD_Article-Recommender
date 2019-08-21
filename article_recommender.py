import pymongo
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
try:
    from configs import constants
except:
    import constants
import pickle
import os
import itertools
import random
import pickle as pkl


class ArticleRecommender:
    def __init__(self, k=2, mongo_uri=None, mongo_db=None, sim_func=cosine_similarity, uuCF=1):
        if mongo_uri is not None:
            client = pymongo.MongoClient(mongo_uri)
            self.db = client[mongo_db]
        self.uuCF = uuCF  # user-user (1) or article-article (0) CF
        self.model_dir = constants.A_RECOMMENDER_MODEL #'model/'
        self.Y_data = None
        self.Y_data_num = None
        self.k = k  # number of neighborhood
        self.sim_func = sim_func
        self.Ybar = None  # normalized data
        self.n_users = None  # number of users
        self.n_articles = None  # number of articles
        self.distinct_users = None
        self.distinct_articles = None
        self.mu = None
        self.max_count = 20000

    def get_all_users(self):
        users = self.db.users.find({}, {'id': 1})
        return [x['id'] for x in users]

    def get_all_articles(self):
        articles = self.db.articles.find({}, {'id': 1})
        return [x['id'] for x in articles]

    def get_articles_read_by_users(self):
        u_answers = self.db.user_answers.find({}, {'u_id': 1,
                                                   'a_id': 1}) \
                                        .sort([('submited_time', pymongo.DESCENDING)]) \
                                        .limit(self.max_count)
        Y_data = []
        for a in u_answers:
            Y_data.append([a['u_id'], a['a_id'], 1])
        # remove duplicates
        Y_data.sort()
        Y_data = list(Y_data for Y_data, _ in itertools.groupby(Y_data))
        Y_data = np.array(Y_data)
        return Y_data

    def get_articles_clicked_by_users(self):
        user_clicks = self.db.user_click.find({}, {'u_id': 1, 'a_id': 1}) \
                                        .sort([('clicked_time', pymongo.DESCENDING)]) \
                                        .limit(self.max_count)
        Y_data = []
        for c in user_clicks:
            Y_data.append([c['u_id'], c['a_id'], 1])
        # remove duplicates
        Y_data.sort()
        Y_data = list(Y_data for Y_data, _ in itertools.groupby(Y_data))
        Y_data = np.array(Y_data)
        return Y_data

    def get_articles_for_random(self, u, num):
        user_clicks = self.db.user_click.find({'u_id': u}, {'a_id': 1, 'topic': 1}) \
                                        .sort([('clicked_time', pymongo.DESCENDING)]) \
                                        .limit(self.max_count)
        clicked_articles = []  # all articles clicked by user "u"
        clicked_topics = []  # all topics clicked by user "u"
        for click in user_clicks:
            clicked_articles.append(click['a_id'])
            clicked_topics.append(click['topic'])
        # print('user ',u, 'has clicked on ', len(clicked_articles), ' articles')
        clicked_topics = list(set(clicked_topics))

        all_unclicked_articles_by_topics = self.db.articles.find({
                                                                'topic': {'$in': clicked_topics}, 
                                                                'id': {'$nin': clicked_articles},
                                                                'flag': 1
                                                            }, {
                                                                'id': 1
                                                            }) \
                                                            .sort([('publish_time', pymongo.DESCENDING)]) \
                                                            .limit(num*50)
        articles_for_random = [x['id']
                               for x in all_unclicked_articles_by_topics]
        # print('number of articles for recommending: ', len(articles_for_random))

        return articles_for_random, clicked_articles

    def dump_data(self, fname, f):
        pkl.dump(f, open(fname, 'wb'), pickle.HIGHEST_PROTOCOL)

    def load_data(self, fname):
        return pkl.load(open(fname, 'rb'))

    def fit(self):
        Y_data = self.get_articles_clicked_by_users()
        self.Y_data = Y_data if self.uuCF else Y_data[:, [
            1, 0, 2]]  # a 2d array of shape (n_users, 3)
        # each row of Y_data has form [u_id, a_id, rating]
        self.Y_data_num = self.Y_data.copy()
        self.distinct_users = list(set(self.Y_data_num[:, 0]))
        self.distinct_articles = list(set(self.Y_data_num[:, 1]))

        self.n_users = len(self.distinct_users)
        self.n_articles = len(self.distinct_articles)

        clicked_users = self.Y_data_num[:, 0]  # all users' hashed ids
        clicked_articles = self.Y_data_num[:, 1]  # all articles' hashed ids
        for i, n in enumerate(self.distinct_articles):
            ids = np.where(clicked_articles == n)[0].astype(np.int32)
            self.Y_data_num[ids, 1] = i
        for i, n in enumerate(self.distinct_users):
            ids = np.where(clicked_users == n)[0].astype(np.int32)
            self.Y_data_num[ids, 0] = i
        self.Y_data_num = self.Y_data_num.astype(np.int32)
        self.Ybar = self.Y_data_num.copy()

#         self.mu = np.zeros((self.n_users,))
#         for i,n in enumerate(distinct_users):
#             # row indices of ratings made by user n
#             ids = np.where(users == n)[0].astype(np.int32)
#             article_ids = self.Y_data[ids, 1] # indices of all articles rated by user n
#             ratings = self.Y_data[ids, 2].astype(np.float32)  # ratings made by user n
#             self.mu[i] = np.mean(ratings) if ids.size > 0 else 0 # avoid zero division
# #             self.Ybar[ids, 2] = ratings - self.mu[i] #not subtracted by mean because all ratings are 0s
        # form the rating matrix as a sparse matrix.
        self.Ybar = sparse.coo_matrix(
            (self.Ybar[:, 2], (self.Ybar[:, 1], self.Ybar[:, 0])), (self.n_articles, self.n_users)).tocsr()
        self.S = self.sim_func(self.Ybar.T, self.Ybar.T)
        # dump model
        self.dump_data(self.model_dir+'distinct_users.dat',
                       self.distinct_users)
        self.dump_data(self.model_dir+'distinct_articles.dat',
                       self.distinct_articles)
        self.dump_data(self.model_dir+'Y_data.dat', self.Y_data)
        self.dump_data(self.model_dir+'Y_data_num.dat', self.Y_data_num)
        self.dump_data(self.model_dir+'S.dat', self.S)
        self.dump_data(self.model_dir+'Ybar.dat', self.Ybar)

    def __pred(self, u, i):
        self.distinct_users = self.load_data(
            self.model_dir+'distinct_users.dat')
        self.distinct_articles = self.load_data(
            self.model_dir+'distinct_articles.dat')
        self.Y_data = self.load_data(self.model_dir+'Y_data.dat')
        self.Y_data_num = self.load_data(self.model_dir+'Y_data_num.dat')
        self.S = self.load_data(self.model_dir+'S.dat')
        self.Ybar = self.load_data(self.model_dir+'Ybar.dat')
        try:
            u_num = self.distinct_users.index(u)
            i_num = self.distinct_articles.index(i)
        except ValueError:
            return 0.0
        ids = np.where(self.Y_data_num[:, 1] == i_num)[
            0].astype(np.int32)  # find article i
        users_rated_i = (self.Y_data_num[ids, 0]).astype(
            np.int32)  # all users who rated i
#         if len(users_rated_i < 2): return 0 #at least 2 users rating i is required
#         print('users rated', self.Y_data[ids,0])
        # similarity of u and users who rated i
        sim = self.S[u_num, users_rated_i]
#         print('sim: ',sim)

        deleted_ids = []
        for i, s in enumerate(sim):
            if s < 0.1:
                deleted_ids.append(i)
        sim = np.delete(sim, deleted_ids)
        users_rated_i = np.delete(users_rated_i, deleted_ids)
        nns = np.argsort(sim)[-self.k:]  # most k similar users
#         print('nns',nns)
        nearest_s = sim[nns]  # and the corresponding similarities
        r = self.Ybar[i_num, users_rated_i[nns]]  # the corresponding ratings
#         print('r',r)
        eps = 1e-8  # a small number to avoid zero division
        return (r*nearest_s).sum()/(np.abs(nearest_s).sum() + eps)
#         return (r*nearest_s).sum()/(np.abs(nearest_s).sum() + eps) + self.mu[u]

    def pred(self, u, i):
        if self.uuCF:
            return self.__pred(u, i)
        return self.__pred(i, u)

    def recommend(self, u, num=10):
        """
        Determine all articles should be recommended for user u.
        The decision is made based on chosen topics and all i such that:
        self.pred(u, i) > 0. Suppose we are considering articles which 
        have not been rated by u yet. 
        """
#         all_articles = self.get_all_articles()
#         unclicked_articles = np.delete(all_articles, np.where(all_articles==clicked_articles)
        articles_for_random, clicked_articles = self.get_articles_for_random(
            u, num)

        # choose ratio for number of articles from random and algo
        if len(clicked_articles) < 10:
            ran_num = num
        else:
            ran_num = num//2
        algo_num = num - ran_num

        # randomized articles
        # use only random in case algo cannot recommend any article
        ran_ids = random.sample(range(len(articles_for_random)), num)
        recommended_by_random = [articles_for_random[i] for i in ran_ids]

        # articles recommended by algo
        recommended_by_algo = []
        for a in articles_for_random:
            if a not in recommended_by_random:
                rating = self.pred(u, a)
#                 print('rating for article ',a , ' is: ', rating)
                if rating > 0:
                    recommended_by_algo.append(a)
        #print('number of articles recommended by algo: ', len(recommended_by_algo))

        if len(recommended_by_algo) < algo_num:
            ran_num = num - len(recommended_by_algo)
        recommended = np.concatenate(
            (recommended_by_random[:ran_num], recommended_by_algo[:algo_num]))
        #print('number of articles recommended by random: ', ran_num)

        return recommended


def main():
    recommender = ArticleRecommender(
        mongo_uri=constants.MONGO_URI, mongo_db=constants.MONGO_DB, uuCF=1)
    # recommender.fit()
    recommended_articles = recommender.recommend('43292', num=20)
    print('number of recommended articles', len(recommended_articles))
    print(recommended_articles)


if __name__ == '__main__':
    main()
