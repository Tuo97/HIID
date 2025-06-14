import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import collections

class Data(object):
    def __init__(self, path, batch_size):                         #数据载入及初始化
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0

        self.trainUser, self.trainItem = [], []

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():                              #读取训练集每一行
                if len(l) > 0:
                    l = l.strip('\n').split(' ')                 #生成列表
                    items = [int(i) for i in l[1:]]              #遍历除第一个元素（User ID）的所有值 生成Item列表
                    uid = int(l[0])                              #读取User ID
                    self.exist_users.append(uid)                 #生成User ID列表
                    self.trainUser.extend([uid] * len(items))    #uid[0, 0, 0, 0, 1, 1]
                    self.trainItem.extend(items)                 #iid[13,14,15,16,3,4]
                    self.n_items = max(self.n_items, max(items)) #生成最大item数量
                    self.n_users = max(self.n_users, uid)        #生成最大user数量
                    self.n_train += len(items)                   #生成训练item数量

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))  #更新最大item数量
                    self.n_test += len(items)                     #生成测试item数量
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)   #创建稀疏评分图

        self.train_items, self.test_set = {}, {}
        self.train_users = collections.defaultdict(list)
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]       #训练集 遍历User ID 和对应的交互Item

                    for i in train_items:                        #生成训练集评分图 有交互为1
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1
                        self.train_users[i].append(uid)

                    self.train_items[uid] = train_items          #生成训练集交互字典

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]         #测试集 遍历User ID和对应的交互Item
                    self.test_set[uid] = test_items               #生成测试集交互字典

        self.user_neigh = collections.defaultdict(list)
        max_u = 0
        min_u = 500
        for uid in range(self.n_users):
            user_neigh = self.train_items[uid]
            user_neigh = list(set(user_neigh))
            self.user_neigh[uid] = user_neigh
            lenth = len(user_neigh)
            max_u = max(max_u, lenth)
            min_u = min(min_u, lenth)
            #if lenth == 0:
                #print(uid)
            #if lenth >= 10:
                #items = rd.sample(user_neigh, 10)
                #self.user_neigh[uid] = items
            #else:
                #items = [rd.choice(user_neigh) for _ in range(10)]
                #self.user_neigh[uid] = items
        ##print(max_u)
        ##print(min_u)

        max_i = 0
        min_i = 500
        self.item_neigh = collections.defaultdict(list)
        for iid in range(self.n_items):
            item_neigh = self.train_users[iid]
            item_neigh = list(set(item_neigh))
            self.item_neigh[iid] = item_neigh
            lenth = len(item_neigh)
            max_i = max(max_i, lenth)
            min_i = min(min_i, lenth)
            #if lenth == 0:

                #self.item_neigh[iid] = [None]
                #continue
            #if lenth >= 10:
                #users = rd.sample(item_neigh, 10)
                #self.item_neigh[iid] = users
            #else:
                #users = [rd.choice(item_neigh) for _ in range(10)]
                #self.item_neigh[iid] = users
        ##print(max_i)
        ##print(min_i)


    def get_ii_uu_mat(self):
        i_i_mat = self.R.T.dot(self.R)
        u_u_mat = self.R.dot(self.R.T)
        #u_u_mat = u_u_mat.todok()
        #i_i_mat = i_i_mat.todok()

        def normalized_adj_double(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()  # 生成D^-1/2度矩阵
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()
        #u_u_mat_norm = normalized_adj_double(u_u_mat)
        #i_i_mat_norm = normalized_adj_double(i_i_mat)

        return u_u_mat.tocsr(), i_i_mat.tocsr()

    def get_adj_mat(self):
        try:  # 加载邻接矩阵
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')

            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()  # 创建邻接矩阵
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)

        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')  # 加载预训练邻接矩阵
            # print('already load pre_adj matrix')
        except Exception:  # 创建预训练邻接矩阵
            adj_mat = adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()  # 生成D^-1/2度矩阵
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)  # 双边标准化
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            # sp.save_npz(self.path + '/s_pre_adj_mat.npz', pre_adj_mat)

        return adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()                                 #生成D^-1度矩阵
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))     #单边标准化+自环
        mean_adj_mat = normalized_adj_single(adj_mat)                                #单边标准化

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)                      #抽取用户的batch_size
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u]+self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))           #生成总交互数量
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items))) #测试交互+训练交互/总交互
