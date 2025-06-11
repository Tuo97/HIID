import tensorflow as tf

import os
import sys
import random as rd
import pickle
import scipy.sparse as sp
import numpy as np
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utility.helper import *
from utility.batch_test import *
from tqdm import tqdm


class HIID(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings

        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.plain_adj = data_config['plain_adj']  #A
        self.norm_adj = data_config['norm_adj']    #D-1/2*A*D-1/2

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.n_layers = args.n_layers
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        self.groups1 = 4
        self.groups2 = 2

        self.alpha = tf.constant(0.2)
        self.beta = tf.constant(0.2)
        self.gamma = tf.constant(0.6)

        self.ssm_temp = tf.constant(0.2)
        self.ssm_reg = tf.constant(0.1)

        self.diversity_temp = tf.constant(0.2)
        self.diversity_reg = tf.constant(0.1)


        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        # tf.compat.v1.disable_eager_execution()
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        # create models
        self.ua_embeddings_1, self.ia_embeddings_1, self.ua_embeddings_2, self.ia_embeddings_2, self.ua_embeddings_3, self.ia_embeddings_3 = self.hiid()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings_1 = tf.nn.embedding_lookup(self.ua_embeddings_1, self.users)
        self.pos_i_g_embeddings_1 = tf.nn.embedding_lookup(self.ia_embeddings_1, self.pos_items)
        self.neg_i_g_embeddings_1 = tf.nn.embedding_lookup(self.ia_embeddings_1, self.neg_items)

        self.u_g_embeddings_2 = tf.nn.embedding_lookup(self.ua_embeddings_2, self.users)
        self.pos_i_g_embeddings_2 = tf.nn.embedding_lookup(self.ia_embeddings_2, self.pos_items)
        self.neg_i_g_embeddings_2 = tf.nn.embedding_lookup(self.ia_embeddings_2, self.neg_items)

        self.u_g_embeddings_3 = tf.nn.embedding_lookup(self.ua_embeddings_3, self.users)
        self.pos_i_g_embeddings_3 = tf.nn.embedding_lookup(self.ia_embeddings_3, self.pos_items)
        self.neg_i_g_embeddings_3 = tf.nn.embedding_lookup(self.ia_embeddings_3, self.neg_items)

        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        # Inference for the testing phase.
        self.batch_ratings = self.hie_scores(self.u_g_embeddings_1, self.pos_i_g_embeddings_1,
                                             self.u_g_embeddings_2, self.pos_i_g_embeddings_2,
                                             self.u_g_embeddings_3, self.pos_i_g_embeddings_3)

        # Generate Predictions & Optimize via BPR loss.
        self.mf_loss, self.emb_loss = self.create_bpr_loss(self.u_g_embeddings_1, self.pos_i_g_embeddings_1, self.neg_i_g_embeddings_1,
                                                           self.u_g_embeddings_2, self.pos_i_g_embeddings_2, self.neg_i_g_embeddings_2,
                                                           self.u_g_embeddings_3, self.pos_i_g_embeddings_3, self.neg_i_g_embeddings_3)

        self.div_loss = self.create_diversity_loss()
        self.ssm_loss = self.create_ssm_loss(self.u_g_embeddings_1, self.pos_i_g_embeddings_1,
                                             self.u_g_embeddings_2, self.pos_i_g_embeddings_2,
                                             self.u_g_embeddings_3, self.pos_i_g_embeddings_3)

        self.loss = self.mf_loss + self.emb_loss + self.ssm_loss + self.div_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        #initializer = tf.keras.initializers.glorot_uniform()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')

            for k in range(self.groups1):
                all_weights['Q_%d' % k] = tf.Variable(initializer([1, 2, 1]), name='Q_%d' % k)

            for k in range(self.groups2):
                all_weights['Q1_%d' % k] = tf.Variable(initializer([1, self.groups1, 1]), name='Q1_%d' % k)

            all_weights['Q2'] = tf.Variable(initializer([1, self.groups2, 1]), name='Q2')

            print('using xavier initialization')

        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True, name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True, name='item_embedding', dtype=tf.float32)

            for k in range(self.groups1):
                all_weights['Q_%d' % k] = tf.Variable(initializer([1, 4, 1]), name='Q_%d' % k)

            print('using pretrained initialization')

        return all_weights

    def hiid(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
###############################################    GCN    #######################################################
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = []
        for k in range(0, self.n_layers):
            if k > 0:
              ego_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings)
              all_embeddings += [ego_embeddings]

            else:
              ego_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings)

        all_embeddings = tf.stack(all_embeddings, 2)               #[N,64,layers]
###############################################    HIE1   #######################################################
        out = []
        att = []
        for k in range(self.groups1):
            att_temp = tf.matmul(all_embeddings, self.weights['Q_%d' % k])   #[N,64,1]
            att += [att_temp]

        att = tf.concat(att, axis=2)                                   #[N,64,groups1]
        att = tf.nn.sigmoid(att)
        att = tf.split(att, self.groups1, axis=2)

        for k in range(self.groups1):
            out_temp = tf.multiply(att[k], all_embeddings)                   #[N,64,layers]
            out_temp = tf.reduce_mean(out_temp, axis=2, keep_dims=False)     #[N,64]
            out += [out_temp]

        all_embeddings = tf.stack(out, axis=2)
        hie_embeddings_1 = tf.concat(out, axis=1)

#####################################################    HIE2    ####################################################
        out1 = []
        att1 = []
        for k in range(self.groups2):
            att_temp1 = tf.matmul(all_embeddings, self.weights['Q1_%d' % k])  # [N,64,1]
            att1 += [att_temp1]

        att1 = tf.concat(att1, axis=2)                                       # [N,64, self.groups2]
        att1 = tf.nn.sigmoid(att1)
        att1 = tf.split(att1, self.groups2, axis=2)

        for k in range(self.groups2):
            out_temp1 = tf.multiply(att1[k], all_embeddings)                 #[N,64, self.groups2]
            out_temp1 = tf.reduce_mean(out_temp1, axis=2, keep_dims=False)   #[N, 64]
            out1 += [out_temp1]

        all_embeddings = tf.stack(out1, axis=2)
        hie_embeddings_2 = tf.concat(out1, axis=1)

 #######################################################   HIE3    ###################################################
        att2 = tf.matmul(all_embeddings, self.weights['Q2'])
        att2 = tf.nn.sigmoid(att2)
        out2 = tf.multiply(att2, all_embeddings)
        hie_embeddings_3 = tf.reduce_sum(out2, axis=2, keep_dims=False)
########################################################   OutPut  ####################################################
        u_g_embeddings_1, i_g_embeddings_1 = tf.split(hie_embeddings_1, [self.n_users, self.n_items], 0)
        u_g_embeddings_2, i_g_embeddings_2 = tf.split(hie_embeddings_2, [self.n_users, self.n_items], 0)
        u_g_embeddings_3, i_g_embeddings_3 = tf.split(hie_embeddings_3, [self.n_users, self.n_items], 0)
        return u_g_embeddings_1, i_g_embeddings_1, u_g_embeddings_2, i_g_embeddings_2, u_g_embeddings_3, i_g_embeddings_3

    def create_bpr_loss(self, users_1, pos_items_1, neg_items_1, users_2, pos_items_2, neg_items_2, users_3, pos_items_3, neg_items_3):
        pos_scores_3 = tf.reduce_sum(tf.multiply(users_3, pos_items_3), axis=1)
        neg_scores_3 = tf.reduce_sum(tf.multiply(users_3, neg_items_3), axis=1)
        ############################    HIE2 ########################################
        pos_scores_2 = tf.reshape(tf.multiply(users_2, pos_items_2), [-1, self.emb_dim, self.groups2])
        neg_scores_2 = tf.reshape(tf.multiply(users_2, neg_items_2), [-1, self.emb_dim, self.groups2])

        pos_scores_2 = tf.reduce_sum(pos_scores_2, axis=1, keep_dims=False)     #[N, self.groups1]
        neg_scores_2 = tf.reduce_sum(neg_scores_2, axis=1, keep_dims=False)     #[N, self.groups1]

        pos_scores_2 = tf.reduce_mean(pos_scores_2, axis=1, keep_dims=False)
        neg_scores_2 = tf.reduce_mean(neg_scores_2, axis=1, keep_dims=False)

        ############################# HIE3 ###########################################
        pos_scores_1 = tf.reshape(tf.multiply(users_1, pos_items_1), [-1, self.emb_dim, self.groups1])
        neg_scores_1 = tf.reshape(tf.multiply(users_1, neg_items_1), [-1, self.emb_dim, self.groups1])

        pos_scores_1 = tf.reduce_sum(pos_scores_1, axis=1, keep_dims=False)  # [N, self.groups1]
        neg_scores_1 = tf.reduce_sum(neg_scores_1, axis=1, keep_dims=False)  # [N, self.groups1]

        pos_scores_1 = tf.reduce_mean(pos_scores_1, axis=1, keep_dims=False)
        neg_scores_1 = tf.reduce_mean(neg_scores_1, axis=1, keep_dims=False)
        ##############################################################################
        pos_scores = self.alpha*pos_scores_1 + self.beta*pos_scores_2 + self.gamma*pos_scores_3
        neg_scores = self.alpha*neg_scores_1 + self.beta*neg_scores_2 + self.gamma*neg_scores_3

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size

        ## In this version,correct and high effitiveness
        mf_loss = tf.reduce_sum(-tf.log_sigmoid(pos_scores-neg_scores))

        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss

    def create_ssm_loss(self, users_1, pos_items_1, users_2, pos_items_2, users_3, pos_items_3):
        norm_users_3 = tf.nn.l2_normalize(users_3, axis=1)
        norm_items_3 = tf.nn.l2_normalize(pos_items_3, axis=1)

        pos_score_3 = tf.reduce_sum(tf.multiply(norm_users_3, norm_items_3), axis=1)
        ttl_score_3 = tf.matmul(norm_users_3, norm_items_3, transpose_a=False, transpose_b=True)

        ######################################  HIE2  ################################################
        users_2 = tf.split(users_2, self.groups2, axis=1)   #[N, 64]*self.groups2
        pos_items_2 = tf.split(pos_items_2, self.groups2, axis=1)

        pos_score_2 = []
        ttl_score_2 = []
        for k in range(self.groups2):
            norm_users_2_temp = tf.nn.l2_normalize(users_2[k], axis=1)
            norm_items_2_temp = tf.nn.l2_normalize(pos_items_2[k], axis=1)

            pos_score_2_temp = tf.reduce_sum(tf.multiply(norm_users_2_temp, norm_items_2_temp), axis=1)
            ttl_score_2_temp = tf.matmul(norm_users_2_temp, norm_items_2_temp, transpose_a=False, transpose_b=True)

            pos_score_2 += [pos_score_2_temp]
            ttl_score_2 += [ttl_score_2_temp]

        pos_score_2 = tf.stack(pos_score_2, axis=1)    #[N,   groups2]
        ttl_score_2 = tf.stack(ttl_score_2, axis=2)    #[N,N, groups2]

        pos_score_2 = tf.reduce_mean(pos_score_2, axis=1)   #[N]
        ttl_score_2 = tf.reduce_mean(ttl_score_2, axis=2)   #[N, N]
        ######################################  HIE3 #################################################
        users_1 = tf.split(users_1, self.groups1, axis=1)   # [N, 64]*self.groups2
        pos_items_1 = tf.split(pos_items_1, self.groups1, axis=1)

        pos_score_1 = []
        ttl_score_1 = []
        for k in range(self.groups1):
            norm_users_1_temp = tf.nn.l2_normalize(users_1[k], axis=1)
            norm_items_1_temp = tf.nn.l2_normalize(pos_items_1[k], axis=1)

            pos_score_1_temp = tf.reduce_sum(tf.multiply(norm_users_1_temp, norm_items_1_temp), axis=1)
            ttl_score_1_temp = tf.matmul(norm_users_1_temp, norm_items_1_temp, transpose_a=False, transpose_b=True)

            pos_score_1 += [pos_score_1_temp]
            ttl_score_1 += [ttl_score_1_temp]

        pos_score_1 = tf.stack(pos_score_1, axis=1)  # [N,   groups2]
        ttl_score_1 = tf.stack(ttl_score_1, axis=2)  # [N,N, groups2]

        pos_score_1 = tf.reduce_mean(pos_score_1, axis=1)  # [N]
        ttl_score_1 = tf.reduce_mean(ttl_score_1, axis=2)  # [N, N]

        ####################################  LOSS ##############################################
        pos_score = self.alpha*pos_score_1 + self.beta*pos_score_2 + self.gamma*pos_score_3
        ttl_score = self.alpha*ttl_score_1 + self.beta*ttl_score_2 + self.gamma*ttl_score_3

        pos_score = tf.exp(pos_score/self.ssm_temp)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score/self.ssm_temp), axis=1)

        loss = -tf.reduce_sum(tf.log(pos_score/ttl_score))
        loss = self.ssm_reg * loss

        return loss

    def create_diversity_loss(self):
        query_1 = []
        query_2 = []

        for k in range(self.groups1):
            query_1_temp = tf.squeeze(self.weights['Q_%d' % k])
            query_1 += [query_1_temp]
        query_1 = tf.stack(query_1, axis=0)   #[groups1, 2]

        for k in range(self.groups2):
            query_2_temp = tf.squeeze(self.weights['Q1_%d' % k])
            query_2 += [query_2_temp]
        query_2 = tf.stack(query_2, axis=0)  # [groups2, groups1]

        ####################################   Hie1 ######################################
        query_1 = tf.nn.l2_normalize(query_1, 1)
        pos_scores_1 = tf.reduce_sum(tf.multiply(query_1, query_1), axis=1)
        ttl_scores_1 = tf.matmul(query_1, query_1, transpose_a=False, transpose_b=True)
        pos_scores_1 = tf.exp(pos_scores_1/self.diversity_temp)
        ttl_scores_1 = tf.reduce_sum(tf.exp(ttl_scores_1/self.diversity_temp), axis=1)
        loss1 = -tf.reduce_sum(tf.log(pos_scores_1/ttl_scores_1))
        ####################################   Hie2 ######################################
        query_2 = tf.nn.l2_normalize(query_2, 1)
        pos_scores_2 = tf.reduce_sum(tf.multiply(query_2, query_2), axis=1)
        ttl_scores_2 = tf.matmul(query_2, query_2, transpose_a=False, transpose_b=True)
        pos_scores_2 = tf.exp(pos_scores_2 / self.diversity_temp)
        ttl_scores_2 = tf.reduce_sum(tf.exp(ttl_scores_2 / self.diversity_temp), axis=1)
        loss2 = -tf.reduce_sum(tf.log(pos_scores_2 / ttl_scores_2))

        loss = self.diversity_reg*(loss1+loss2)
        return loss

    def hie_scores(self, users_1, pos_items_1, users_2, pos_items_2, users_3, pos_items_3):
        batch_item = tf.shape(pos_items_1)[0]
        batch_user = tf.shape(users_1)[0]

        users_1 = tf.tile(users_1, multiples=[1, batch_item])
        users_1 = tf.reshape(users_1, shape=[-1, (self.emb_dim * self.groups1)])
        pos_items_1 = tf.tile(pos_items_1, multiples=[batch_user, 1])

        users_2 = tf.tile(users_2, multiples=[1, batch_item])
        users_2 = tf.reshape(users_2, shape=[-1, (self.emb_dim * self.groups2)])
        pos_items_2 = tf.tile(pos_items_2, multiples=[batch_user, 1])

        users_3 = tf.tile(users_3, multiples=[1, batch_item])
        users_3 = tf.reshape(users_3, shape=[-1, self.emb_dim])
        pos_items_3 = tf.tile(pos_items_3, multiples=[batch_user, 1])

        ############################     HIE3 #######################################
        pos_scores_3 = tf.reduce_sum(tf.multiply(users_3, pos_items_3), axis=1)
        ############################    HIE2 ########################################
        pos_scores_2 = tf.reshape(tf.multiply(users_2, pos_items_2), [-1, self.emb_dim, self.groups2])
        pos_scores_2 = tf.reduce_sum(pos_scores_2, axis=1, keep_dims=False)  # [N, self.groups2]
        pos_scores_2 = tf.reduce_mean(pos_scores_2, axis=1, keep_dims=False)
        ############################# HIE1 ###########################################
        pos_scores_1 = tf.reshape(tf.multiply(users_1, pos_items_1), [-1, self.emb_dim, self.groups1])
        pos_scores_1 = tf.reduce_sum(pos_scores_1, axis=1, keep_dims=False)  # [N, self.groups1]
        pos_scores_1 = tf.reduce_mean(pos_scores_1, axis=1, keep_dims=False)
        ##############################################################################
        pos_scores = self.alpha*pos_scores_1 + self.beta*pos_scores_2 + self.gamma*pos_scores_3

        scores = tf.reshape(pos_scores, shape=[batch_user, batch_item])
        return scores


    def model_save(self, ses):
        save_pretrain_path = '../output_parameters/ml1m'
        np.savez(save_pretrain_path, user_embed=np.array(self.weights['user_embedding'].eval(session=ses)),
                 item_embed=np.array(self.weights['item_embedding'].eval(session=ses)),
                 Q_0=np.array(self.weights['Q_0'].eval(session=ses)),
                 Q_1=np.array(self.weights['Q_1'].eval(session=ses)),
                 Q_2=np.array(self.weights['Q_2'].eval(session=ses)),
                 Q_3=np.array(self.weights['Q_3'].eval(session=ses)),
                 Q1_0=np.array(self.weights['Q1_0'].eval(session=ses)),
                 Q1_1=np.array(self.weights['Q1_1'].eval(session=ses)),
                 Q2=np.array(self.weights['Q2'].eval(session=ses))
                 )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape



def load_best(name="best_model"):
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, name)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the best model:', name)
    except Exception:
        pretrain_data = None
    return pretrain_data

def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)
    return all_h_list, all_t_list, all_v_list


if __name__ == '__main__':
    whether_test_batch = True
    print("************************* Run with following settings 🏃 ***************************")
    print(args)
    print("************************************************************************************")

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    #config['trainUser'] = data_generator.trainUser
    #config['trainItem'] = data_generator.trainItem

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    #all_h_list, all_t_list, all_v_list = load_adjacency_list_data(plain_adj)
    #u_u_mat, i_i_mat = data_generator.get_ii_uu_mat()

    config['plain_adj'] = plain_adj   #A
    config['norm_adj'] = pre_adj      #D-1/2*A*D-1/2
    #config['all_h_list'] = all_h_list
    #config['all_t_list'] = all_t_list
    #config['u_u_mat'] = u_u_mat
    #config['i_i_mat'] = i_i_mat

    t0 = time()
    """
    *********************************************************
    pretrain = 1: load embeddings with name such as embedding_xxx(.npz), l2_best_model(.npz)
    pretrain = 0: default value, no pretrained embeddings.
    """
    if args.pretrain == 1:
        print("Try to load pretain: ", args.embed_name)
        pretrain_data = load_best(name=args.embed_name)
        if pretrain_data == None:
            print("Load pretrained model(%s)fail!!!!!!!!!!!!!!!" % (args.embed_name))
    else:
        pretrain_data = None

    model = HIID(data_config=config, pretrain_data=pretrain_data)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    cur_best_pre_0 = 0.

    """
    *********************************************************
    Train
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, ssm_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in tqdm(range(n_batch)):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_ssm_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.ssm_loss],
                                                                                    feed_dict={model.users: users,
                                                                                               model.pos_items: pos_items,
                                                                                               model.neg_items: neg_items})
            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch
            ssm_loss += batch_ssm_loss / n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            print(mf_loss, emb_loss)
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % args.show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, emb_loss, ssm_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True, batch_test_flag=whether_test_batch)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ssm_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=args.early)
        # early stopping when cur_best_pre_0 is decreasing for given steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            model.model_save(sess)
            print('save the model with performance: ', cur_best_pre_0)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
