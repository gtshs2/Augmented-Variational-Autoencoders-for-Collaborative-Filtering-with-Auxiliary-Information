import tensorflow as tf
import time
from DataReader import DataReader

from tensorflow.contrib.layers import batch_norm
import logging

from utils import evaluation
import numpy as np
from os import mkdir
from os.path import exists

class NeuralVariationalCollabFiltering():
    def __init__(self, sess, data_reader, NS, activation_fn, mode, k_list, loss_fn, batch_size=500, decay_rate=0.96,
                 decay_epoch_step=50, embed_dim=500, latent_dim=50, learning_rate=0.001, batch_normalization=False,
                 max_epochs=1000, max_iter=5000, densify=False, user_side_info=None, item_side_info=None,
                 alternate_opt=False, printing=False):

        self.sess = sess
        self.data_reader = data_reader
        self.mode = mode
        self.k_list = k_list
        self.loss_fn = loss_fn
        self.NS = NS
        self.printing = printing

        self.activation_fn = activation_fn
        self.init_batch_size = batch_size
        self.decay_rate = decay_rate
        self.decay_epoch_step = decay_epoch_step
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.batch_normalization = batch_normalization

        self.max_epochs = max_epochs
        self.max_iter = max_iter

        self.densify = densify
        self.user_side_info = user_side_info
        self.item_side_info = item_side_info
        self.alternate_opt = alternate_opt

        self.step = tf.Variable(0, trainable=False)

        self.num_first_tr, self.num_second_tr, self.user_ids_tr, self.item_ids_tr, self.ratings_tr, \
        self.lda_formed_data_ids_tr, self.lda_formed_data_vals_tr = self.data_reader.get_data("train")

        self.set_second_ids = set(self.item_ids_tr)

        self.num_ratings_tr = len(self.ratings_tr)
        self.mean_rating_tr = np.mean(self.ratings_tr)
        # self.stddev_ratings_tr = np.std(self.ratings_tr)
        self.mean_rating_tr = 0.0
        self.stddev_ratings_tr = 1.0
        self.max_rating_tr = np.max(self.ratings_tr)
        self.min_rating_tr = np.min(self.ratings_tr)

        self.decay_step = self.decay_epoch_step * int(self.num_first_tr / self.init_batch_size)

        # self.lr = tf.train.exponential_decay(
        #     learning_rate, self.step, self.decay_step, decay_rate, staircase=True, name="lr"
        # )
        self.lr = tf.constant(learning_rate, dtype=tf.float32)

        self.num_first_val, self.num_second_val, self.user_ids_val, self.item_ids_val, self.ratings_val, \
        self.lda_formed_data_ids_val, self.lda_formed_data_vals_val = self.data_reader.get_data("valid")
        self.num_ratings_val = len(self.ratings_val)
        self.mean_ratings_val = np.mean(self.ratings_val)

        self.num_first_te, self.num_second_te, self.user_ids_te, self.item_ids_te, self.ratings_te, \
        self.lda_formed_data_ids_te, self.lda_formed_data_vals_te = self.data_reader.get_data("test")
        self.num_ratings_te = len(self.ratings_te)
        self.mean_ratings_te = np.mean(self.ratings_te)

        self.sparse_ratings = np.zeros((self.num_first_tr, self.num_second_tr), dtype=float)

        self.num_data = len(self.lda_formed_data_ids_tr)
        self._permutation = np.random.permutation(self.num_data)

        self.batch_counter = 0
        self.epochs = 0

        self.first_dim = self.num_first_tr
        self.second_dim = self.num_second_tr

        if self.user_side_info:
            self.side_info = "USER_SI"
        elif self.item_side_info:
            self.side_info = "ITEM_SI"
        elif not self.user_side_info and not self.item_side_info:
            self.side_info = "NONE_SI"
        else:
            raise NotImplementedError("Both SI not available")

        if (self.mode == "user"):
            # User-based
            print("[TRAIN] User-Based-%s: # of num_users: %d, # of num_items: %d, # of ratings: %d"
                  % (self.side_info, self.first_dim, self.second_dim, self.num_ratings_tr))
        elif (self.mode == "item"):
            # Item-based
            print("[TRAIN] Item-Based-%s: # of num_users: %d, # of num_items: %d, # of ratings: %d"
                  % (self.side_info, self.second_dim, self.first_dim, self.num_ratings_tr))
        else:
            raise NotImplementedError("Learning mode should be \"user\" or \"item\"")

        self._attrs = ["latent_dim", "embed_dim", "max_iter",
                       "learning_rate", "decay_rate", "decay_step"]

        self.prepare_model()
        self.build_encoder()
        self.build_decoder()
        self.build_objective_fn()
        # self.train()

    def build_objective_fn(self):

        if self.loss_fn == "Gaussian":
            if self.densify:
                self.reconstr_loss = tf.reduce_sum(
                    0.5 * self.batch_C * (tf.square(self.batch_dense_ratings - self.dec_x_mu))
                )
            else:
                self.reconstr_loss = tf.reduce_sum(
                    0.5 * self.batch_C * (tf.square(self.batch_sparse_ratings.values - self.sliced_dec_x_mu))
                )
        elif self.loss_fn == "Sigmoid":
            if not self.densify:
                self.reconstr_loss = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.sliced_dec_x_mu,
                        targets=self.batch_sparse_ratings.values,
                        name="sigmoid_loss"
                    )
                )
            else:
                self.reconstr_loss = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.dec_x_mu,
                        targets=self.batch_sparse_ratings,
                        name="sigmoid_loss"
                    )
                )

        elif self.loss_fn == "Poisson":
            self.reconstr_loss = tf.reduce_sum(
                tf.nn.log_poisson_loss(log_input=self.sliced_dec_x_mu,
                                       targets=self.batch_sparse_ratings.values,
                                       name="Poisson_loss")
            )

        """
        \beta variational autoencoder scheme
        introducing free paramter \beta to control the ralative strength of KL-divergence.
        """

        self.beta = tf.placeholder(tf.float32, name="beta")

        self.latent_loss = tf.reduce_sum(self.kullbackLeibler(self.enc_z_mu, self.enc_z_log_sigma_sq))


        self.encoder_var_list, self.decoder_var_list = [], []

        for var in tf.trainable_variables():
            if "encoder" in var.name:
                self.encoder_var_list.append(var)
            elif "decoder" in var.name:
                self.decoder_var_list.append(var)

        if not self.alternate_opt:
            if not self.densify:
                self.cost = self.reconstr_loss + self.beta * self.latent_loss
            else:
                self.cost = tf.reduce_sum(self.reconstr_loss) + self.beta * self.latent_loss

            with tf.name_scope("Adam_optimizer"):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                tvars = tf.trainable_variables()
                grads_and_vars = self.optimizer.compute_gradients(self.cost, tvars)

                clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
                           for grad, tvar in grads_and_vars]
                self.train_op = self.optimizer.apply_gradients(clipped, global_step=self.step, name="minimize_cost")
        else:
            # optimizer for alternative update
            self.enc_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.dec_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            enc_grads_and_vars = self.enc_optimizer.compute_gradients(self.latent_loss, self.encoder_var_list)
            dec_grads_and_vars = self.dec_optimizer.compute_gradients(self.reconstr_loss, self.decoder_var_list)

            enc_clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
                           for grad, tvar in enc_grads_and_vars]
            dec_clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
                           for grad, tvar in dec_grads_and_vars]

            self.enc_train_op = self.enc_optimizer.apply_gradients(enc_clipped, global_step=self.step,
                                                                   name="enc_minimize_cost")
            self.dec_train_op = self.dec_optimizer.apply_gradients(dec_clipped, global_step=self.step,
                                                                   name="dec_minimize_cost")

    def prepare_model(self):
        self.batch_first_indices = tf.placeholder(tf.int64, [None], name="batch_first_indices")
        self.batch_second_indices = tf.placeholder(tf.int64, [None], name="batch_second_indices")
        self.indices = tf.pack([self.batch_first_indices, self.batch_second_indices], axis=1)
        self.ratings = tf.placeholder(tf.float32, [None], name="batch_ratings")
        self.batch_size = tf.placeholder(tf.int64, name="batch_size")

        self.batch_sparse_ratings = tf.SparseTensor(self.indices, self.ratings,
                                                    shape=[self.batch_size, self.second_dim])
        if self.densify:
            self.batch_dense_ratings = tf.sparse_tensor_to_dense(self.batch_sparse_ratings, default_value=0.0)
            self.batch_C = tf.sparse_tensor_to_dense(self.batch_sparse_ratings, default_value=0.01)

        if self.loss_fn == "Gaussian":
            if not self.batch_normalization:
                self.variables_dict = {
                    "enc_L1_W": tf.get_variable("enc_L1_W", shape=[self.second_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_L1_b": tf.get_variable("enc_L1_b", shape=[self.embed_dim],
                                                initializer=tf.constant_initializer(0.0)),
                    "enc_L2_W": tf.get_variable("enc_L2_W", shape=[self.embed_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_L2_b": tf.get_variable("enc_L2_b", shape=[self.embed_dim],
                                                initializer=tf.constant_initializer(0.0)),
                    "enc_z_mu_W": tf.get_variable("enc_z_mu_W", shape=[self.embed_dim, self.latent_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_z_mu_b": tf.get_variable("enc_z_mu_b", shape=[self.latent_dim],
                                                  initializer=tf.constant_initializer(0.0)),
                    "enc_z_log_sigma_sq_W": tf.get_variable("enc_z_log_sigma_sq_W",
                                                            shape=[self.embed_dim, self.latent_dim],
                                                            initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_z_log_sigma_sq_b": tf.get_variable("enc_z_log_sigma_sq_b", shape=[self.latent_dim],
                                                            initializer=tf.constant_initializer(0.0)),
                    "dec_L1_W": tf.get_variable("dec_L1_W", shape=[self.latent_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_L1_b": tf.get_variable("dec_L1_b", shape=[self.embed_dim],
                                                initializer=tf.constant_initializer(0.0)),
                    "dec_L2_W": tf.get_variable("dec_L2_W", shape=[self.embed_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_L2_b": tf.get_variable("dec_L2_b", shape=[self.embed_dim],
                                                initializer=tf.constant_initializer(0.0)),
                    "dec_x_mu_W": tf.get_variable("dec_x_mu_W", shape=[self.embed_dim, self.second_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_x_mu_b": tf.get_variable("dec_x_mu_b", shape=[self.second_dim],
                                                  initializer=tf.constant_initializer(0.0))
                }
            else:
                self.variables_dict = {
                    "enc_L1_W": tf.get_variable("enc_L1_W", shape=[self.second_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_L2_W": tf.get_variable("enc_L2_W", shape=[self.embed_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_z_mu_W": tf.get_variable("enc_z_W", shape=[self.embed_dim, self.latent_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_z_mu_b": tf.get_variable("enc_z_b", shape=[self.latent_dim],
                                                  initializer=tf.constant_initializer(0.0)),
                    "enc_z_log_sigma_sq_W": tf.get_variable("enc_z_log_sigma_sq_W",
                                                            shape=[self.embed_dim, self.latent_dim],
                                                            initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_z_log_sigma_sq_b": tf.get_variable("enc_z_log_sigma_sq_b", shape=[self.latent_dim],
                                                            initializer=tf.constant_initializer(0.0)),
                    "dec_L1_W": tf.get_variable("dec_L1_W", shape=[self.latent_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_L2_W": tf.get_variable("dec_L2_W", shape=[self.embed_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_x_mu_W": tf.get_variable("dec_x_mu_W", shape=[self.embed_dim, self.second_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_x_mu_b": tf.get_variable("dec_x_mu_b", shape=[self.second_dim],
                                                  initializer=tf.constant_initializer(0.0))
                }
        else:
            if not self.batch_normalization:
                self.variables_dict = {
                    "enc_L1_W": tf.get_variable("enc_L1_W", shape=[self.second_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_L1_b": tf.get_variable("enc_L1_b", shape=[self.embed_dim],
                                                initializer=tf.constant_initializer(0.0)),
                    "enc_L2_W": tf.get_variable("enc_L2_W", shape=[self.embed_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_L2_b": tf.get_variable("enc_L2_b", shape=[self.embed_dim],
                                                initializer=tf.constant_initializer(0.0)),
                    "enc_z_mu_W": tf.get_variable("enc_z_mu_W", shape=[self.embed_dim, self.latent_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_z_mu_b": tf.get_variable("enc_z_mu_b", shape=[self.latent_dim],
                                                  initializer=tf.constant_initializer(0.0)),
                    "enc_z_log_sigma_sq_W": tf.get_variable("enc_z_log_sigma_sq_W",
                                                            shape=[self.embed_dim, self.latent_dim],
                                                            initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_z_log_sigma_sq_b": tf.get_variable("enc_z_log_sigma_sq_b", shape=[self.latent_dim],
                                                            initializer=tf.constant_initializer(0.0)),
                    "dec_L1_W": tf.get_variable("dec_L1_W", shape=[self.latent_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_L1_b": tf.get_variable("dec_L1_b", shape=[self.embed_dim],
                                                initializer=tf.constant_initializer(0.0)),
                    "dec_L2_W": tf.get_variable("dec_L2_W", shape=[self.embed_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_L2_b": tf.get_variable("dec_L2_b", shape=[self.embed_dim],
                                                initializer=tf.constant_initializer(0.0)),
                    "dec_x_mu_W": tf.get_variable("dec_x_mu_W", shape=[self.embed_dim, self.second_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_x_mu_b": tf.get_variable("dec_x_mu_b", shape=[self.second_dim],
                                                  initializer=tf.constant_initializer(0.0))
                }
            else:
                self.variables_dict = {
                    "enc_L1_W": tf.get_variable("enc_L1_W", shape=[self.second_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_L2_W": tf.get_variable("enc_L2_W", shape=[self.embed_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_z_mu_W": tf.get_variable("enc_z_W", shape=[self.embed_dim, self.latent_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_z_mu_b": tf.get_variable("enc_z_b", shape=[self.latent_dim],
                                                  initializer=tf.constant_initializer(0.0)),
                    "enc_z_log_sigma_sq_W": tf.get_variable("enc_z_log_sigma_sq_W",
                                                            shape=[self.embed_dim, self.latent_dim],
                                                            initializer=tf.contrib.layers.xavier_initializer()),
                    "enc_z_log_sigma_sq_b": tf.get_variable("enc_z_log_sigma_sq_b", shape=[self.latent_dim],
                                                            initializer=tf.constant_initializer(0.0)),
                    "dec_L1_W": tf.get_variable("dec_L1_W", shape=[self.latent_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_L2_W": tf.get_variable("dec_L2_W", shape=[self.embed_dim, self.embed_dim],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_x_mu_W": tf.get_variable("dec_x_mu_W", shape=[self.embed_dim, self.second_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer()),
                    "dec_x_mu_b": tf.get_variable("dec_x_mu_b", shape=[self.second_dim],
                                                  initializer=tf.constant_initializer(0.0))
                }

    def build_encoder(self):
        self.num_samples = tf.placeholder(dtype=tf.int64, name="num_samples")

        with tf.variable_scope("encoder"):
            if not self.batch_normalization:
                if not self.densify:
                    self.enc_layer_1 = self.activation_fn(
                        tf.sparse_tensor_dense_matmul(self.batch_sparse_ratings, self.variables_dict["enc_L1_W"])
                        + self.variables_dict["enc_L1_b"]
                    )
                else:
                    self.enc_layer_1 = self.activation_fn(
                        tf.matmul(self.batch_dense_ratings, self.variables_dict["enc_L1_W"], a_is_sparse=True)
                        + self.variables_dict["enc_L1_b"]
                    )

                self.enc_layer_2 = self.activation_fn(
                    tf.matmul(self.enc_layer_1, self.variables_dict["enc_L2_W"])
                    + self.variables_dict["enc_L2_b"]
                )

            else:
                if not self.densify:
                    self.enc_layer_1 = self.activation_fn(
                        tf.sparse_tensor_dense_matmul(self.batch_sparse_ratings, self.variables_dict["enc_L1_W"])
                    )
                else:
                    self.enc_layer_1 = self.activation_fn(
                        tf.matmul(self.batch_dense_ratings, self.variables_dict["enc_L1_W"], a_is_sparse=True)
                    )
                self.enc_layer_1 = batch_norm(self.enc_layer_1, scope="BN_enc_L1")

                self.enc_layer_2 = self.activation_fn(
                    tf.matmul(self.enc_layer_1, self.variables_dict["enc_L2_W"])
                )
                self.enc_layer_2 = batch_norm(self.enc_layer_2, scope="BN_enc_L2")

            self.enc_z_mu = tf.identity(
                tf.matmul(self.enc_layer_2, self.variables_dict["enc_z_mu_W"])
                + self.variables_dict["enc_z_mu_b"]
            )

            self.enc_z_log_sigma_sq = tf.identity(
                tf.matmul(self.enc_layer_2, self.variables_dict["enc_z_log_sigma_sq_W"])
                + self.variables_dict["enc_z_log_sigma_sq_b"]
            )

            self.candidate_eps = tf.random_normal((self.batch_size, self.latent_dim, self.num_samples), 0.0, 1.0, dtype=tf.float32)
            self.eps = tf.reduce_mean(self.candidate_eps, axis=2)
            self.enc_z_sigma = tf.sqrt(tf.exp(self.enc_z_log_sigma_sq))

            self.z = tf.add(self.enc_z_mu, tf.mul(self.enc_z_sigma, self.eps))  # N x T

    def build_decoder(self):
        """
        It should make p(r|z, h), p(x|z), p(x|h)

        We assume side information as texts.

        p(r|z, h) can be Gaussian or Poisson, and etc.
        p(x|z) and p(x|h) are multinomial logistic regression as in NVDM.

        """

        with tf.variable_scope("decoder"):
            ### TODO ###
            # estimated_ratings = tf.matmul(self.z, self.h, transpose_b=True)
            # shape = (num_users in minibatch, num_items), estimated ratings

            # flattend (and corresponding) estimated ratings

            if self.loss_fn == "Gaussian" or self.loss_fn == "Poisson":
                logit_activation = tf.identity
            else:
                logit_activation = tf.nn.sigmoid

            if not self.batch_normalization:
                self.dec_layer_1 = self.activation_fn(
                    tf.matmul(self.z, self.variables_dict["dec_L1_W"])
                    + self.variables_dict["dec_L1_b"]
                )

                self.dec_layer_2 = self.activation_fn(
                    tf.matmul(self.dec_layer_1, self.variables_dict["dec_L2_W"])
                    + self.variables_dict["dec_L2_b"]
                )
            else:
                self.dec_layer_1 = self.activation_fn(
                    tf.matmul(self.z, self.variables_dict["dec_L1_W"])
                )
                self.dec_layer_1 = batch_norm(self.dec_layer_1, scope="BN_dec_L1")

                self.dec_layer_2 = self.activation_fn(
                    tf.matmul(self.dec_layer_1, self.variables_dict["dec_L2_W"])
                )
                self.dec_layer_2 = batch_norm(self.dec_layer_2, scope="BN_dec_L2")


            self.dec_x_mu = tf.matmul(self.dec_layer_2, self.variables_dict["dec_x_mu_W"]) \
                            + self.variables_dict["dec_x_mu_b"]

            if not self.densify:
                self.sliced_dec_x_mu = tf.gather(
                    tf.reshape(self.dec_x_mu, [-1]),
                    self.batch_first_indices * self.second_dim + self.batch_second_indices
                )

    def load_batch(self, lda_formed_data_ids, lda_formed_data_vals):
        if self.batch_counter + self.init_batch_size > self.num_first_tr:
            self.batch_counter = 0
            self.epochs += 1
            self._permutation = np.random.permutation(self.num_data)

        self.this_perm = self._permutation[self.batch_counter:self.batch_counter + self.init_batch_size]
        # print(self.step, self.this_perm)
        batch_user_ids = list()  # not real user ids... just index of users in mini-batch
        batch_item_ids = list()
        batch_ratings = list()

        for i in range(len(self.this_perm)):
            # user_id = self.this_perm[i]
            user_id = i
            item_ids = lda_formed_data_ids[self.this_perm][i]
            ratings = lda_formed_data_vals[self.this_perm][i]

            num_of_NS_for_each_user = self.NS * len(item_ids)  # same for the CDAE setting
            candidate_negative_samples = np.random.permutation(
                list(self.set_second_ids - set(item_ids)))[:num_of_NS_for_each_user]
            # print(candidate_negative_samples)
            for j in range(len(item_ids)):
                batch_user_ids.append(user_id)
                batch_item_ids.append(item_ids[j])
                batch_ratings.append(ratings[j])
            if not self.densify:
                for j in range(len(candidate_negative_samples)):
                    batch_user_ids.append(user_id)
                    batch_item_ids.append(candidate_negative_samples[j])
                    batch_ratings.append(0.0)

        batch_user_ids = np.asarray(batch_user_ids, dtype=int)
        batch_item_ids = np.asarray(batch_item_ids, dtype=int)
        batch_ratings = np.asarray(batch_ratings, dtype=float)

        self.batch_counter += self.init_batch_size

        return batch_user_ids, batch_item_ids, batch_ratings


    def load_test_data(self, lda_formed_data_ids, lda_formed_data_vals):
        self.this_perm = self._permutation
        # print(self.step, self.this_perm)
        batch_user_ids = list()  # not real user ids... just index of users in mini-batch
        batch_item_ids = list()
        batch_ratings = list()

        for i in range(len(self.this_perm)):
            # user_id = self.this_perm[i]
            user_id = i
            item_ids = lda_formed_data_ids[self.this_perm][i]
            ratings = lda_formed_data_vals[self.this_perm][i]

            for j in range(len(item_ids)):
                batch_user_ids.append(user_id)
                batch_item_ids.append(item_ids[j])
                batch_ratings.append(ratings[j])


        batch_user_ids = np.asarray(batch_user_ids, dtype=int)
        batch_item_ids = np.asarray(batch_item_ids, dtype=int)
        batch_ratings = np.asarray(batch_ratings, dtype=float)

        return batch_user_ids, batch_item_ids, batch_ratings

    def train(self):

        tf.initialize_all_variables().run()

        start_time = time.time()
        start_iter = self.step.eval()
        max_R = -1.0
        for step in range(start_iter, start_iter + self.max_iter):
            annealed_beta = min(self.epochs + 1, 50.0) / 50.0  # this is the same with Ladder VAE (warm-up setting)
            if self.epochs > self.max_epochs:
                break
            """
            Here, we should access user_indices, item_indices, ratings,
            and these have to be stored in corresponding placehoder variables.
            """
            batch_user_ids, batch_item_ids, batch_ratings \
                = self.load_batch(self.lda_formed_data_ids_tr, self.lda_formed_data_vals_tr)


            if not self.alternate_opt:
                (_,
                 cost,
                 estimated_ratings,
                 reconstr_loss,
                 KL_term,
                 beta,
                 learning_rate
                 ) = \
                    self.sess.run(
                        [self.train_op, self.cost,
                         self.dec_x_mu, self.reconstr_loss, self.latent_loss, self.beta, self.lr],
                        feed_dict={self.batch_first_indices: batch_user_ids,
                                   self.batch_second_indices: batch_item_ids,
                                   self.ratings: batch_ratings - self.mean_rating_tr,
                                   self.beta: annealed_beta,
                                   self.batch_size: self.init_batch_size,
                                   self.num_samples: 1
                                   }
                    )

            else:
                (_,
                 enc_cost,
                 enc_estimated_ratings,
                 reconstr_loss,
                 KL_term,
                 beta,
                 learning_rate
                 ) = \
                    self.sess.run(
                        [self.enc_train_op, self.latent_loss,
                         self.dec_x_mu, self.reconstr_loss, self.latent_loss, self.beta, self.lr],
                        feed_dict={self.batch_first_indices: batch_user_ids,
                                   self.batch_second_indices: batch_item_ids,
                                   self.ratings: batch_ratings - self.mean_rating_tr,
                                   self.beta: annealed_beta,
                                   self.batch_size: self.init_batch_size,
                                   self.num_samples: 1
                                   }
                    )

                (_,
                 cost,
                 estimated_ratings,
                 reconstr_loss,
                 KL_term,
                 beta,
                 learning_rate
                 ) = \
                    self.sess.run(
                        [self.dec_train_op, self.reconstr_loss,
                         self.dec_x_mu, self.reconstr_loss, self.latent_loss, self.beta, self.lr],
                        feed_dict={self.batch_first_indices: batch_user_ids,
                                   self.batch_second_indices: batch_item_ids,
                                   self.ratings: batch_ratings - self.mean_rating_tr,
                                   self.beta: annealed_beta,
                                   self.batch_size: self.init_batch_size,
                                   self.num_samples: 1
                                   }
                    )


            if step % (10 * int(self.num_first_tr / self.init_batch_size)) == 0:
                print("Test result...")
                print("Epochs\t%d" % self.epochs)
                print("=====================================================")

                logging.info("Test result...")
                logging.info("Epochs\t%d" % self.epochs)
                logging.info("=====================================================")

                batch_user_ids, batch_item_ids, batch_ratings \
                    = self.load_test_data(self.lda_formed_data_ids_tr, self.lda_formed_data_vals_tr)

                (estimated_ratings) = \
                    self.sess.run(
                        [self.dec_x_mu],
                        feed_dict={self.batch_first_indices: batch_user_ids,
                                   self.batch_second_indices: batch_item_ids,
                                   self.ratings: batch_ratings,
                                   self.beta: 1.0,
                                   self.batch_size: self.num_first_tr,
                                   self.num_samples: 20
                                   }
                    )

                self.sparse_ratings[self.this_perm, :] = estimated_ratings

                precs, normalized_precs, recs, maps, MRR, NDCG = \
                    evaluation(self.k_list, self.first_dim, self.lda_formed_data_ids_tr, self.lda_formed_data_ids_val,
                               self.lda_formed_data_ids_te, self.sparse_ratings, self.user_ids_tr, self.item_ids_tr,
                               self.ratings_tr, self.user_ids_val, self.item_ids_val, self.ratings_val,
                               self.user_ids_te, self.item_ids_te, self.ratings_te,
                               self.num_ratings_tr, self.num_ratings_val, self.num_ratings_te)

                precs_name_list = ["prec@%d" % k for k in k_list]
                precs_val_list = ["%.4f" % prec_k for prec_k in precs]

                N_precs_name_list = ["N_prec@%d" % k for k in k_list]
                N_precs_val_list = ["%.4f" % N_prec_k for N_prec_k in normalized_precs]

                recs_name_list = ["rec@%d" % k for k in k_list]
                recs_val_list = ["%.4f" % rec_k for rec_k in recs]

                maps_name_list = ["map@%d" % k for k in k_list]
                maps_val_list = ["%.4f" % map_k for map_k in maps]

                if self.printing:
                    print("MRR\t\t%.4f" % MRR)
                    print("NDCG\t\t%.4f" % NDCG)

                    print("\t" + "\t".join(precs_name_list))
                    print("\t" + "\t".join(precs_val_list))

                    print("\t" + "\t".join(N_precs_name_list))
                    print("\t" + "\t".join(N_precs_val_list))

                    print("\t" + "\t".join(recs_name_list))
                    print("\t" + "\t".join(recs_val_list))

                    print("\t" + "\t".join(maps_name_list))
                    print("\t" + "\t".join(maps_val_list))

                logging.info("MRR\t\t%.4f" % MRR)
                logging.info("NDCG\t\t%.4f" % NDCG)

                logging.info("\t" + "\t".join(precs_name_list))
                logging.info("\t" + "\t".join(precs_val_list))

                logging.info("\t" + "\t".join(N_precs_name_list))
                logging.info("\t" + "\t".join(N_precs_val_list))

                logging.info("\t" + "\t".join(recs_name_list))
                logging.info("\t" + "\t".join(recs_val_list))

                logging.info("\t" + "\t".join(maps_name_list))
                logging.info("\t" + "\t".join(maps_val_list))

                logging.info("\n")

                logging.info("\n")

                if max_R < float(recs_val_list[-1]):
                    max_R = float(recs_val_list[-1])


        return max_R, float(recs_val_list[-1])

    def kullbackLeibler(self, mu, log_sigma_sq):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
            return -0.5 * tf.reduce_sum(1 + tf.clip_by_value(log_sigma_sq, -10.0, 10.0)
                                        - tf.clip_by_value(mu, -10.0, 10.0) ** 2 -
                                        tf.exp(tf.clip_by_value(log_sigma_sq, -10.0, 10.0)), 1)


# # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
def xavier_init(n_inputs, n_outputs, uniform=True):
    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
             Understanding the difficulty of training deep feedforward neural
             networks. International conference on artificial intelligence and
             statistics.
    Args:
      n_inputs: The number of input nodes into each output.
      n_outputs: The number of output nodes for each input.
      uniform: If true use a uniform distribution, otherwise use a normal.
    Returns:
      An initializer.
    """
    if uniform:
        # 6 was used in the paper.
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


if __name__ == '__main__':
    tf.set_random_seed(53)
    np.random.seed(53)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    mode = "user"  # or "item"

    k_list = [5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300]

    explicit_feedback_ = False
    explicit_threshold_ = 0.0  # CDAE used 4.0
    activation_fn = tf.nn.relu  # or tanh sigmoid
    # loss_fn = "Poisson" # Poisson loss has a worse performance in implicit & explicit settings
    loss_fn = "Sigmoid"
    # loss_fn = "Gaussian"

    data_names = ["citeulike"]
    # data_names = ["ml-10m"]
    # M_list = [5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300]

    explicit_feedback_ = False
    explicit_threshold_ = 0.0  # CDAE used 4.0

    folds = [1]
    NSs = [10]

    printing = True
    path = "./logs/VAE-CF/"
    log_file = path + "Latent_dim_varying_170216.log"
    if not exists(path):
        mkdir(path)

    logging.basicConfig(filename=log_file, level=logging.INFO)

    latent_dims = [5, 10, 50, 100, 200]
    embed_dims = [50,100,200,300,500]
    for fold in folds:
        for i in range(latent_dims):
            latent_dim = latent_dims[i]
            embed_dim = embed_dims[i]
            for data_name in data_names:
                if data_name == "aiv":
                    batch_size = 256
                    num_user = 29757
                    num_items = 15149
                elif data_name == "ml-1m":
                    batch_size = 256  # or 100 for "ml-1m"
                    num_user = 6040
                    num_item = 3544
                    lr = 1e-3
                    # NSs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                elif data_name == "ml-10m":
                    batch_size = 256
                    num_user = 69878
                    num_item = 10073
                    lr = 1e-3
                    # NSs = [1,2,3,4,5,6,7,8,9,10]
                elif data_name == "citeulike":
                    batch_size = 256
                    num_user = 5551
                    num_item = 16980
                    lr = 3e-3
                    # NSs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                if data_name == "citeulike":
                    data_reader = DataReader("./data/%s/0.2_%d/" % (data_name, fold), mode,
                                             explicit_feedback=explicit_feedback_,
                                             explicit_threshold=explicit_threshold_, is_citeulike=True)
                else:
                    data_reader = DataReader("./data/%s/0.2_%d/" % (data_name, fold), mode,
                                             explicit_feedback=explicit_feedback_,
                                             explicit_threshold=explicit_threshold_)

                for NS in NSs:
                    print("NVCF Fold:%d\tdata:%s\tNS:%d" % (fold, data_name, NS))
                    logging.info("NVCF Fold:%d\tdata:%s\tNS:%d\n" % (fold, data_name, NS))
                    tf.reset_default_graph()
                    with tf.Session() as sess:
                        tf.set_random_seed(53)
                        run = NeuralVariationalCollabFiltering(sess, data_reader, NS, activation_fn, mode, k_list, loss_fn,
                                                               batch_size=batch_size, decay_rate=0.96, decay_epoch_step=200,
                                                               embed_dim=embed_dim, latent_dim=latent_dim, learning_rate=lr,
                                                               batch_normalization=True, max_epochs=100, max_iter=99999,
                                                               densify=False, user_side_info=False, item_side_info=False,
                                                               alternate_opt=False, printing=printing)
                        max_R, last_R = run.train()

                        print("Latent_dim\t%d\tFold\t%d\tdata\t%s\tNS\t%d\tMax_R@300\t%.4f\tLast_R@300\t%.4f" % (
                        latent_dim, fold, data_name, NS, max_R, last_R))
                        logging.info("Latent_dim\t%d\tFold\t%d\tdata\t%s\tNS\t%d\tMax_R@300\t%.4f\tLast_R@300\t%.4f" % (
                        latent_dim, fold, data_name, NS, max_R, last_R))
