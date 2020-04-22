import numpy as np
from utils import *
import time
from os.path import exists

class DataReader:
    def __init__(self, path, mode, explicit_feedback=False, explicit_threshold=1.0, nvcf_form=False, normalize=False, is_citeulike=False):
        # path = "./data/m1-1m/0.2/"

        self.user_ids_tr, self.item_ids_tr, self.ratings_tr,\
        self.lda_formed_data_ids_tr, self.lda_formed_data_vals_tr,\
        self.num_users_tr, self.num_items_tr = self.read_ratings(path + "train_%s.dat" % mode, explicit_feedback, explicit_threshold,is_citeulike)

        if is_citeulike:
            self.user_ids_val, self.item_ids_val, self.ratings_val, \
            self.lda_formed_data_ids_val, self.lda_formed_data_vals_val, \
            self.num_users_val, self.num_items_val = self.read_ratings(path + "test_%s.dat" % mode, explicit_feedback,
                                                                       explicit_threshold, is_citeulike)
        else:
            self.user_ids_val, self.item_ids_val, self.ratings_val,\
            self.lda_formed_data_ids_val, self.lda_formed_data_vals_val,\
            self.num_users_val, self.num_items_val = self.read_ratings(path + "valid_%s.dat" % mode, explicit_feedback,explicit_threshold,is_citeulike)

        self.user_ids_te, self.item_ids_te, self.ratings_te,\
        self.lda_formed_data_ids_te, self.lda_formed_data_vals_te,\
        self.num_users_te, self.num_items_te = self.read_ratings(path + "test_%s.dat" % mode, explicit_feedback, explicit_threshold,is_citeulike)

        if not nvcf_form:
            self.doc_ids, self.doc_cnt, self.vocab_size, self.vocab = self.load_item_texts(path)
        else:
            self.doc_ids, self.doc_cnt, self.vocab_size, self.vocab, self.x, self.x_idx, self.doc_lengths =\
                self.load_item_texts(path, nvcf_form, normalize, is_citeulike)


    #
    # def save_in_cdl_format(self, path):
    #     # TODO!!!
    #
    #     #
    #     np.savetxt(path + "mult_nor.dat", self.x.T, fmt="%.4f", delimiter="\t")
    #
    #     with open(path + "train_user.dat", "w") as f:
    #         for i in range(len(self.lda_formed_data_ids_tr)):
    #
    #
    #     with open(path + "train_item.dat", "w") as f:
    #
    #
    #     with open(path + "test_user.dat", "w") as f:
    #
    #     with open(path + "test_item.dat", "w") as f:




    def get_item_texts(self, nvcf_form=False):
        if not nvcf_form:
            return self.doc_ids, self.doc_cnt, self.vocab_size, self.vocab
        else:
            return self.doc_ids, self.doc_cnt, self.vocab_size, self.vocab, self.x, self.x_idx, self.doc_lengths




    def load_item_texts(self, path, nvcf_form=False, normalize=False, is_citeulike=False):
        print("Start reading item texts...")

        vocab = list()
        with open(path + "vocab.dat", "r") as f:
            lines = f.readlines()
            for line in lines:
                vocab.append(line)

        vocab_size = len(vocab)


        with open(path + "mult.dat", "r") as f:
            lines = f.readlines()
            # vocab_size = int(lines[0])

            doc_ids = list()
            doc_cnt = list()

            # I'll follow the convention of nvdm.py
            """
            vocabs = ['foo', 'boo', 'bar', 'too', 'one']
            if data = ['foo', 'foo', 'foo', 'bar', 'foo', 'bar', 'boo', 'too']
            x = [4, 1, 2, 1, 0] : # of occurrences of each vocab, bag-of-words representation
            x_idx = [0, 0, 0, 2, 0, 2, 1, 3] # index of each word in order (or not), len(x_idx) = length of doc (or sentence)
            """

            x = list() # BOW, # of occurrences of each vocab, len(x) = vocab_size
            x_idx = list() # index of each word in order (or not), len(x_idx) = length of doc (or sentence)
            doc_lengths = list()



            for line in lines:
                max_freq = 0
                elems = line.split(" ")
                num_words = int(elems[0])
                # print(num_words)
                word_ids = np.zeros(num_words, dtype=int)
                word_cnts = np.zeros(num_words, dtype=int)

                tmp_x = np.zeros(vocab_size, dtype=float)
                if is_citeulike:
                    tmp_x_idx = np.zeros(4000, dtype=int)
                else:
                    tmp_x_idx = np.zeros(300, dtype=int)

                doc_length = 0
                for i in range(num_words):
                    elem = elems[1:][i]

                    word_id, word_cnt = elem.split(":")
                    word_ids[i] = int(word_id)
                    word_cnts[i] = int(word_cnt)

                    if max_freq < int(word_cnt):
                        max_freq = int(word_cnt)

                    if nvcf_form:
                        tmp_x[word_ids[i]] = word_cnts[i]
                        for cnt in range(word_cnts[i]):
                            tmp_x_idx[doc_length] = word_ids[i]
                            doc_length += 1
                            # print(doc_length)

                if normalize:
                    tmp_x /= float(max_freq)


                doc_lengths.append(doc_length)

                doc_ids.append(word_ids)
                doc_cnt.append(word_cnts)

                x.append(tmp_x)
                x_idx.append(tmp_x_idx)


            doc_lengths = np.asarray(doc_lengths, dtype=int)
            doc_ids = np.asarray(doc_ids)
            doc_cnt = np.asarray(doc_cnt)
            x = np.asarray(x, dtype=float)
            x_idx = np.asarray(x_idx, dtype=int)

        # out_path = path + "mult_CDL.dat"
        # if not exists(out_path):
        #     np.savetxt(out_path, x, fmt="%d", delimiter="\t")

        if not nvcf_form:
            return doc_ids, doc_cnt, vocab_size, vocab
        else:
            return doc_ids, doc_cnt, vocab_size, vocab, x, x_idx, doc_lengths


    def get_data(self, name):
        if name == "train":
            return self.num_users_tr, self.num_items_tr, self.user_ids_tr, self.item_ids_tr, self.ratings_tr,\
                    self.lda_formed_data_ids_tr, self.lda_formed_data_vals_tr
        elif name == "valid":
            return self.num_users_val, self.num_items_val, self.user_ids_val, self.item_ids_val, self.ratings_val, \
                    self.lda_formed_data_ids_val, self.lda_formed_data_vals_val
        elif name == "test":
            return self.num_users_te, self.num_items_te, self.user_ids_te, self.item_ids_te, self.ratings_te, \
                    self.lda_formed_data_ids_te, self.lda_formed_data_vals_te
    # def load_data(self, path):
    #
    #
    #
    # def save_data(self, path):
    #     save_npy(path + "")

    def read_ratings(self, file_path, explicit_feedback=True, explicit_threshold=4.0, is_citeulike=False):
        start_time = time.time()
        print("Start reading ratings...")
        user_ids = list()
        item_ids = list()
        ratings = list()

        lda_formed_data_ids = list()
        lda_formed_data_vals = list()
        test_user_ids = list() # for evaluation purpose

        with open(file_path, "r") as f:
            num_users = 0
            for line in  f.readlines():
                tokens = line.split(' ')
                num_records = int(tokens[0])

                user_item_ids = np.zeros(num_records, dtype=int)
                user_ratings = np.zeros(num_records, dtype=float)

                for j in range(num_records):
                    token = tokens[1:][j]
                    if is_citeulike:
                        item_id = int(token)
                        rating = 1.0
                    else:
                        id_and_rating = token.split(':')
                        item_id = id_and_rating[0]
                        rating = float(id_and_rating[1])
                        if not explicit_feedback:
                            if rating >= explicit_threshold:
                                rating = 1.0
                            else:
                                rating = 0.0

                    user_item_ids[j] = item_id
                    user_ratings[j] = rating

                    user_ids.append(num_users)
                    item_ids.append(item_id)
                    ratings.append(rating)

                lda_formed_data_ids.append(user_item_ids)
                lda_formed_data_vals.append(user_ratings)
                test_user_ids.append(num_users)
                num_users += 1

        user_ids = np.asarray(user_ids, dtype=int)
        item_ids = np.asarray(item_ids, dtype=int)
        ratings = np.asarray(ratings, dtype=float)
        lda_formed_data_ids = np.asarray(lda_formed_data_ids)
        lda_formed_data_vals = np.asarray(lda_formed_data_vals)

        num_items = len(np.unique(item_ids))

        out_path = file_path.replace(".dat", ".txt").replace("_user", "")
        if not exists(out_path):
            with open(out_path, "w") as f:
                for i in range(len(user_ids)):
                    if i == 0:
                        f.write("%d::%d::%.1f" % (user_ids[i], item_ids[i], ratings[i]))
                    else:
                        f.write("\n%d::%d::%.1f" % (user_ids[i], item_ids[i], ratings[i]))

        print("End reading ratings: %.5f sec elapsed" % (time.time() - start_time))
        return user_ids, item_ids, ratings, lda_formed_data_ids, lda_formed_data_vals, num_users, num_items

