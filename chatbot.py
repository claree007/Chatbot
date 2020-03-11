import tensorflow as tf
import tensorlayer as tl
import numpy as np
from sklearn.model_selection import train_test_split
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from tensorlayer.models.seq2seq import Seq2seq
from tensorlayer.models.seq2seq_with_attention import Seq2seqLuongAttention
import nltk
import re
import time
import random
import pickle


class Seq2SeqChatBot:
    def __init__(self, dataset):
        self.unique_tokens = list(set(token for sentence in dataset for token in sentence))
        self.unique_tokens.append("<start>")
        self.unique_tokens.append("<end>")
        self.unique_tokens.append("<pad>")
        self.unique_tokens.append("<unknown>")
        self.word2idx = {token:i for i, token in enumerate(self.unique_tokens)}
        self.idx2word = {i:token for i, token in enumerate(self.unique_tokens)}
        self.vocab_size = len(self.unique_tokens)
        self.start_id = self.word2idx["<start>"]
        self.end_id = self.word2idx["<end>"]
        self.pad_id = self.word2idx["<pad>"]
        self.unk_id = self.word2idx["<unknown>"]
        self.max_seq_len = np.max([len(line) for line in dataset]) + 1 # for <start> or <end>
        self.create_model()

    def save_metadata(self):
        metadata = {"index_to_word": self.idx2word, 
                    "word_to_index": self.word2idx,
                    "start_id": self.start_id, 
                    "end_id": self.end_id, 
                    "pad_id": self.pad_id, 
                    "unknown_id": self.unk_id, 
                    "max_seq_len": self.max_seq_len}

        with open("metadata.pickle", 'w') as f:
            pickle.dump(metadata, f)

    def create_model(self):
        self.seq_length = self.max_seq_len
        self.emb_dim = 1024
        self.lr = 0.001

        self.model_ = Seq2seq(
            decoder_seq_length = self.seq_length,
            cell_enc=tf.keras.layers.GRUCell,
            cell_dec=tf.keras.layers.GRUCell,
            n_layer=3,
            n_units=256,
            embedding_layer=tl.layers.Embedding(vocabulary_size=self.vocab_size, 
                                                                     embedding_size=self.emb_dim),
                                                                    )
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.model_.train()

    def convert_to_idx(self, X):
        for i in range(len(X)):
            sent_len = len(X[i])
            for j in range(sent_len):
                X[i][j] = self.word2idx.get(X[i][j], self.unk_id)
            X[i] += [self.pad_id] * ((self.max_seq_len - 1) - sent_len) # leaving 2 indices for start and end
        return X

    def convert_to_word(self, X):
        y = []
        for i in range(len(X)):
            y_temp = []
            sent_len = len(X[i])
            for j in range(sent_len):
                word_idx = X[i][j]
                #print(word_idx)
                if word_idx == self.start_id:
                    continue
                elif word_idx == self.end_id or word_idx == self.pad_id:
                    y.append(y_temp)
                    break
                y_temp.append(self.idx2word[word_idx.numpy()])
        return y

    # def inference(self, query, top_n):
    #     self.model_.eval()
    #     idx_encoded_query = nltk.word_tokenize(query)
    #     idx_encoded_query = self.convert_to_idx([idx_encoded_query])
    #     idx_encoded_answer = self.model_(
    #         inputs=[idx_encoded_query], 
    #         seq_length=self.seq_length, 
    #         start_token=self.start_id, 
    #         top_n=top_n)
    #     return self.convert_to_word(idx_encoded_answer)
    #     sentence = []
    #     for w_id in idx_encoded_answer[0]:
    #         w = self.idx2word[w_id]
    #         if w == "<end>":
    #             break
    #         sentence = sentence + [w]
    #     return sentence

    def train_model(self, X_train, y_train, validation=False, batch_size=64, num_epochs=4):
        training_loss = []
        validation_loss = []
        for epoch in range(num_epochs):
            self.model_.train()
            #trainX, trainY = shuffle(trainX, trainY, random_state=0)
            train_loss, n_iter, loss = 0, 0, None
            for X, Y in tqdm(tl.iterate.minibatches(inputs=X_train, targets=y_train, batch_size=batch_size, shuffle=False),
                             total=len(X_train)//batch_size, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs)):
                X = self.convert_to_idx(X)
                _target_seqs = []
                _decode_seqs = []
                for i in range(batch_size):
                    X[i] = [self.start_id] + X[i]
                    _target_seqs.append(Y[i] + ["<end>"])
                Y = self.convert_to_idx(Y)
                _target_seqs = self.convert_to_idx(_target_seqs)
                for i in range(batch_size):
                    if _target_seqs[i][-1] == self.pad_id:
                        _target_seqs[i] += [self.pad_id]
                    _decode_seqs.append([self.start_id] + Y[i])
                
                _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

                with tf.GradientTape() as tape:
                    ## compute outputs
                    output = self.model_(inputs = [X, _decode_seqs])
                    output = tf.reshape(output, [-1, self.vocab_size])
                    ## compute loss and update model
                    loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)
                    grad = tape.gradient(loss, self.model_.all_weights)
                    self.optimizer.apply_gradients(zip(grad, self.model_.all_weights))
                
                train_loss += loss
                n_iter += 1
            train_loss /= n_iter
            training_loss.append(train_loss)
            print('train_loss {:.4f}'.format(train_loss))

            if validation:
                val_loss, n_iter, loss = 0, 0, None
                for X, Y in tqdm(tl.iterate.minibatches(inputs=validation[0], targets=validation[1], batch_size=batch_size, shuffle=False), 
                            total=len(validation[0])//batch_size, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs)):
                    X = self.convert_to_idx(X)
                    _target_seqs = []
                    _decode_seqs = []
                    for i in range(batch_size):
                        X[i] = [self.start_id] + X[i]
                        _target_seqs.append(Y[i] + ["<end>"])
                    Y = self.convert_to_idx(Y)
                    _target_seqs = self.convert_to_idx(_target_seqs)
                    for i in range(batch_size):
                        if _target_seqs[i][-1] == self.pad_id:
                            _target_seqs[i] += [self.pad_id]
                        _decode_seqs.append([self.start_id] + Y[i])
                    
                    _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

                    output = self.model_(inputs = [X, _decode_seqs])
                    output = tf.reshape(output, [-1, self.vocab_size])
                    ## compute loss and update model
                    loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)

                    val_loss += loss
                    n_iter += 1
                val_loss /= n_iter
                validation_loss.append(val_loss)
                print('val_loss {:.4f}'.format(val_loss))

            ## save model as .h5
            d = time.localtime()
            tl.files.save_weights_to_hdf5("seq2seq_{}/{}/{}_{}:{}:{}.h5".format(d[0], d[1], d[2], d[3], d[4], d[5]), network.all_weights)
            
        if validation:
            return train_loss, val_loss
        return train_loss

def shuffle_dataset(X, y):
    X_shuffle = []
    y_shuffle = []
    shuffle_batch_size = 128
    X_len = len(X)
    shuffle_batches = X_len // shuffle_batch_size
    last_batch = X_len % shuffle_batch_size
    rem_batch_pos = [i for i in range(0, X_len, shuffle_batch_size)][:-1]
    for _ in range(shuffle_batches):
        idx = rem_batch_pos[random.randint(0, len(rem_batch_pos)-1)]
        for j in range(idx, idx + shuffle_batch_size):
            X_shuffle.append(X[j])
            y_shuffle.append(y[j])
        rem_batch_pos.remove(idx)
    for i in range(last_batch, X_len):
        X_shuffle.append(X[i])
        y_shuffle.append(y[i])
    print("The dataset has been shuffled.\n")
    return X_shuffle, y_shuffle


if __name__ == '__main__':
    try:
        # load numpy array of questions and answers
        X_full = np.load('tokenized_questions.npy', allow_pickle=True)
        y_full = np.load('tokenized_answers.npy', allow_pickle=True)
        print("Total questions:", len(X_full))
        print("Total answers:", len(y_full))
    except Exception as e:
        print("Either the dataset is not created or the files were not found\n", e)
        exit()

    # shuffle the dataset
    X_full, y_full = shuffle_dataset(X_full, y_full)

    # create train, test and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, shuffle=False)

    # if X_full does not contains all the words(tokens) that y_full
    # contains, then run join_qa() and pass the output to Seq2SeqChatBot
    bot = Seq2SeqChatBot(X_full)
    print("Started model training...")
    bot.train_model(X_train, y_train, validation=[X_test, y_test], batch_size=32, num_epochs=1)
    print("""The model has been saved, you can now test it by running inference.py with the correct
        model name and metadata.pickle file""")
