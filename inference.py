import nltk
import pickle
import tensorlayer as tl


def inference(query, top_n):
    model.eval()
    idx_encoded_query = nltk.word_tokenize(query)
    idx_encoded_query = convert_to_idx([idx_encoded_query])
    idx_encoded_answer = model(
        inputs=[idx_encoded_query], 
        seq_length=seq_length, 
        start_token=start_id, 
        top_n=top_n)
    return convert_to_word(idx_encoded_answer)

def convert_to_idx(X):
        for i in range(len(X)):
            sent_len = len(X[i])
            for j in range(sent_len):
                X[i][j] = word2idx.get(X[i][j], unk_id)
            X[i] += [pad_id] * ((seq_len - 1) - sent_len) # leaving 1 index for start
        return X

def convert_to_word(X):
    y = []
    for i in range(len(X)):
        y_temp = []
        sent_len = len(X[i])
        for j in range(sent_len):
            word_idx = X[i][j]
            if word_idx == start_id:
                continue
            elif word_idx == end_id or word_idx == pad_id:
                y.append(y_temp)
                break
            y_temp.append(idx2word[word_idx.numpy()])
    return y

if __name__ == '__main__':
    try:
        model = tl.files.load_hdf5_to_weights('seq2seq.h5', network.all_weights)
        with open("metadata.pickle", 'r') as f:
            metadata = pickle.load(f)
        start_id = metadata["start_id"]
        end_id = metadata["end_id"]
        pad_id = metadata["pad_id"]
        unk_id = metadata["unknown_id"]
        word2idx = metadata["word_to_index"]
        idx2word = metadata["index_to_word"]
        seq_len = metadata["max_seq_len"]

    except Exception as e:
        print("Either the model is not trained or the metadata file is not found\n", e)
        exit()
    nltk.download('punkt')
    print("Type 'bye' to stop running the bot...\n")
    while True:
        query = input("> ")
        print("> ", query)
        top_n = 3
        if query == "bye":
            print(">> ", "bye!")
            return
        for i in range(top_n):
            sentences = inference(query, top_n)
            for sentence in sentences:
                print(">> ", ' '.join(sentence))
