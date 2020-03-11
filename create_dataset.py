import re
from datetime import datetime
import multiprocessing
import os


# max_seq_len = 150

def create_timestamp(timestamp):
    h = int(timestamp[:2])
    m = int(timestamp[3:5])
    s = int(timestamp[6:8])
    return datetime(1970, 1, 1, h, m, s)

def extract_qa(file):
    time_pat = re.compile(r'\d\d:\d\d:\d\d,\d\d\d')
    questions = ""
    answers = ""
    sentence = ""
    len_count = 0
    ques_only = True
    prev_time = datetime(1970, 1, 1)

    def save_sent(sent):
        nonlocal questions, answers, len_count, ques_only
        len_count = len(questions)
        # decode the sent
        #print(len_count)
        if ques_only == False:
            answers += sent.strip() + '\n'
        else:
            ques_only = False
        questions += sent.strip() + '\n'

    def remove_sent(a):
        nonlocal questions
        #print(questions[a:])
        questions = questions[:a]

    # open the file
    # print("Processing", file)
    with open(file, 'r') as f:
        while f.readline():
            line = f.readline()
            timestamp = time_pat.findall(line)
            try:
                cur_time = create_timestamp(timestamp[0])
            except:
                print("Skipped this line:", line)
                continue

            line = f.readline()
            if line[0].isupper() and sentence != "":
                save_sent(sentence)
                sentence = ""
            # if (cur_time - prev_time).total_seconds() > 2.0:
            #     remove_sent(len_count)
            #     ques_only = True
            while line != '\n' and len(line) > 0:
                if line[0] == '-':
                    if line[2].islower():
                        line = line.replace('- ', '')
                    else:
                        if sentence != "":
                            save_sent(sentence)
                            sentence = ""
                        line = line.replace('- ', '')
                elif line[0] == '<' or line[-1] == '>':
                    line = f.readline()
                    continue
                sentence += line.replace('\n', ' ')
                line = f.readline()

            prev_time = create_timestamp(timestamp[1])
    remove_sent(len_count)
    return questions, answers

# def create_dataset():
#     for folder in os.listdir("subs"):
#         try:
#             for file in os.listdir('/'.join(["subs", folder])):
#                 print('/'.join(["subs", folder, file]))
#                 a, b = extract_qa('/'.join(["subs", folder, file]))

#                 with open("data_in.txt", 'a') as f:
#                     f.write(a)
#                 with open("data_out.txt", 'a') as f:
#                     f.write(b)
#         except Exception as e:
#             pass

import nltk
import numpy as np

# instead of tokenizing afterwards, do it while saving
# pad the sequences

def create_tokens(qa):
    a = qa[0].split('\n')[:-1]
    b = qa[1].split('\n')[:-1]
    tk_a, tk_b = [], []
    for line in a:
        tk_a.append(nltk.word_tokenize(line))
    for line in b:
        tk_b.append(nltk.word_tokenize(line))
    return tk_a, tk_b

def preprocess_subs():
    p = multiprocessing.Pool()
    subs_files = []
    for folder in os.listdir("subs"):
        try:
            for file in os.listdir('/'.join(["subs", folder])):
                subs_files.append('/'.join(["subs", folder, file]))
        except:
            pass
    data = p.map(extract_qa, subs_files)
    print("Parsed through the subs")
    tk_data = p.map(create_tokens, data)
    p.close()
    tk_ques, tk_ans = [], []
    for tk_a, tk_b in tk_data:
        for tk_line in tk_a:
            tk_ques.append(tk_line)
        for tk_line in tk_b:
            tk_ans.append(tk_line)

    # max_ques_len = 0
    # for line in tk_ques:
    #     len_of_line = len(line)
    #     if len_of_line > max_ques_len:
    #         max_ques_len = len_of_line

    # max_ans_len = 0
    # for line in tk_ans:
    #     len_of_line = len(line)
    #     if len_of_line > max_ans_len:
    #         max_ans_len = len_of_line

    # for i in range(len(tk_ques)):
    #     tk_ques[i] += [pad_tk] * (max_ques_len - len(tk_ques[i]))

    # for i in range(len(tk_ans)):
    #     tk_ans[i] += [pad_tk] * (max_ans_len - len(tk_ans[i]))

    tk_ques, tk_ans = np.array(tk_ques), np.array(tk_ans)
    print("Tokenized the ques/ans", len(tk_ques), len(tk_ans))
    print(tk_ques[:5])
    np.save("tokenized_questions.npy", tk_ques, allow_pickle=True)
    np.save("tokenized_answers.npy", tk_ans, allow_pickle=True)

if __name__ == "__main__":
    nltk.download('punkt')
    preprocess_subs()
