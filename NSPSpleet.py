import numpy as np
import pickle
import random
import tqdm
import re
import bisect

def Shuffle(Lis):
    print("Shuffeling :")
    l = len(Lis) - 1
    for i in tqdm.tqdm(range(len(Lis))):
        j = random.randint(0, l)
        temp = Lis[i]
        Lis[i] = Lis[j]
        Lis[j] = temp
    return Lis



def NSP_Split(data, cont_win_len = 128, avg_word_len = 5.5) :
    NSP_True = []
    max_len = cont_win_len // 12
    NSP_False = [[], []]
    temp = 0.6 / (max_len - 2)
    arr = np.arange(0, 0.4, temp) + temp
    print("Creating split:")
    for sents in tqdm.tqdm(data):
        i = random.randint(0,1)
        l = random.randint(cont_win_len - 20, cont_win_len + 10) * avg_word_len
        cont_sent = []
        r = random.uniform(0,1)
        for sent in sents:
            sent = re.sub(r'https?://\S+', '[Removed Link]', sent)
            sent = re.sub(r'www\.\S+', '[Removed Link]', sent)
            sent = re.sub(r'(\D)\1{2,}', '', sent)
            sent = re.sub(r'#', '', sent)
            l -= len(sent)
            cont_sent.append(sent)
            if l <= 0:
                if i == 0:
                    i = 1
                    for x, sent in enumerate(cont_sent):
                        NSP_False[x % 2].append(sent)
                else:
                    NSP_True.append(tuple(cont_sent))
                    i = 0

                cont_sent = list()
                l = random.randint(cont_win_len - 20, cont_win_len + 10) * avg_word_len
                r = random.uniform(0,1)
        if len(cont_sent) >= 2:
            if i == 0:
                for x, sent in enumerate(cont_sent):
                    NSP_False[x % 2].append(sent)
                i = 1
            else:
                NSP_True.append(tuple(cont_sent))
                i = 0
    print("False 0 ", end = '')
    Shuffle(NSP_False[0])
    print("False 1 ", end = '')
    Shuffle(NSP_False[1])
    lis = []
    for FF , sents in enumerate(NSP_False):
        print(f"Creating False: {FF}")
        l = random.randint(cont_win_len - 20, cont_win_len + 10) * avg_word_len
        discont_sent = []
        r = random.uniform(0,1)
        for sent in tqdm.tqdm(sents):
            l -= len(sent)
            discont_sent.append(sent)
            if l < 0:
                lis.append(tuple(discont_sent))
                discont_sent = []
                l = random.randint(cont_win_len - 20, cont_win_len + 10) * avg_word_len
                r = random.uniform(0,1)
        if len(discont_sent) >= 2:
            lis.append(tuple(discont_sent))
    NSP_False = Shuffle(lis)
    NSP_True = Shuffle(NSP_True)
    return [[NSP_False, NSP_True], [0, 1]]

if __name__ == "__main__" :
    with open("sentence_corpa.pkl", 'rb') as f:
        data = pickle.load(f)
    x = NSP_Split(data)
    print(len(x))
    print(len(x[0]))
    print(len(x[1]))
    print(len(x[0][0]))
    print(len(x[0][1]))
    del data
    print("Store?[Y/n]  ", end='')
    inp = input()
    if inp.lower()[0] != 'n':
        with open("NSP.pkl", "wb") as f:
            pickle.dump(x, f)

