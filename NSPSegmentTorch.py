import torch
import pickle
import tqdm
import random
from transformers import BertTokenizer
import logging
import threading
logging.getLogger("transformers").setLevel(logging.ERROR)

def NSPSeg(data):
    Seg = [[],[]]
    for i, d in enumerate(data):
        print(f"{i}")
        for sents in tqdm.tqdm(d):
            # r = first segment length
            if(i == 1):
                if random.uniform(0,1) >0.1:
                    r = random.randint(1, max(1, len(sents) - 1))
                else :
                    if random.uniform(0, 1) < 0.5:
                        r = 0
                    else:
                        r = 1
            else:
                r = random.randint(1, max(1, len(sents) - 1))

            s1 = ''
            s2 = ''
            for j in range(r):
                s1 += ' ' + sents[j]
            for j in range(r, len(sents)):
                s2 += ' ' + sents[j]

            Seg[i].append(tuple((s1, s2)))
        # end for seg
    # end for whole data
    return Seg

def dict_shuffle(dic):
    keys = list(dic.keys())
    l = len(dic[keys[0]])
    for i in range(l):
        r = random.randint(0, l - 1)
        for k in keys:
            temp = dic[k][r].detach().clone()
            dic[k][r] = dic[k][i]
            dic[k][i] = temp
    return dic

def toTokenData(data, label, append_list, append_list_lock = None, tokeniser = BertTokenizer.from_pretrained("bert-base-uncased"), contxt_window = 128,):
    NSPTokens = []
    # Create tokens
    for leb, d in enumerate(data):
        step = 1000
        for i in tqdm.tqdm(range(0,len(d),step)):
            tokens = tokeniser(d[i : min(i + step, len(d))],return_tensors='pt', max_length=contxt_window,
                               padding='max_length', truncation='longest_first',
                               return_overflowing_tokens=False)
            tokens["labels"] = torch.zeros(len(tokens[list(tokens.keys())[0]]), dtype=torch.float32) + label[leb]
            NSPTokens.append(tokens)
    if append_list_lock == None:
        append_list.append(merge_dict(NSPTokens, False))
    else :
        with append_list_lock:
            append_list.append(merge_dict(NSPTokens, False))
    print("done")


def merge_dict(dic_List, shuffle = True):
    # Concatinate the data and create full data
    keys = list(dic_List[0].keys())
    dic = {}
    for k in keys:
        dic[k] = []
    for i, d in enumerate(dic_List):
        keys = list(d.keys())
        for k in keys:
            dic[k].append(d[k])
    keys = list(dic.keys())
    for k in keys:
        dic[k] = torch.cat(dic[k], dim=0)

    if not shuffle:
        return dic

    return dict_shuffle(dic)

if __name__ == "__main__":
    with open("NSP.pkl", 'rb') as f:
        data = pickle.load(f)
    num_sets = 5
    num_threads = 12
    lables = data[1]
    data = NSPSeg(data[0])
    temp = max(len(data[0]), len(data[1]))
    div_lis = [i for i in range(0, temp, temp // num_sets)]
    div_lis.append(temp)
    tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")

    for i in range(num_sets):
        append_list = []
        append_list_lock = threading.Lock()
        set = [data[0][div_lis[i] : min(len(data[0]), div_lis[i + 1])],
               data[1][div_lis[i] : min(len(data[1]), div_lis[i + 1])]]
        temp = max(len(set[0]), len(set[1]))
        set_div_lis = [i for i in range(0, temp, temp // num_threads)]
        set_div_lis.append(temp)

        threads = []
        for j in range(num_threads):
            d = [set[0][set_div_lis[j] : min(len(set[0]), set_div_lis[j + 1])],
                 set[1][set_div_lis[j] : min(len(set[1]), set_div_lis[j + 1])]]
            threads.append(threading.Thread(target=toTokenData, args=(d, lables, append_list, append_list_lock, tokeniser)))

        for j in range(num_threads):
            threads[j].start()

        for j in range(num_threads):
            threads[j].join()
        save_data = merge_dict(append_list,False)
        with open(f"TokenNSPDataSet{i}.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Set {i + 1} done.")
        break
    print(save_data)