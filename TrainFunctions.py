import torch
import torch.nn.functional as F
import random
import pickle
import tqdm

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
def Mask(inp:torch.tensor, mask_id:int, vocab_size : int,change_prob = 0.2)->torch.tensor :
    ret = inp.clone()
    prob_mat = torch.rand(inp.shape, device=inp.device)
    un_changed = prob_mat < change_prob * 0.1
    x = prob_mat < (change_prob * 0.2)
    random = (~un_changed)  & x
    y = prob_mat < change_prob
    mask = ~x & y
    ret[mask] = mask_id
    num_random = int(random.sum().item())
    ret[random] = torch.randint(0, vocab_size, (num_random,), device=inp.device)
    return ret, y

def one_hot_tokens(inp:torch.tensor, vocab_size, dtype = torch.float32):
    return F.one_hot(inp, num_classes= vocab_size).to(dtype = dtype, device= inp.device)

def dict_shuffle(dic):
    keys = list(dic.keys())
    l = len(dic[keys[0]])
    for i in tqdm.tqdm(range(l)):
        r = random.randint(0, l - 1)
        for k in keys:
            temp = dic[k][r].detach().clone()
            dic[k][r] = dic[k][i]
            dic[k][i] = temp
    return dic

def Load_Data():
    data = []
    for i in range(5):
        with open(f"TokenNSPDataSet{i}.pkl", 'rb') as f:
            d = pickle.load(f)
        data.append(d)
    return data

def cvt_dict_to_TensorDataset(dictonary, batch_size = 20, lebels = None, shuffle = True):
    if shuffle:
        dictonary = dict_shuffle(dictonary)
    if lebels == None:
        lebels = list(dictonary.keys())
    lis = [dictonary[k] for k in lebels]
    dataset = torch.utils.data.TensorDataset(*lis)
    data_loder = torch.utils.data.DataLoader(dataset=dataset, batch_size= batch_size,shuffle=shuffle)
    return data_loder

def test(model, dl, mask_id, vocab_size, criterion_mlm = None, criterion_nsp = None,p = 0.2, device = None, enable_bf16 = False) :
    if device == None:
        device = next(model.parameters()).device
    else:
        model = model.to(device)
    model = model.eval()
    sum_loss = 0
    sum_accu = 0
    sum_masked_accu = 0
    sum_NSP_accu = 0
    masked_sum = 0.00001
    num_tokens = 0.00001
    batch_no = 0.00001
    num_sents = 0.00001
    pbar = tqdm.tqdm(dl, leave= True)
    for token_id, seg,attain_mask, nsp_leb in pbar:
        token_id = token_id.to(device)
        seg = seg.to(device)
        nsp_leb = nsp_leb.to(device)
        attain_mask = attain_mask.to(device= device)
        attain_mask = (attain_mask == 0)
        id_masked, masked = Mask(token_id,mask_id,vocab_size, p)
        with torch.amp.autocast("cuda", enabled=enable_bf16, dtype=torch.bfloat16):
            with torch.no_grad():
                out_mlm, out_nsp = model.forward(id_masked, seg, attain_mask)
                if criterion_mlm != None:
                    sum_loss += float(criterion_mlm(out_mlm.view(-1, vocab_size), token_id.view(-1)))
                if criterion_nsp != None:
                    sum_loss += float(criterion_nsp(out_nsp.view(-1), nsp_leb))
        sum_accu += float(torch.sum(torch.argmax(out_mlm,dim=2) == token_id))
        sum_masked_accu += float(torch.sum((torch.argmax(out_mlm,dim=2) == token_id) & masked))
        masked_sum += float(torch.sum(masked))
        num_tokens += float(token_id.numel())
        num_sents += len(token_id)
        batch_no += 1
        sum_NSP_accu += float(torch.sum((out_nsp > 0.5) == (nsp_leb > 0.5)))
        x = (sum_loss / batch_no, sum_accu / num_tokens, sum_masked_accu / masked_sum, sum_NSP_accu / num_sents)
        pbar.set_postfix(
            {"Train Loss ": f'{x[0]:4f}', 'accu': f'{x[1]:4f}', 'masked_accu': f'{x[2]:4f}', 'NSP_accu': f'{x[3]:4f}'})
    return (sum_loss / len(dl), sum_accu / num_tokens, sum_masked_accu / masked_sum, sum_NSP_accu / len(dl.dataset))

def train(model, dl_lis, mask_id, vocab_size, criterion_mlm, ignore_idx, criterion_nsp,
          optimizer, seheduler = None,epochs = 1, enable_bf16 = False,
          test_dl = None,clip_grad = 0.5, p = 0.2,device = None, checkpoint_epoch = -1, checkpoint_train_dl = -1) :
    if device == None:
        device = next(model.parameters()).device
    else:
        model = model.to(device)
    prev_loss = 1000
    prev_accu = 0
    prev_masked_accu = 0
    prev_NSP_accu = 0
    loss_list = []
    iter = 1
    scaler = torch.amp.GradScaler('cuda', enabled=enable_bf16)
    for epoch in range(1, epochs + 1):
        print("Epoch: {}".format(epoch))
        if(epoch < checkpoint_epoch):
            print("Skipping")
            continue
        print("Training :")
        l = []
        for num_dl, dl in enumerate(dl_lis):
            print(f"Data Loader No.{num_dl}")
            if epoch <= checkpoint_epoch and num_dl <= checkpoint_train_dl:
                print("Skipping Data Loader")
                continue

            model = model.train()
            sum_loss = 0
            sum_accu = 0
            sum_masked_accu = 0
            sum_NSP_accu = 0
            masked_sum = 0.00001
            num_tokens = 0.00001
            batch_no = 0.00001
            num_sents = 0.00001
            pbar = tqdm.tqdm(dl, leave=True)
            for token_id, seg, attain_mask, cls_leb in pbar:
                # Zero Grad the optimizer
                optimizer.zero_grad()

                # To device and necessary changes
                token_id = token_id.to(device)
                seg = seg.to(device)
                cls_leb = cls_leb.to(device)
                attain_mask = attain_mask.to(device)
                attain_mask = (attain_mask == 0)
                id_masked, masked = Mask(token_id,mask_id,vocab_size, p)

                token_id_cp = token_id.clone()
                token_id_cp[~masked] = ignore_idx
                num_masked = int(torch.sum(masked).item())

                #forward
                with torch.amp.autocast("cuda", enabled=enable_bf16, dtype=torch.bfloat16):
                    out_mlm, out_nsp = model.forward(id_masked, seg, attain_mask)
                    # Loss calculation
                    loss_mlm = criterion_mlm(out_mlm.reshape(-1, vocab_size), token_id_cp.reshape(-1)) if num_masked > 0 else torch.zeros((), dtype=out_mlm.dtype, device=out_mlm.device)
                    loss_nsp = criterion_nsp(out_nsp, cls_leb)
                    loss = loss_mlm + loss_nsp
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                current_lr = optimizer.param_groups[0]["lr"]
                if seheduler != None:
                    seheduler.step()

                # Update the accuracy values
                sum_loss += float(loss)
                sum_accu += float(torch.sum(torch.argmax(out_mlm,dim=2) == token_id))
                sum_masked_accu += float(torch.sum((torch.argmax(out_mlm,dim=2) == token_id) & masked))
                sum_NSP_accu += float(torch.sum((out_nsp.view(-1) > 0.5) == (cls_leb > 0.5)))
                masked_sum += float(num_masked)
                num_tokens += float(token_id.numel())
                iter += 1
                num_sents += len(token_id)
                batch_no += 1
                x = (sum_loss / batch_no, sum_accu / num_tokens, sum_masked_accu /masked_sum, sum_NSP_accu / num_sents)
                pbar.set_postfix({"Train Loss ": f'{x[0]:4f}', 'accu' : f'{x[1]:4f}', 'masked_accu' : f'{x[2]:4f}', 'NSP_accu' : f'{x[3]:4f}', 'lr' : f'{current_lr:1.3}'})
            torch.cuda.empty_cache()
            x = (sum_loss / len(dl), sum_accu / num_tokens, sum_masked_accu /masked_sum, sum_NSP_accu / len(dl.dataset))
            print("Train Loss, accu, masked_accu, NSP_accu: ", x)
            l.append(x)
        optimizer.zero_grad()
        if test_dl != None:
            print("Testing :")
            y = test(model, test_dl, mask_id, vocab_size, criterion_mlm, criterion_nsp, p, device)
            loss_list.append([l, y])
            if(y[0] < prev_loss):
                print(f'Loss Model Saved at epoch :{epoch}')
                torch.save({'state_dict' : model.state_dict(), 'cfgDict' : model.cfgDict}, "Model_loss.pth")
                prev_loss = y[0]
            if(y[1] > prev_accu):
                print(f'Accu Model Saved at epoch :{epoch}')
                torch.save({'state_dict' : model.state_dict(), 'cfgDict' : model.cfgDict}, "Model_accu.pth")
                prev_accu = y[1]
            if(y[2] > prev_masked_accu):
                print(f'Masked Accu Model Saved at epoch :{epoch}')
                torch.save({'state_dict' : model.state_dict(), 'cfgDict' : model.cfgDict}, "Model_masked_accu.pth")
                prev_masked_accu = y[2]
            if(y[3] > prev_NSP_accu):
                print(f'NSP Accu Model Saved at epoch :{epoch}')
                torch.save({'state_dict' : model.state_dict(), 'cfgDict' : model.cfgDict}, "Model_nsp_accu.pth")
                prev_NSP_accu = y[3]
            print("Test Loss, accu, masked_accu, NSP_accu: ", y)
        else:
            loss_list.append([l])
        print(f"Model Saved")
        torch.save({'state_dict': model.state_dict(), 'cfgDict': model.cfgDict}, "Model.pth")
    return loss_list, model




