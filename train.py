import os
import sys
import torch
import torch.nn as nn
from torch import optim
import scipy.io as spio
from tqdm import tqdm
import scipy.io.wavfile as wavio
script_path = "./Libs/"
sys.path.append(os.path.abspath(script_path))
from Libs.feature_utils import *
import numpy as np
from collections import OrderedDict
from random import randrange

class DNN128(nn.Module):
    def __init__(self, input_length, hidden_length, cls_num):
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(input_length, hidden_length)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(hidden_length, hidden_length)),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(hidden_length, hidden_length)),
                ('relu3', nn.ReLU()),
                ('fc4', nn.Linear(hidden_length, cls_num)),
                # ('relu4', nn.ReLU()),
                # ('fc5', nn.Linear(hidden_length, cls_num)),
                # ('output', nn.Softmax(dim=1))
            ])
        )
    def forward(self, x):
        x = self.layers(x.float())
        return x

def next_batch(batchsize, xtrain, ytrain, all_data_idx):
    # idx = np.arange(0 , len(xtrain))
    np.random.shuffle(all_data_idx)
    idx = all_data_idx[:batchsize]
    data_shuffle = [xtrain[i] for i in idx]
    labels_shuffle = [ytrain[j] for j in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def get_second_shuffled_wav_list(batchSize, totalDataList):
#     np.random.shuffle(totalDataList)

def gen_labeled_footprint(data_record):
    wav_file = data_record[0]
    label = data_record[1]
    return wav_file, label

def get_random_file_list(sur_file_list):
    pass

def get_sub_data_list(batchSize, total_data_list, current_sublist_index):
    total_data_list_len = len(total_data_list)
    if current_sublist_index < total_data_list_len:
        start = current_sublist_index * batchSize
        end = start + batchSize
        sub_list = total_data_list[start:end]
        return sub_list
    return None

def get_training_data_gen(batchSize, total_data_list, current_sublist_index):
    sub_list = get_sub_data_list(batchSize, total_data_list, current_sublist_index)
    print("sub_list is {}".format(sub_list))
    data_list = (
        gen_labeled_footprint(r)
        for r in sub_list
    )
    return data_list


def get_val_data_footprint(rec):
    # aFB = get_filterbanks()
    f = rec[0].strip()
    l = int(float(rec[1].strip()))
    # sr, sig = wavio.read(f)
    tmp_x, tmp_y = gen_main_entry(f, l)
    return tmp_x, tmp_y

def get_val_data_generator(val_file_list):
    # tmp_val_data = val_file_list[:, 0]
    # tmp_val_label = val_file_list[:, 1]
    val_data_itor=(
        get_val_data_footprint(d)
        for d in tqdm(val_file_list)
    )
    return val_data_itor

def get_validation_traing_input(val_file_list):
    itor = get_val_data_generator(val_file_list)
    all_footprints = np.array([])
    all_labels = np.array([])
    for e in itor:
        all_footprints = np.vstack((all_footprints, e[0])) if all_footprints.size else e[0]
        all_labels = np.vstack((all_labels, e[1])) if all_labels.size else e[1]
    return all_footprints, all_labels

# def error_rate(predictions, labels):
#     predictions = np.argmax(predictions, 1)
#     return 100.0 - (
#       100.0 * np.sum( predictions == labels) / predictions.shape[0])

def calculat_acc(pred, labels):
    """
    ref: https://discuss.pytorch.org/t/how-does-one-get-the-predicted-classification-label-from-a-pytorch-model/91649/3
    """
    acc = (pred == labels).sum().item() / pred.size(0)
    return acc

def get_idx_from_range(theRange):
    selected_idx = randrange(theRange)
    return selected_idx

def get_list_max_ele_pos(list_to_find):
    return list_to_find.index(max(list_to_find))

def logit_to_label(tensor_to_convert):
    big_list = tensor_to_convert.tolist()
    retList = []
    for l in big_list:
        label_value = get_list_max_ele_pos(l)
        retList.append(label_value)
    return retList

def train():
    # train_data = spio.loadmat('../MyTrainData/kws_train_data_20200713_v4.mat')
    # train_lbl = spio.loadmat('../MyTrainData/kws_train_lbl_20200713_v4.mat')
    TRAINMAT = "../TrainingMetaData/kws_train_wav_file_list_20210514.mat"
    current_training_sublist_idx = 0
    total_data_list = spio.loadmat(TRAINMAT)['train_data']
    total_wav_file_num = len(total_data_list)
    split_idx = int(total_wav_file_num * 0.8)
    second_split_idx = int(total_wav_file_num * 0.9)
    train_data_list = total_data_list[0:split_idx]
    # print(train_data_list)
    # return
    total_train_wav_file_num = len(train_data_list)
    val_data_list = total_data_list[split_idx:second_split_idx]
    test_data_list = total_data_list[second_split_idx:total_wav_file_num]
    train_files_num = len(train_data_list)
    val_files_num = len(val_data_list)
    test_files_num = len(test_data_list)
    print("Total wav files:{}\n".format(total_train_wav_file_num))
    print("Total Training Files:{}\n".format(train_files_num))
    print("Total Validation Files:{}\n".format(val_files_num))
    print("Total Test Files:{}\n".format(test_files_num))
    print("Parparing Validation Data.......\n")
    print("train data list is {}/n".format(train_data_list))
    
    input_len = 1600
    hidden_len = 128
    output_classes = 3
    learning_rate = 0.00001
    outter_epoachs = 150
    inner_epoachs = 200
    # wav_feat = None
    model = DNN128(input_len, hidden_len, output_classes)
    # print(model)
    #define the loss
    criterion = nn.CrossEntropyLoss() #nn.NLLLoss()
    #define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,)
    melfb_ = get_filterbanks()
    print("Start to Training.......")
    model.train()
    for epoch in tqdm(range(outter_epoachs)):
        # train_loss, valid_loss = [], []
        train_loss, acc_list = [], []
        for step in range(inner_epoachs):
            #inner epochs中對訓練檔案的挑選要用亂數。
            if current_training_sublist_idx < total_train_wav_file_num-1:
                current_training_sublist_idx += 1
            else:
                current_training_sublist_idx = 0
            current_f = train_data_list[current_training_sublist_idx][0].strip()
            current_lbl = float(train_data_list[current_training_sublist_idx][1].strip())
            x_n = gen_train_lfbe_1600x1_with_cfft_power(current_f, melfb_)
            #performing training
            optimizer.zero_grad()
            #covert ndarray to tensor
            x_n_prime = torch.from_numpy(x_n)
            y_n_prime = torch.FloatTensor([current_lbl])
            ## 1. forward propagation
            logits = model(x_n_prime)
            logits = logits.view(1,3)
            # print(type(logits))
            # return
            max_indices = logits.max(1).indices
            this_acc = calculat_acc(max_indices, y_n_prime)
            acc_list.append(this_acc)
            ## 2. loss calculation
            # print("logits is {} | y_n_prime is {}".format(logits, y_n_prime))
            loss = criterion(logits, y_n_prime.long())
            ## 3. backward propagation
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            ## 4. weight optimization
            optimizer.step()
        train_loss.append(loss.item())
        
        print("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Training Accuracy: ", np.mean(acc_list))

    #ref: https://stackoverflow.com/questions/44734327/pytorch-extract-learned-weights-correctly
    result = model.parameters()
    w = list(result)
    w1 = w[0].data.numpy()
    b1 = w[1].data.numpy()
    w2 = w[2].data.numpy()
    b2 = w[3].data.numpy()
    w3 = w[4].data.numpy()
    b3 = w[5].data.numpy()
    w4 = w[6].data.numpy()
    b4 = w[7].data.numpy()
    # w5 = w[8].data.numpy()
    # b5 = w[9].data.numpy()
    trained_model_dict = {
        "w1": w1.T,
        "b1": b1.T,
        "w2": w2.T,
        "b2": b2.T,
        "w3": w3.T,
        "b3": b3.T,
        "w4": w4.T,
        "b4": b4.T
    }
    weights_save_path = "../kws_trained_weights/goertek_kws_weights_20210513.mat"
    spio.savemat(weights_save_path, trained_model_dict)
    print("***********************")
    print(model.state_dict())
    print("weights saved!")
    print("w length is ", len(w))

if __name__ == "__main__":
    train()