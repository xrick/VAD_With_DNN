import numpy as np
import scipy.io as spio
import scipy.io.wavfile as wavio
import os
import sys
import torch
# mfcc_script_path = "../../Libs"
libs_script = "./Libs/"
sys.path.append(os.path.abspath(libs_script))
# sys.path.append(os.path.abspath(mfcc_script_path))
from Libs.utils import get_recursive_files
from Libs.feature_utils import *
np.set_printoptions
KEYWORD = 2
FILLER = 1
UNKNOWN = 0

# def debug_print(outString, variables):
#     print(outString.format(variables))

def nth_root(num,root):
   answer = num ** (1/root)
   return answer

def read_paths(test_files_path):
    files = get_recursive_files(test_files_path)
    return files

# def get_mfcc(sig, sr, frame_size, hop_length):
#     pass

def sigmoid(X):
    return 1/(1+np.exp(-X))

def relu(X):
    return np.maximum(0, X)

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum

def load_weights(weights_file = None):
    weights = spio.loadmat(weights_file)
    return weights

# def smooth_answer(toSmoothAns, SmoothFactor):
#     all_smooth_answer = None
#     if j<=SmoothFactor:
#         smooth_answer = (1/(j-1+1))*sum(toSmoothAns(:,1:j),2)
#         all_smooth_answer = [SmoothFactor,smooth_answer]
#     else:
#         k = j-SmoothFactor+1;
#         smooth_answer = (1/(j-k+1))*sum(SmoothFactor(:,k:j),2)
#         all_smooth_answer = [toSmoothAns,smooth_answer]
#     return all_smooth_answer
    
def string_clean(inString):
    return inString.stripe()

def calculat_acc(pred, labels):
    """
    ref: https://discuss.pytorch.org/t/how-does-one-get-the-predicted-classification-label-from-a-pytorch-model/91649/3
    """
    # acc = (pred == labels).sum().item() / pred.size(0)
    acc = sum(pred==labels) / len(pred)
    return acc


# def calculat_acc(pred, labels):
#     """
#     ref: https://discuss.pytorch.org/t/how-does-one-get-the-predicted-classification-label-from-a-pytorch-model/91649/3
#     """
#     acc = (pred == labels).sum().item() / pred.size(0)
#     return acc

#計算一個list中出現最多次的數字
def most_frequent(List):
    return max(set(List), key = List.count)


def processing_evaluatoin(softmax_ans, ans_lbl):#1x3(一個)
    """傳入的只有一個實際語音音檔的一檔所產生的四十秒的一秒結果及其對應label
    """
    #first we divide the 1600 point into 40 group, totally 40 group
    #each group map 1 label value
    # print(ans_lbl)
    # softmax_ans_t = torch.from_numpy(np.transpose(softmax_ans))
    softmax_ans_t = np.transpose(softmax_ans)
    # print(len(ans_lbl))
    # return
    totalRec = loop_num = softmax_ans_t.shape[0]
    retAcc = 0.0
    rightAns = 0
    splitLen = outerRun = len(ans_lbl)
    all_max_pos_list = []
    test_patter_rec_list = []
    stspos = 1
    acc_list = []
    for ans_l in ans_lbl:
        current_sec = softmax_ans_t[stspos-1:stspos*splitLen]#一次取四十個
        part_max_pos_list = []
        # sec_prime = torch.from_numpy(np.transpose(current_sec))
        for e in current_sec:
            max_index = np.argmax(e)
            part_max_pos_list.append(max_index)
            freqnum = most_frequent(part_max_pos_list)
            test_patter_rec_list.append(freqnum)
        stspos = stspos + 1
        all_max_pos_list.append(part_max_pos_list)
        acc = calculat_acc(part_max_pos_list, ans_l)
        acc_list.append(acc)
        return acc_list, all_max_pos_list

    # print("len of all_max_pos_list:{}".format(len(all_max_pos_list)))
# 

def test_logic(weights=None, wav_file=None, ans_type=2):
    """此函式包函了主要的測試邏輯，敘述如下：
    """
    # weight_file_ = ""
    w1 = weights['w1']
    w2 = weights['w2']
    w3 = weights['w3']
    w4 = weights['w4']
    # w5 = weights['w5']
    b1 = weights['b1']
    b2 = weights['b2']
    b3 = weights['b3']
    b4 = weights['b4']
    # b5 = weights['b5']
    # print("w1 shape is {}".format(w1.shape))
    # return
    threshold = 0.25
    w_smooth = 10
    w_max = 100
    n_class = 3
    #preprocessing
    test_data = gen_train_signal_ver_2_1600x1_cfft(wav_file)
    test_data = np.reshape(test_data,(1,len(test_data)))
    # print(test_data.shape)
    # return
    # print("The test wav_file is {} \nand label is {}\n")
    # print("check feautre data from raw wav file")
    # print("current returned data shape is {}\n, lbl_data shape is {}\n".format(test_data.shape,lbl_data.shape))
    loopsize = test_data.shape[0]
    #start training process, describing as following
    run_count = 0
    acc_list = []
    all_pattern_list= []
    for i in range(loopsize):
        run_count = run_count + 1
        d = test_data[i]
        # print("wav is \n {}\n and lable is {}\n*************************".format(d,l))
        pred = np.matmul(d, w1)#d* w1#x_data[j, :]*w1
        # print("w1:{}".format(w1.shape))
        pred = pred + b1
        pred = relu(pred)
        pred = np.matmul(pred, w2)
        # print("w2:{}".format(w2.shape))
        # print("pred after w2:{}".format(pred.shape))
        # print(pred.shape)
        pred = pred + b2
        pred = relu(pred)
        pred = np.matmul(pred, w3)#pred * w3
        # print("w3:{}".format(w3.shape))
        # print("pred after w3:{}".format(pred.shape))
        # print(pred.shape)
        pred = pred + b3
        pred = relu(pred)
        pred = np.matmul(pred, w4)#pred * w4
        # print("w4:{}".format(w4.shape))
        # print("pred after w4:{}".format(pred.shape))
        # print(pred.shape)
        pred = pred + b4
        # print("pred is {}\n".format(pred))
        pred_ans = np.argmax(pred[0])
        # print("pred_ans is {}\n".format(pred_ans))
        acc = 1 if pred_ans==ans_type else 0 #calculat_acc(pred_ans, ans_lbl[i])
        acc_list.append(acc)
        # acc, max_idx_list = processing_evaluatoin(pred_ans,ans_lbl)
        # true_acc = np.sum(acc[acc>0])
        # print(true_acc)
        # all_acc_list.append(acc)
        # all_pattern_list.append(max_idx_list)
        # break
        # print("wav:{} the {} test\n".format(get_wav_file_name(wav_file), i))
        # print("the label is {}\n".format(l))
        # print("the softmax shape is {}\n*********".format(answer.shape))
    trueAcc = sum(acc_list)/len(acc_list)
    # print(trueAcc)
    return trueAcc
    
    """
    all_answer = []
    all_smooth_answer = []

    loop_size = len(test_data[:, 1])
    
    for j in range(loop_size):
        record_dict = {}
        pred = reshaped_x_data * w1#x_data[j, :]*w1
        pred = pred + b1
        pred = relu(pred)
        pred = np.matmul(pred, w2)
        pred = pred + b2
        pred = relu(pred)
        pred = np.matmul(pred, w3)#pred * w3
        pred = pred + b3
        pred = relu(pred)
        pred = np.matmul(pred, w4)#pred * w4
        pred = pred + b4
        answer = softmax(np.transpose(pred)) #result: 1600x3
        # print(answer.shape)
        # print("j:{} answer:{}".format(j,answer))
        record_dict['prediction'] = answer
    """
        
    """
        >>> xs = np.array([[1,2,3,4,5],[10,20,30,40,50]])
        >>> ys = np.array([], dtype=np.int64).reshape(0,5)
        >>> ys
                array([], shape=(0, 5), dtype=int64)
        >>> np.vstack([ys, xs])
            array([[  1.,   2.,   3.,   4.,   5.],
                   [ 10.,  20.,  30.,  40.,  50.]])
            if not:
        >>> ys = np.array([])
        >>> ys = np.vstack([ys, xs]) if ys.size else xs
            array([[ 1,  2,  3,  4,  5],
                   [10, 20, 30, 40, 50]])
    """
    """
        all_answer = np.hstack((all_answer, answer)) if all_answer.size else answer
        # print("j:{}, all_answer:{}".format(j, all_answer))
        # print("size of all_answer is {}".format(all_answer.size))
        #smooth
        if j <= w_smooth:
            #sum(A,2):把每列的每個元素相加，回傳一個行向量(Column Vector)
            smooth_answer = (1 / ((j+1) - 1 + 1)) * sum(all_answer[:, 0: j], 2)
            # print("j:{}, smooth_answer in w_smooth is {}".format(j, smooth_answer))
            all_smooth_answer = [all_smooth_answer, smooth_answer]
        else:
            k = (j+1) - w_smooth + 1
            smooth_answer = (1 / ((j+1) - k + 1)) * sum(all_answer[:, k: j], 2)
            # print("j:{}, smooth_answer in else is {}".format(j, smooth_answer))
            all_smooth_answer = [all_smooth_answer, smooth_answer]
        # writein = "j:{}, all_smooth_answer is {}".format(j, all_smooth_answer)
        # f.writelines(writein)
        # print(writein)
        # calculate confidence, please reference t0he paper "small footprint kws using deep neural networks
        # error 20210217 debug:list indices must be integers or slices, not tuple
        # if j <= w_max:
        #     confi = nth_root(np.amax(all_smooth_answer[1, 1:j])*np.amax(all_smooth_answer[2, 1: j]), n_class - 1)
        #     print("j:{}, confi in w_max:{}".format(j, confi))
        #     all_confi = np.insert(all_confi, confi)#[all_confi, confi]
        # else:
        #     k = j - w_max + 1
        #     print("j:{}, confi in else:{}".format(j, confi))
        #     confi = nth_root(np.amax(all_smooth_answer[1, k:j])*np.amax(all_smooth_answer[2, k: j]), n_class - 1)
        #     all_confi = np.insert(all_confi, confi)
        
        # print("********************** per run end *******************")
    # f.flush()
    # f.close()
    return
    """
    
def run_test_main(weights, wav_file_path, labeltype):
    #load weights
    weights_ = load_weights(weights_file_path)
    #load all test files in to a list
    fnlist = get_recursive_files(wav_file_path)
    f  = fnlist[0]
    accVallist = []
    for f in fnlist:
        accVallist.append(test_logic(weights=weights_, wav_file=f, ans_type=labeltype))
    print("total use {} wav files to test\n".format(len(fnlist)))
    print("The Average accuracy is {}".format(sum(accVallist)/len(accVallist)))

if __name__ == "__main__":
    weights_file_path = "../kws_trained_weights/goertek_kws_weights_20210428.mat"
    test_data_path = "../Speech_DataSets/test/filler"
    test_data_lbl = 1
    run_test_main(weights_file_path, test_data_path, test_data_lbl)
    