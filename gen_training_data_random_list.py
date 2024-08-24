import numpy as np
import scipy.io.wavfile as wavio
# import wave
import scipy.io as spio
import os
import sys
script_path = "./Libs/"
# script_path2 = "../Libs/"
sys.path.append(os.path.abspath(script_path))
from scipy import signal
from numpy.linalg import norm
from Libs.utils import get_recursive_files
from Libs.lcj_io import getDirsInFolder
import librosa
from tqdm import tqdm
import timeit
# import yaml
from random import shuffle
import json
from datetime import date

json_f = open("./config.json")
#read the first dictionary
config_data = json.load(json_f)[0]

fs = 16000
windowSize = fs * 0.025
windowStep = fs * 0.010
nDims = 40
context_l = 30
context_r = 10

keyword_path = config_data["keyword_data_path"]#"../speech_data/goertek/Silent_room/"
filler_path = config_data["filler_data_path"]#"../speech_data/goertek/voice_noise/"
noise_path = config_data["noise_data_path"]#"../speech_data/goertek/Silent_room"
json_f.close()
meta_file_name = config_data["Training_MetaData"]+"kws_train_wav_file_list_{}.mat".format(str(date.today()).replace("-", ""))
# bk_noise_path = "../../../../Rick/speech_data/goertek/noise/"
x_class1 = []
y_class1 = []
keyword_label = 2
filler_label = 1
noise_label = 0

def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def vad_test(s, fs):
    s = s - np.amin(s)
    s = s / np.amax(s)
    FrameSize = int(fs * 0.025)  # 400
    ShiftSize = int(fs * 0.010)  # 160
    Overlap = FrameSize - ShiftSize  # 240
    threshold = -1.9
    s_temp = []
    temp = []
    temp_all = []
    new = []
    rest_s = []
    t = s
    n = np.floor((len(s) - FrameSize) / ShiftSize)  # 97
    loop_size = int(ShiftSize * n + FrameSize)  # 15920
    norm_t = norm(t, 2)  # 115.2325447
    for i in range(FrameSize, loop_size, ShiftSize):
        temp = np.log(norm(t[i - FrameSize:i], 2) / norm_t + 0.00001)
        # temp_all = np.insert(temp_all, temp)#[temp_all, temp]
        temp_all = np.hstack((temp_all, temp))
        if temp > threshold:
            # new = [new, 1 * np.ones(ShiftSize, 1)]
            new = np.hstack((new, 1 * np.ones(ShiftSize)))
        else:
            # new = [new, 0 * np.ones(ShiftSize, 1)]
            new = np.hstack((new, 0 * np.ones(ShiftSize)))

    # for i in range(ShiftSize * n + FrameSize):
    # for i in range(loop_size): #15920
    s_temp = np.array(s)

    end = len(new)  # len(s_temp)
    s_temp = s_temp[0:end]  # s_temp[0:(end - Overlap)]
    new_s = np.transpose(new) * s_temp

    for j in range(len(new)):
        if new[j] == 1:
            rest_s = np.hstack((rest_s, new_s[j]))
            # rest_s = np.insert(rest_s, new_s[j])
    return rest_s

def get_mel_fb(SR=16000, num_fft=1024, num_mels=40, F_Min=133, F_Max=1200):
    mel_fb = librosa.filters.mel(sr=SR, n_fft=num_fft, n_mels=num_mels, fmin=F_Min, fmax=F_Max, norm=None)
    ret_fb = mel_fb.T
    return ret_fb


# def get_mfcc_librosa(wav_sig=None, sample_rate=16000, frame_length=400, step_length=160,
#                      num_mels=40, num_mfccs=40, mel_fb=None, dct_type=2, window='hamming'):
#     tmp_melspec = librosa.feature.melspectrogram(y=wav_sig, sr=sample_rate,
#                                                  S=mel_fb,
#                                                  n_mels=num_mels,
#                                                  n_fft=1024,
#                                                  hop_length=step_length,
#                                                  win_length=frame_length,
#                                                  window=window)

#     tmp_melspec = librosa.power_to_db(tmp_melspec)
#     _mfcc = librosa.feature.mfcc(S=tmp_melspec, dct_type=dct_type, n_mfcc=num_mfccs, norm=None, lifter=0)
#     return _mfcc

# def get_librosa_defult_mfcc(wav_sig=None, sample_rate=16000, frame_length=400, step_length=160,
#                             num_mels=40, num_mfccs=40):
#     mfcc_ = librosa.feature.mfcc(y=wav_sig, sr=16000, n_mfcc=40, n_mels=40,
#                                  win_length=frame_length, hop_length=step_length)
#     return mfcc_

# def gen_train_data(sig_, sr_, label):
#     removed_sig = vad_test(sig_, 16000)
#     len_of_sig_ = len(removed_sig)
#     if len_of_sig_ > 8000:
#         removed_sig = removed_sig[(len_of_sig_ - 8000):len_of_sig_]
#     elif len_of_sig_ < 8000:
#         pad_array = 0.5 + np.random.rand(8000 - len_of_sig_) * 10 ** -6
#         removed_sig = np.hstack((pad_array, removed_sig))
#     melfb = get_mel_fb()
#     coeff = get_mfcc_librosa(wav_sig=removed_sig, mel_fb=melfb, window=None)
#     # coeff = get_librosa_defult_mfcc(wav_sig=removed_sig)
#     temp = coeff[0, :]-np.amin(coeff[0,:])
#     coeff[0, :] = temp / np.amax(temp)
#     nframe = len(coeff[0, :])
#     #[zeros(nDims, context_l), coeff, zeros(nDims, context_r)];
#     coeff = np.hstack((np.zeros((nDims, context_l)), coeff))
#     coeff = np.hstack((coeff, np.zeros((nDims, context_r))))
#     x = np.zeros((nDims,0))
#     y = np.zeros(0)
#     for context in range(nframe):
#         xx = np.zeros((0,40))
#         window = coeff[:, context:(context+context_l+context_r)]
#         wLoop = context_l+context_r
#         for w in range(wLoop):
#             be_stacked_win = window[:, w]
#             xx = np.vstack((xx, window[:, w]))
#         # xx = xx[1:]
#         x = np.hstack((x, xx))
#         y = np.hstack((y, float(label)))

#     return x, y

TOTAL_POSITIVE_SAMPLE_QUANTITIES = 10
TOTAL_FILLER_SAMPLE_QUANTITIES = 10
# TOTAL_SILENCE_SAMPLE_QUANTITIES = 1

def get_avg_filler_files(file_path):
    filler_classes = getDirsInFolder(file_path)
    print(type(filler_classes))
    print(filler_classes)
    """
    To-Do:
    get all different fillers in same proportion.
    """

def get_shuffled_wav_array():
    # keyword_files = get_recursive_files(keyword_path)[0:TOTAL_POSITIVE_SAMPLE_QUANTITIES]
    # test_files = get_recursive_files(test_path)
    # filler_files = get_recursive_files(filler_path)[0:TOTAL_FILLER_SAMPLE_QUANTITIES]
    keyword_files = get_recursive_files(keyword_path)
    filler_files = get_recursive_files(filler_path)#get_avg_filler_files(filler_path)
    noise_files = get_recursive_files(noise_path)
    # bk_noise_files = get_recursive_files(bk_noise_path)
    len_of_kw_files = len(keyword_files)
    len_of_filler_files = len(filler_files)
    len_of_noise_files = len(noise_files)
    # len_of_noise_files = len(bk_noise_files)
    print("ALL keyword training files are {}".format(len_of_kw_files))
    print("ALL filler training files are {}".format(len_of_filler_files))
    print("ALL silence training files are {}".format(len_of_noise_files))
    # print("ALL bk noise training files are {}".format(len_of_noise_files))
    print(type(filler_files))
    total_files_list = []
    ####https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array###
    # keyword_files = np.array(keyword_files[0:TOTAL_POSITIVE_SAMPLE_QUANTITIES]).reshape([TOTAL_POSITIVE_SAMPLE_QUANTITIES,1])
    # padded_ary = np.full((TOTAL_POSITIVE_SAMPLE_QUANTITIES, 2), keyword_label)
    # print(padded_ary.shape)
    # padded_ary = padded_ary[:, : 1].astype(np.str)
    # padded_ary[:, : 1] = keyword_files
    # print(padded_ary.shape)
    ###############################################################################################
    for f in keyword_files:
        total_files_list.append([f, keyword_label])
    for f in filler_files:
        total_files_list.append([f, filler_label])
    for f in noise_files:
        total_files_list.append([f, noise_label])
    # for f in bk_noise_files:
    #     total_files_list.append([f, noise_label])
    total_files_ary = np.array(total_files_list)
    np.random.shuffle(total_files_ary)
    return total_files_ary

def debug_get_shuffled_wav_array():
    # keyword_files = get_recursive_files(keyword_path)[0:TOTAL_POSITIVE_SAMPLE_QUANTITIES]
    # test_files = get_recursive_files(test_path)
    # filler_files = get_recursive_files(filler_path)[0:TOTAL_FILLER_SAMPLE_QUANTITIES]
    keyword_files = get_recursive_files(keyword_path)
    filler_files = get_recursive_files(filler_path)#get_avg_filler_files(filler_path)
    noise_files = get_recursive_files(noise_path)
    # bk_noise_files = get_recursive_files(bk_noise_path)
    len_of_kw_files = len(keyword_files)
    len_of_filler_files = len(filler_files)
    len_of_noise_files = len(noise_files)
    # len_of_noise_files = len(bk_noise_files)
    print("ALL keyword training files are {}".format(len_of_kw_files))
    print("ALL filler training files are {}".format(len_of_filler_files))
    print("ALL silence training files are {}".format(len_of_noise_files))
    # print("ALL bk noise training files are {}".format(len_of_noise_files))
    # print(type(filler_files))
    total_files_list = []
    ####https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array###
    # keyword_files = np.array(keyword_files[0:TOTAL_POSITIVE_SAMPLE_QUANTITIES]).reshape([TOTAL_POSITIVE_SAMPLE_QUANTITIES,1])
    # padded_ary = np.full((TOTAL_POSITIVE_SAMPLE_QUANTITIES, 2), keyword_label)
    # print(padded_ary.shape)
    # padded_ary = padded_ary[:, : 1].astype(np.str)
    # padded_ary[:, : 1] = keyword_files
    # print(padded_ary.shape)
    ###############################################################################################
    for f in keyword_files:
        total_files_list.append([f, keyword_label])
    for f in filler_files:
        total_files_list.append([f, filler_label])
    for f in noise_files:
        total_files_list.append([f, noise_label])
    # for f in bk_noise_files:
    #     total_files_list.append([f, noise_label])
    total_files_ary = np.array(total_files_list)
    np.random.shuffle(total_files_ary)
    return total_files_ary

def debug_read_mat(mat_path):
    filelistdict = spio.loadmat(mat_path)["train_data"]
    print(filelistdict)

def debug_main():
    # matpath = "./TrainingMetaData/kws_train_wav_file_list_20210219_fix.mat"
    matpath = "./TrainingMetaData/kws_train_wav_file_list_20210220_office_v1.mat"
    debug_read_mat(matpath)

def main_entry_2():
    all_wav_file_array = debug_get_shuffled_wav_array()
    print("all_wav_file_array shape is {}".format(all_wav_file_array.shape))
    train_files_dict = {"train_data": all_wav_file_array}
    print(train_files_dict)
    # spio.savemat("./TrainingMetaData/kws_train_wav_file_list_20210326.mat", \
    #     train_files_dict, oned_as="row")
    spio.savemat(meta_file_name, train_files_dict, oned_as="row")
    # test_data_array = spio.loadmat("./TrainingMetaData/kws_train_wav_file_list_20210209.mat")["train_data"]
    # print(test_data_array[1][0])
    # print("data saved!")


if __name__ == "__main__":
    # get_avg_filler_files(filler_path)
    main_entry_2()
    # debug_main()