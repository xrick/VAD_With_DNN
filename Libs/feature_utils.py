import numpy as np
import scipy.io.wavfile as wavio
import os
import logging

#import cython libraries
import ctypes
from numpy.ctypeslib import ndpointer
#load c library
current_folder = os.path.dirname(os.path.realpath(__file__))
# lib_path = "./CFFT/libfft.so"#current_folder + "/libfft.so"
lib_path = current_folder + "/CFFT/libfft.so"
lib = ctypes.cdll.LoadLibrary(lib_path)
fun = lib.get_footprint
fun.restype = None
fun.argtypes = [ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                #ctypes.c_size_t,
                ctypes.c_size_t,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def hz2mel_nature(freq):
    return 1127. * np.log(1. + freq / 700.)

def mel2hz_nature(mel):
    return 700. * (np.exp(mel / 1127.) - 1.)

def hz2mel(hz):
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

def magspec(frames, NFFT):
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def get_filterbanks(nfilt=40,nfft=1024,samplerate=16000,lowfreq=0,highfreq=8000):
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    # compute points evenly spaced in mels
    lowmel = hz2mel_nature(lowfreq)
    highmel = hz2mel_nature(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    mid_freqs = mel2hz_nature(melpoints)
    bins = np.floor((nfft+1)*mid_freqs/samplerate)
    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bins[j]), int(bins[j+1])):
            fbank[j,i] = (i - bins[j]) / (bins[j+1]-bins[j])
        for i in range(int(bins[j+1]), int(bins[j+2])):
            fbank[j,i] = (bins[j+2]-i) / (bins[j+2]-bins[j+1])
    return fbank

#計算log mel-scale filter bank energies(要記錄在文件，關於理論的部份)
def compute_sig_lfbe(signal, mel, num_fft=1024):
    sig = signal.reshape(1,len(signal))
    sig = magspec(sig,num_fft)
    feat = np.dot(sig,mel.T) # compute the filterbank energies
    feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is
    db_feat = 10 * np.log10(feat)
    return db_feat

def padding_ndarray_to_1024(ori_ary, padding_ary, size):
    ret_ary = np.resize(ori_ary,new_shape=len(ori_ary) + size)
    ret_ary[-size:] = padding_ary
    return ret_ary

def compute_sig_lfbe_cfft(signal):
    CC = 16
    BANDNUM = 40
    inputary = np.array(signal, dtype=np.int32)
    outary = np.empty((BANDNUM),dtype=np.float32)
    fun(inputary,BANDNUM,CC,outary)
    return outary

def compute_sig_power(signal):
    ret_ary_size = 513
    inputary = np.array(signal, dtype=np.int32)
    retary = np.empty((ret_ary_size),dtype=np.float32)
    BANDNUM = 40
    fun(inputary,BANDNUM,retary)
    return retary

def safe_wav_read(wav_file):
    try:
        std_sr = 16000
        sr, sig = wavio.read(wav_file)
        if sig.shape[0] < sig.size:
            sig = sig[0]
            print("\n{} is channel 2".format(wav_file))
        return sr, sig
    except:
        print("Error occured in read and convert wav to ndarray in file {}".format(wav_file))

def gen_new_4_frames(sig_lfbe, std_len=1600):
    sigLen = len(sig_lfbe)
    zero_len = std_len - sigLen
    zeroAry = None
    if zero_len > 39:
        zeroAry = np.full((zero_len),0)
        retAry = np.hstack([zeroAry,sig_lfbe])
        return retAry
    else:
        print("zero_len:{}".format(zero_len))
        return None
    
def padding_zeros(sig, want_len=16000):
    pad_len = want_len - len(sig)
    zeroAry = np.zeros(pad_len)
    retAry = np.hstack([sig,zeroAry])
    return retAry

def read_signal(wav_file):
    sr, sig = safe_wav_read(wav_file)
    std_wav_len = 16000
    if len(sig)>std_wav_len:
        sig = sig[0:std_wav_len]
    if len(sig) < std_wav_len:
        sig = padding_zeros(sig)
    return sr, sig


def gen_train_signal_ver_2_1600x1_cfft(wav_file):
    """The method is used for generate lfbes for 
    only raw speech file.
    output: 
          lfbe:1x1600
    """
    pad_len = 1024-400
    padding_zeros = np.array(np.zeros(pad_len,dtype=np.int))
    sr, sig = read_signal(wav_file)
    seg_unit_len = 400
    segs = int(len(sig)/seg_unit_len)
    aFB = get_filterbanks()
    #read the first segment
    part_sig = padding_ndarray_to_1024(sig[0:seg_unit_len], padding_zeros,pad_len)
    sig_lfbe = compute_sig_lfbe_cfft(part_sig)
    # sig_labels = np.ndarray(0)
    #compute the rest segments
    for idx in range(1,segs):
        part_sig = padding_ndarray_to_1024(sig[seg_unit_len*idx : seg_unit_len*(idx+1)], padding_zeros,pad_len)
        tmp_lfbe = compute_sig_lfbe_cfft(part_sig)
        sig_lfbe = np.hstack((sig_lfbe, tmp_lfbe))
    return sig_lfbe

def compute_power_lfbe(signal_power, mel):
    feat = np.dot(signal_power,mel.T) # compute the filterbank energies
    feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is
    db_feat = 10 * np.log10(feat)
    return db_feat


def gen_train_lfbe_1600x1_with_cfft_power(wav_file, mel):
    """The method is used for generate lfbes for
    only raw speech file.
    output:
          lfbe:1x1600
    """
    pad_len = 1024-400
    padding_zeros = np.array(np.zeros(pad_len,dtype=np.int))
    sr, sig = read_signal(wav_file)
    seg_unit_len = 400
    segs = int(len(sig)/seg_unit_len)
    # aFB = get_filterbanks()
    #read the first segment
    part_sig = padding_ndarray_to_1024(sig[0:seg_unit_len], padding_zeros,pad_len)
    sig_power = compute_sig_power(part_sig)
    sig_lfbe = compute_power_lfbe(sig_power,mel)
    # sig_labels = np.ndarray(0)
    #compute the rest segments
    for idx in range(1,segs):
        part_sig = padding_ndarray_to_1024(sig[seg_unit_len*idx : seg_unit_len*(idx+1)], padding_zeros,pad_len)
        tmp_power = compute_sig_power(part_sig)
        tmp_lfbe = compute_power_lfbe(tmp_power,mel)
        sig_lfbe = np.hstack((sig_lfbe, tmp_lfbe))
    return sig_lfbe

def gen_train_signal_ver_2_1600x1_nocfft(wav_file):
    """The method is used for generate lfbes for 
    only raw speech file.
    output: 
          lfbe:1x1600
    """
    pad_len = 1024-400
    padding_zeros = np.array(np.zeros(pad_len,dtype=np.int))
    sr, sig = read_signal(wav_file)
    seg_unit_len = 400
    segs = int(len(sig)/seg_unit_len)
    aFB = get_filterbanks()
    sig_lfbe = compute_sig_lfbe(sig[0:seg_unit_len], aFB)
    #compute the rest segments
    for idx in range(1,segs):
        part_sig = sig[seg_unit_len*idx : seg_unit_len*(idx+1)]
        tmp_lfbe = compute_sig_lfbe(part_sig,aFB)
        sig_lfbe = np.hstack((tmp_lfbe,sig_lfbe))
    return sig_lfbe

def gen_train_signal(wav_file, wav_type=0):
    sr, sig = safe_wav_read(wav_file)
    std_wav_len = 16000
    if len(sig)>std_wav_len:
        sig = sig[0:std_wav_len]
    if len(sig) < std_wav_len:
        sig = padding_zeros(sig)
    seg_len = 400
    seg_num = int(np.floor(std_wav_len/seg_len))
    have_processed_frames = np.ndarray([])
    before_kw_frames = 5
    after_kw_frames = 10
    end_of_kw_seg = 30
    before_kw_idx = end_of_kw_seg-before_kw_frames
    after_kw_idx = end_of_kw_seg + after_kw_frames
    tmp_data_list = []
    tmp_lbl_list = []
    lbl_val = 0
    new_1600_lfbe = None
    aFB = get_filterbanks()
    for loopidx in range(seg_num):
        #get frame
        processing_frames = sig[loopidx*seg_len : (loopidx+1)*seg_len]
        #compute log-Mel filter bank energies
        sig_lfbe = compute_sig_lfbe(processing_frames, aFB)
        #stack horizontally
        have_processed_frames = np.hstack((have_processed_frames, sig_lfbe[0])) \
        if have_processed_frames.size > 1 else sig_lfbe[0]
        #generate new training record ie. 1600
        if loopidx < (seg_num-1):
            new_1600_lfbe = gen_new_4_frames(have_processed_frames)
        else:
            new_1600_lfbe = have_processed_frames
        #compute the label value
        if wav_type == 2:
            if loopidx < before_kw_idx:
                lbl_val = 0
            elif loopidx > after_kw_idx:
                lbl_val = 0
            else:
                lbl_val = 2
        else:
            lbl_val = wav_type

        tmp_data_list.append(new_1600_lfbe)
        tmp_lbl_list.append(lbl_val)
    return tmp_data_list, tmp_lbl_list

def get_wav_file_name(wav_file_path):
    wavfilename = os.path.basename(wav_file_path).replace(".","_")
    return wavfilename

def gen_main_entry(wavpath, wav_lbl):
    filename = get_wav_file_name(wavpath)
    datalist, lbllist = gen_train_signal(wavpath, wav_lbl)
    #先將datalist第一列的資料作為dataary的第一個元素
    dataary = np.array(datalist[0])
    lblary = np.array(lbllist)
    for e in datalist[1:len(datalist)]:
        dataary = np.vstack((dataary,e))
    return dataary, lblary
    
# def test_main():
#     testwav = "../../../speech_data/goertek/Silent_room/20201021/0037393.wav"
#     mel_ = get_filterbanks()
#     lfbe = gen_train_lfbe_1600x1_with_cfft_power(testwav, mel_)
#     print(lfbe.shape)
#     # print(labels[0])
# #     # fb = get_filterbanks()
# #     # print(fb.shape)
#
#
# if __name__ == "__main__":
#     test_main()