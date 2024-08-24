import numpy as np

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

def dump_FB_Array(FB=None, fb_save_path=None):
    #     theFB = get_filterbank_from_midfreqs(midfreqs, 16000, 10, 1024)
    fb_dict = {"fb":FB}
    rows = FB.shape[0]
    cols = FB.shape[1]
    with open(fb_save_path, "w")as f:
        f.write("const float filterCoeff[{}][{}]=".format(rows,cols))
        f.write("{")
        for i in range(rows):
            f.write("{")
            for j in range(cols):
                f.write(str(FB[i][j]))
                f.write(",")
            f.write("},")
            f.write("\n")
        f.write("};")
    print("write fileterbank coeff successfully.{}".format(fb_save_path))

if __name__ == "__main__":
    fb = get_filterbanks()
    h_save_path = "../coeff_files/fb_coeff.h"
    mat_save_path = "../coeff_files/fb.mat"
    dump_FB_Array(fb, h_save_path)