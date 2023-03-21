## Speaker anonymization using McAdams Coefficient
## Publication: Patino, Jose, et al. "Speaker anonymisation using the McAdams coefficient." arXiv preprint arXiv:2011.01130 (2020).
## Originally implemented by Xin Wang for 2nd VoicePrivacy Challenge Workshop & ISCA SPSC Symposium Friday, September 23, 2022
## Modified into a CLI utility by Ünal Ege Gaznepoglu

## Usage:
## python b2.py --input-file={file_name}.wav
## python b2.py --input-file={file_name}.wav --output-file={custom_output_name}.wav
## python b2.py --input-file={file_name}.wav --output-file={custom_output_name}.wav --mcadams=0.65

from pathlib import Path
import random

import click
import numpy as np
import scipy
import scipy.signal
import scipy.io.wavfile as wav
import librosa
import resampy

# TODO: code is not vectorized, hence very slow
def anonym(freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8):
    """ output_wav = anonym(freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8)

    Anonymization using McAdam coefficient.

    :input: freq, int, sampling rate in Hz, 16000 in this case
    :input: samples, np.array, (L, 1) where L is the length of the waveform
    :input: winLengthinms, int, analysis window length (ms), default 20 ms
    :input: shiftLengthinms, int, window shift (ms), default 10 ms
    :input: lp_order, int, order of LP analysis, default 20
    :input: mcadams, float, alpha coefficients, default 0.8

    :output: output_wav, np.array, same shape as samples

    Code by Xin Wang, VoicePrivacy Workshop
    """
    
    eps = np.finfo(np.float32).eps
    samples = samples + eps
    
    # window length and shift (in sampling points)
    winlen = np.floor(winLengthinms * 0.001 * freq).astype(int)
    shift = np.floor(shiftLengthinms * 0.001 * freq).astype(int)
    length_sig = len(samples)
    
    # fft processing parameters
    NFFT = 2 ** (np.ceil((np.log2(winlen)))).astype(int)
    # anaysis and synth window which satisfies the constraint
    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)
    # number of of complete frames  
    Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int) 
    
    # Buffer for output signal
    # this is used for overlap - add FFT processing
    sig_rec = np.zeros([length_sig]) 
    
    # For each frame
    for m in np.arange(1, Nframes):

        # indices of the mth frame
        index = np.arange(m * shift, np.minimum(m * shift + winlen, length_sig))    

        # windowed mth frame (other than rectangular window)
        frame = samples[index] * win 

        # get lpc coefficients
        a_lpc = librosa.core.lpc(frame + eps, order=lp_order)

        # get poles
        poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]

        #index of imaginary poles
        ind_imag = np.where(np.isreal(poles) == False)[0]

        #index of first imaginary poles
        ind_imag_con = ind_imag[np.arange(0, np.size(ind_imag), 2)]
        
        # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
        # values >1 expand the spectrum, while values <1 constract it for angles>1
        # values >1 constract the spectrum, while values <1 expand it for angles<1
        # the choice of this value is strongly linked to the number of lpc coefficients
        # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
        # a smaller lpc coefficients number allows for a bigger flexibility
        new_angles = np.angle(poles[ind_imag_con]) ** mcadams
        #new_angles = np.angle(poles[ind_imag_con])**path[m]
        
        # make sure new angles stay between 0 and pi
        new_angles[np.where(new_angles >= np.pi)] = np.pi
        new_angles[np.where(new_angles <= 0)] = 0  
        
        # copy of the original poles to be adjusted with the new angles
        new_poles = poles
        for k in np.arange(np.size(ind_imag_con)):
            # compute new poles with the same magnitued and new angles
            new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]]) * np.exp(1j * new_angles[k])
            # applied also to the conjugate pole
            new_poles[ind_imag_con[k] + 1] = np.abs(poles[ind_imag_con[k] + 1]) * np.exp(-1j * new_angles[k])            
        
        # recover new, modified lpc coefficients
        a_lpc_new = np.real(np.poly(new_poles))

        # get residual excitation for reconstruction
        res = scipy.signal.lfilter(a_lpc,np.array(1),frame)

        # reconstruct frames with new lpc coefficient
        frame_rec = scipy.signal.lfilter(np.array([1]),a_lpc_new,res)
        frame_rec = frame_rec * win    
        outindex = np.arange(m * shift, m * shift + len(frame_rec))

        # overlap add
        sig_rec[outindex] = sig_rec[outindex] + frame_rec

    sig_rec = sig_rec / np.max(np.abs(sig_rec))
    return sig_rec

@click.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_file', type=click.Path(file_okay=True, path_type=Path), required=False)
@click.argument('mcadams', type=click.FloatRange(min=0.5, max=0.9), required=False)
def main(input_file: Path, output_file: Path, mcadams: float):
    # the default behavior is to randomize
    if not mcadams:
        mcadams = random.uniform(0.5, 0.9)
    # the default behavior is to write a new file in the same directory 
    # with '_anon' appended to the file name
    if not output_file:
        output_file = input_file.with_stem(input_file.stem + f'_{mcadams:.3f}_anon')
    
    # open input file
    fs, data = wav.read(input_file)
    
    # process (system expects FS:16khz so resample first)
    data = resampy.resample(data, sr_orig=fs, sr_new=16000, axis=0)
    data_anonym = anonym(16000, data[...,0], mcadams=mcadams)
    data_anonym = resampy.resample(data_anonym, sr_orig=16000, sr_new=fs, axis=0)

    # write output .wav (floating point, max amplitude normalized to 1)
    wav.write(output_file, fs, data_anonym)

if __name__ == '__main__':
    main()