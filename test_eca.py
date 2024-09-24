#All the code goes here
import argparse

import unet_eca as unet
import tensorflow as tf
import soundfile as sf
import numpy as np
from tqdm import tqdm
import scipy.signal
import hydra
import os
#workaround to load hydra conf file
import yaml
import os

os.environ['CUDA_VISIBLE_DEVICES'] ="0"
#os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

from pathlib import Path


def do_stft(noisy):
        
    window_fn = tf.signal.hamming_window

    win_size=args.stft["win_size"]
    hop_size=args.stft["hop_size"]

    
    stft_signal_noisy=tf.signal.stft(noisy,frame_length=win_size, window_fn=window_fn, frame_step=hop_size, pad_end=True)
    stft_noisy_stacked=tf.stack( values=[tf.math.real(stft_signal_noisy), tf.math.imag(stft_signal_noisy)], axis=-1)

    return stft_noisy_stacked

def do_istft(data):
    
    window_fn = tf.signal.hamming_window

    win_size=args.stft["win_size"]
    hop_size=args.stft["hop_size"]

    inv_window_fn=tf.signal.inverse_stft_window_fn(hop_size, forward_window_fn=window_fn)

    pred_cpx=data[...,0] + 1j * data[...,1]
    pred_time=tf.signal.inverse_stft(pred_cpx, win_size, hop_size, window_fn=inv_window_fn)
    return pred_time

def denoise_audio(audio):

    data, samplerate = sf.read(audio)
    print(data.dtype)
    #Stereo to mono
    if len(data.shape)>1:
        data=np.mean(data,axis=1)
    
    if samplerate!=44100: 
        print("Resampling")
   
        data=scipy.signal.resample(data, int((44100  / samplerate )*len(data))+1)
    
    
    segment_size=44100*5  #20s segments

    length_data=len(data)
    overlapsize=2048 #samples (46 ms)
    window=np.hanning(2*overlapsize)
    window_right=window[overlapsize::]
    window_left=window[0:overlapsize]
    audio_finished=False
    pointer=0
    denoised_data=np.zeros(shape=(len(data),))
    residual_noise=np.zeros(shape=(len(data),))
    numchunks=int(np.ceil(length_data/segment_size))
     
    for i in tqdm(range(numchunks)):
        if pointer+segment_size<length_data:
            segment=data[pointer:pointer+segment_size]
            #dostft
            segment_TF=do_stft(segment)
            segment_TF_ds=tf.data.Dataset.from_tensors(segment_TF)
            pred = unet_model.predict(segment_TF_ds.batch(1))
            pred=pred[0]
            residual=segment_TF-pred[0]
            residual=np.array(residual)
            pred_time=do_istft(pred[0])
            residual_time=do_istft(residual)
            residual_time=np.array(residual_time)

            if pointer==0:
                pred_time=np.concatenate((pred_time[0:int(segment_size-overlapsize)], np.multiply(pred_time[int(segment_size-overlapsize):segment_size],window_right)), axis=0)
                residual_time=np.concatenate((residual_time[0:int(segment_size-overlapsize)], np.multiply(residual_time[int(segment_size-overlapsize):segment_size],window_right)), axis=0)
            else:
                pred_time=np.concatenate((np.multiply(pred_time[0:int(overlapsize)], window_left), pred_time[int(overlapsize):int(segment_size-overlapsize)], np.multiply(pred_time[int(segment_size-overlapsize):int(segment_size)],window_right)), axis=0)
                residual_time=np.concatenate((np.multiply(residual_time[0:int(overlapsize)], window_left), residual_time[int(overlapsize):int(segment_size-overlapsize)], np.multiply(residual_time[int(segment_size-overlapsize):int(segment_size)],window_right)), axis=0)
                
            denoised_data[pointer:pointer+segment_size]=denoised_data[pointer:pointer+segment_size]+pred_time
            residual_noise[pointer:pointer+segment_size]=residual_noise[pointer:pointer+segment_size]+residual_time

            pointer=pointer+segment_size-overlapsize
        else: 
            segment=data[pointer::]
            lensegment=len(segment)
            segment=np.concatenate((segment, np.zeros(shape=(int(segment_size-len(segment)),))), axis=0)
            audio_finished=True
            #dostft
            segment_TF=do_stft(segment)

            segment_TF_ds=tf.data.Dataset.from_tensors(segment_TF)

            pred = unet_model.predict(segment_TF_ds.batch(1))
            pred=pred[0]
            residual=segment_TF-pred[0]
            residual=np.array(residual)
            pred_time=do_istft(pred[0])
            pred_time=np.array(pred_time)
            pred_time=pred_time[0:segment_size]
            residual_time=do_istft(residual)
            residual_time=np.array(residual_time)
            residual_time=residual_time[0:segment_size]
            if pointer==0:
                pred_time=pred_time
                residual_time=residual_time
            else:
                pred_time=np.concatenate((np.multiply(pred_time[0:int(overlapsize)], window_left), pred_time[int(overlapsize):int(segment_size)]),axis=0)
                residual_time=np.concatenate((np.multiply(residual_time[0:int(overlapsize)], window_left), residual_time[int(overlapsize):int(segment_size)]),axis=0)

            denoised_data[pointer::]=denoised_data[pointer::]+pred_time[0:lensegment]
            residual_noise[pointer::]=residual_noise[pointer::]+residual_time[0:lensegment]
    return denoised_data

def SNR_singlech(clean, ori):
    length = min(len(clean), len(ori))
    est_noise = ori[:length] - clean[:length]#计算噪声语音

    #计算信噪比
    SNR = 10*np.log10((np.sum(clean**2))/(np.sum(est_noise**2)))
    return SNR

def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--input_audio_path', type=str, help='input_audio_path')
    parser.add_argument('--out_put_dir', type=str, help='out_put_dir')
    parser.add_argument('--yaml_path', type=str, help='yaml_path')
    parser.add_argument('--ckpt_path', type=str, help='ckpt_path')
    args_temp = parser.parse_args()
    return args_temp


if __name__ == '__main__':
    args2 = parse_args()
    args = yaml.safe_load(Path(args2.yaml_path).read_text())
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    args=dotdict(args)
    unet_args=dotdict(args.unet)

    path_experiment=str(args.path_experiment)

    unet_model = unet.build_model_denoise(unet_args=unet_args)

    # ckpt=os.path.join('./experiments/trained_model/eca', 'checkpoint')
    ckpt=args2.ckpt_path
    unet_model.load_weights(ckpt)

    fn = args2.input_audio_path
    # fn1='./clean/%03d.wav'%i
    denoise_data = denoise_audio(fn)
    basename = args2.out_put_dir
    wav_output_name = os.path.join(basename, os.path.basename(fn)[:-4]+'_denoised.wav')
    sf.write(wav_output_name, denoise_data, 44100)
    # clean,_=sf.read(fn1)
    # print(SNR_singlech(clean,denoise_data))