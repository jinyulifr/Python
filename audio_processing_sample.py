#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:21:13 2024

@author: jinyuli
"""

#This script can be used to separate an audio file by pauses.
#Then, amplitude modulation analysis is performed to extract 
#syllabic and supra-syllabic components (syllAM and stressAM) in each separated interval.
#The coordination of these two components is analyzed by computing the Phase Locking Value (PLV).
#Finally, 100 surrogate PLVs are generated to examine if the coordination 
#between syllAM and stressAM is significant 
#by comparing the real PLV to the surrogate PLVs using a one-sample t-test.

# import libraries
import os
import csv
import glob
import torch
torch.set_num_threads(1)
from IPython.display import Audio
from spBandsAMCoord import normalize_IMF, MFB_coeffs, get_PLV, find_valleys, get_closest_Vals_Idxs, get_vow_peaks
from scipy import signal
from scipy.fftpack import next_fast_len
from scipy.io import wavfile
from scipy.signal import hilbert
from scipy import stats
import numpy as np
# import matplotlib.pyplot as plt

# General settings
path="/Volumes/Jinyu LI//test/"
outFileName="Coordination.csv"
outTxtFile=path+outFileName #Name of output file
newFile=open(outTxtFile,'w', newline='') # open output file for writing
newFileWriter = csv.writer(newFile) # open output file for writing
# write headers line to output file
newFileWriter.writerow(["filename","interval_nb","PLV","pValue"])

sr = 16000
Fcor=np.array([0.1,0.9,2.5,12,40]) # vector of frontiers for AM filters
NSamp=1000 #sampling frequency of the AM signals
winLenS=0 # window size for PLV computation
winStepS=0.05 # window hop size for PLV computation
maxRelFreqOrd=10 # maximum ratio between n and m for PLV computation
lPasFord=100 # 11
frame_size = 0.025 # window size for MFCCs computation
frame_stride = 0.001# window hop size for MFCCs computation

lPassDelay=int((lPasFord-1)/2) #delay introduced by the low pass filter
bLow = signal.firwin(lPasFord, 12*frame_stride, window = "hamming", pass_zero='lowpass') #low pass filter coefficients
ds2=sr/NSamp #ratio between the initial sampling freq (sr) and the sampling freq of the obtained AMs (1000 Hz)
bLow1= signal.firwin(lPasFord, 12/NSamp, window = "hamming", pass_zero='lowpass') #low pass filter coefficients
winLen=winLenS*NSamp # win len in frames for PLV computations
winStep=winStepS*NSamp # win step in frames for PLV computations

# Identify the pauses in the audio file
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

for wav_file in glob.glob(path+"*.wav"):

    wav = read_audio(wav_file, sampling_rate=sr)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr) #get speech timestamps from full audio file
    
    filename = os.path.split(wav_file)[1]
    print("Processing {}...".format(filename))
    
    sr0, d = wavfile.read(wav_file)
    if len(np.shape(d))>1:
        d=d[:,0] #get first channel
    if (sr0 != sr): #resample to 16000 Hz
        myRatio=sr/sr0
        newLen=np.round(np.size(d)*myRatio)
        d=signal.resample(d, newLen.astype(int))
    T=np.array(range(1,int((np.shape(d)[0]+1))))/sr
    
    # audioPLVlst=[]
    # audioSurrPLVlst=[]
    
    
    for i in range(len(speech_timestamps)):
        thisStart=speech_timestamps[i]["start"]/sr
        thisEnd=speech_timestamps[i]["end"]/sr
        
        if thisEnd-thisStart>=3:
            thisIdxs=np.where( (T>=thisStart) & (T<=thisEnd) )[0]#get index of current interval
            thisWAV=d[thisIdxs]
            
            #### GET AMPLITUE MODULATIONS ####
            CF_MFB , MFB_bpfs = MFB_coeffs(Fcor,NSamp)  # get coefficients to band pass fiter acoustic energy signal
            sil = np.zeros(int(np.shape(MFB_bpfs)[0]/2)); #build silent chunk                                   
            CF_MFB = CF_MFB[1:] #remove coefficients for the first AM band (the first one is only used to compute to lower frontier of the second one)
            
            # get acoustic energy by extracting the amplitude of the Hilbert tranfrorm
            targetYlen=next_fast_len(np.size(thisWAV)) #length to which bring the signal to make hilber transform fast 
            Ahil = abs(signal.hilbert(d))#compute the amplitude of the hilbert transform and discard the values corresponding to the added chunk
            Ahil=Ahil[::int(ds2)]# decimate amplitude of the hilber transform at 1000 Hz
            amPsig= signal.lfilter(bLow1, 1, Ahil)[lPassDelay:] #smooth (low pass filter) the amplitude signal and remove the initial frames to correct for the filter delay
            M=np.size(Fcor)-2 # number of AM bands to be extracted
            Ahil = np.hstack((sil, Ahil, sil)) # add initial and final silent portions
            Afil = np.zeros((np.size(Ahil),M)) # initialize matrix that wil contain the AM signals

            for m in range(1,M+1):  #for each AM signal
                bpf_len  = int(MFB_bpfs[0,m]) # filter order                                              
                bpf_chan = MFB_bpfs[1:bpf_len,m] #filter coefficients                         
                Atmp = signal.lfilter(bpf_chan, 1, Ahil) # AM signal
                dly_shift = np.floor(bpf_len/2) #compute filter delay                     
                len_A = int(np.size(Atmp) - dly_shift) #length of the filtered signal once compensated for the delay      
                Afil[:len_A-1,m-1] = Atmp[1+int(dly_shift):np.size(Atmp)] #store the useful portion of the filtered signal
                Afil[len_A:np.size(Atmp),m-1] = 0 # set the remaining portion of the singal to 0      
            AMs=Afil[np.size(sil):-np.size(sil),:]   #remove the silent portion of the AM signals from the matrix      
            
            AMs=AMs[:,0:2] #retain only suprasyllabic (index 0) and syllabic (index 1)  AMS
            AMPT=np.array(range(1,int((np.shape(AMs)[0]+1))))/NSamp #time stamps of AM signals
            
            #### GET amplitude of AMPLITUE MODULATIONS via hilbert transform (it is applied once to the matrix of AM signals and operated separately on each column)
            relAMamp=np.abs(signal.hilbert(AMs,axis=0)) # get amplitude of hilbert transform (discard amplitude of added chunks)
            relSyllAMamp=relAMamp[:,1]/np.sum(relAMamp,1) #get syllable relative amplitude (divide each val by the sum of the corresponding syll and suprasyll vals)
            AMampT=np.array(range(1,int((np.size(relSyllAMamp)+1))))/NSamp #time stamps of relative amplitude signals
            
            #### BRING SIGNALS TO SAME TIME SCALE AND NORMALIZE AMPLITUDE
            theseSigs=np.zeros((np.shape(AMs)[0],2)) #initialize matrix containing the AM singals with normalized peaks and valleys
            allStrPcks= signal.find_peaks(AMs[:,0]) #find all peaks of suprasyllabic AM before normaliztoin (it's here that we get the number of peaks)
            allStrPckTs=AMampT[allStrPcks[0]] #find the time stamps of the peaks of suprasyllabic AM (for later usage)
            allSyllPcks= signal.find_peaks(AMs[:,1]) #find all peaks of syllabic AM before normaliztoin (it's here that we get the number of peaks)
            allSyllPckTs=AMampT[allSyllPcks[0]] #find the time stamps of the peaks of syllabic AM (for later usage)
            
            # the normalization procedure works only for portions of signals from the first peak or valley to the last one
            #preceding and following values become NaN. We need to store the first and the last usable values 
            #initilialize matrices containing first and last good indexes
            firstNonNanIdx=np.empty(3)
            lastNonNanIdx=np.empty(3)
            firstNonNanIdx.fill(np.nan)
            lastNonNanIdx.fill(np.nan)

            #initilialize matrices containing peaks and valleys of the signals
            allPcks=[None] * 2
            allPcksT=[None]*2
            allPckVals=[None]*2
            allValls=[None] * 2
            
            for sigN in range(0,2):#for each signal
                mySig=AMs[:,sigN]#get the signal
                newPcksLocs=signal.find_peaks(mySig)[0]
                newVallsLocs=find_valleys(mySig,newPcksLocs.T,1)# get valleys between peaks (and preceding/following the first/last peak)
                
                allValls[sigN]=newVallsLocs# store valleys locations
                allPcksT[sigN]=AMPT[newPcksLocs]#store times of peaks
                allPcks[sigN]=newPcksLocs#store peak locatins in frames
                allPckVals[sigN]=mySig[newPcksLocs]#store signal peak values
                
                theseSigs[:,sigN]=AMs[:,sigN]
                firstNonNanIdx[sigN]=0
                lastNonNanIdx[sigN]=np.shape(AMs[:,sigN])[0]
            
            #get lastest first good index and first last good index
            firstNonNanIdx=max(firstNonNanIdx).astype(int)+1
            lastNonNanIdx=min(lastNonNanIdx).astype(int)
            
            #extract portions of signals and time stamps corresponding to values correctly normalized
            tmpSigs=theseSigs[firstNonNanIdx:lastNonNanIdx,]
            shortAMT0=AMPT[firstNonNanIdx:lastNonNanIdx]
            shortsyllT0=AMPT[firstNonNanIdx:lastNonNanIdx]
            
            #### GET PHASE VALUES OVER TIME
            phVals=np.angle(hilbert(tmpSigs, axis=0))+np.pi #phase vals via hilbert transform
            syllPh=phVals[:,1]#syllable AMphase
            
            #AM signals instantaneous frequency over time    
            phValsD=np.diff(np.unwrap(phVals),axis=0)
            syllPhD=phValsD[:,1] #syllable AM instantaneous freq.
            badGuys=np.where(np.sum(phValsD<0,axis=1)>0) #get indexes where syllabe AM od stress AM have negative frequencies

            #remove portions of signals where negative frequencies are observed
            phVals=np.delete(phVals,badGuys,0)#
            shortAMT=np.delete(shortAMT0,badGuys)  
            normAMsigs=np.delete(tmpSigs,badGuys,0)#get normalized AM signals without portion with negative frequencies
            
            #### GET PLVs
            PLV,PLVidx, n_theta1,m_theta2=get_PLV(phVals[:,0],phVals[:,1],np.shape(phVals)[0],np.shape(phVals)[0],maxRelFreqOrd)
            if n_theta1==m_theta2:
                n_theta1=m_theta2=1
            #get the PLV in different time windows by using the obtained m and n values
            PLV,PLVidx, n_theta1,m_theta2=get_PLV(phVals[:,0],phVals[:,1],np.shape(phVals)[0],np.shape(phVals)[0],[n_theta1,m_theta2])
            
            #### GET surrogate PLV
            ramdomArray=np.random.randint(1,len(tmpSigs), size=100)
            surrPLVlst=[]
            for r in range(len(ramdomArray)):
                thisSyllSurrSigs=np.concatenate((tmpSigs[:,1][ramdomArray[r]:],tmpSigs[:,1][0:ramdomArray[r]]))
                surrSyllPhVals=np.angle(hilbert(thisSyllSurrSigs, axis=0))+np.pi
                
                surrSyllphValsD=np.diff(np.unwrap(surrSyllPhVals),axis=0)
                surrSyllbadGuys=np.where(np.sum(surrSyllphValsD<0)>0)
                
                surrSyllPhVals=np.delete(surrSyllPhVals,surrSyllbadGuys,0)
                
                surrPLV,surrPLVidx, surr_n_theta1,surr_m_theta2=get_PLV(phVals[:,0],surrSyllPhVals,np.shape(phVals)[0],np.shape(phVals)[0],maxRelFreqOrd)
                if surr_n_theta1==surr_m_theta2:
                    surr_n_theta1=surr_m_theta2=1
                surrPLV,surrPLVidx, surr_n_theta1,surr_m_theta2=get_PLV(phVals[:,0],phVals[:,1],np.shape(phVals)[0],np.shape(phVals)[0],[surr_n_theta1,surr_m_theta2])
                
                surrPLVlst=surrPLVlst+list(surrPLV)
                
            #### RUN one sample t-test 
            statsResults=stats.ttest_1samp(surrPLVlst, popmean=PLV[0])
            pValue=statsResults[1]
            
            newFileWriter.writerow([filename,r,PLV[0],pValue])
            
            





