#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:14:53 2024

@author: jinyuli
"""
#%%
import os
# os.chdir('/Volumes/Jinyu LI/PASDCODE/Experiments/exp_SynchroSpeech/analyses/')
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from parselmouth.praat import call
# import tgt

#%% General settings
part=2
# level=0 #0=acoustic level; 1=larynx level

path="/Users/jinyuli/Downloads/test/"
# path="/Volumes/Jinyu LI/PASDCODE/Experiments/exp_SynchroSpeech/"
outFileName="plvTest2.csv"
outTxtFile=path+outFileName #Name of output file
# outTxtFile=path+"analyses/"+outFileName #Name of output file
newFile=open(outTxtFile,'w', newline='') # open output file for writing
newFileWriter = csv.writer(newFile) # open output file for writing
# write headers line to output file
newFileWriter.writerow(["filename","Sex","speakerN","trial","word","language","pattern","intervalN","PLV","level"])

sr = 16000

# frontiers for AM filters
segN=72
VowN=36
AccN=18
segRate=segN/12; #segment rate
syllRate=VowN/12; #syllable rate
stressRate=AccN/12; #accent rate
stressCut=(stressRate+syllRate)/2; #frontier between stress and syllables
syllCut=syllRate+(syllRate-stressCut); #frontier between syllables and segment
segCut=segRate+(segRate-syllCut); #upper limit of segment AM band
Fcor=np.array([0.1,0.9,stressCut,syllCut,segCut]) # vector of frontiers for AM filters
# Fcor=np.array([0.1,0.9,2.5,12,40]) # vector of frontiers for AM filters

nSurr=100

NSamp=1000 #sampling frequency of the AM signals
winLenS=0 # window size for PLV computation
winStepS=0.05 # window hop size for PLV computation
maxRelFreqOrd=10 # maximum ratio between n and m for PLV computation
lPasFord=100 # 11
frame_size = 0.025 # window size for MFCCs computation
frame_stride = 0.001# window hop size for MFCCs 

lPassDelay=int((lPasFord-1)/2) #delay introduced by the low pass filter
bLow = signal.firwin(lPasFord, 12*frame_stride, window = "hamming", pass_zero='lowpass') #low pass filter coefficients
ds2=sr/NSamp #ratio between the initial sampling freq (sr) and the sampling freq of the obtained AMs (1000 Hz)
bLow1= signal.firwin(lPasFord, 12/NSamp, window = "hamming", pass_zero='lowpass') #low pass filter coefficients
winLen=winLenS*NSamp # win len in frames for PLV computations
winStep=winStepS*NSamp # win step in frames for PLV computations

#%%
for level in [0,1]:
    for wav_file in glob.glob("/Users/jinyuli/Downloads/test/"+"*.wav"):
    
        filename = os.path.split(wav_file)[1]
        print("Processing {}...".format(filename))
        # txt_file=filename.split(".")[0]+".textgrid"
        
        #get information about the audio
        sex=filename.split(".")[0].split("_")[0]
        speaker=filename.split(".")[0].split("_")[1]
        trial=filename.split(".")[0].split("_")[2]
        word=filename.split(".")[0].split("_")[3]
       
        
        if int(speaker)<11:
            lang="ge"
        else:
            lang="fr"
        
        if part==1:
            if word[2].isupper():
                prom="LH"
            else:
                prom="HL"
        
        if part==2:
            prom=filename.split(".")[0].split("_")[4]
        
        #read audio and resample
        sr0, d = wavfile.read(wav_file)
        if len(np.shape(d))>1:
            d=d[:,level] #get the channel to analyze
        if (sr0 != sr): #resample to 16000 Hz
            myRatio=sr/sr0
            newLen=np.round(np.size(d)*myRatio)
            d=signal.resample(d, newLen.astype(int))
        T=np.array(range(1,int((np.shape(d)[0]+1))))/sr
            
        txtgrd=call("Read from file", os.path.splitext(wav_file)[0]+".textgrid")
        # tg = tgt.read_textgrid(os.path.splitext(wav_file)[0]+".textgrid")
        noi=call(txtgrd,"Get number of intervals",1)
        
        # AM0array=np.empty([1,600])
        # AM1array=np.empty([1,600])
        intVerN=0
        for intN in range(1,noi+1):
            print("Processing interval {}...".format(intN))
            thisLab=call(txtgrd,"Get label of interval", 1, intN)
            if thisLab=='0':
                intVerN=intVerN+1
                thisT1=call(txtgrd,"Get start time of interval",1,intN)
                thisT2=call(txtgrd,"Get end time of interval",1,intN)
            
                thisStartFrWav=np.floor(thisT1*sr).astype(int)
                thisEndFrWav=np.floor(thisT2*sr).astype(int)
                
                d0=d[int(np.floor(thisStartFrWav)):int(np.floor(thisEndFrWav))]
                
                        
                #### GET AMPLITUE MODULATIONS ####
                CF_MFB , MFB_bpfs = MFB_coeffs(Fcor,NSamp)  # get coefficients to band pass fiter acoustic energy signal
                sil = np.zeros(int(np.shape(MFB_bpfs)[0]/2)); #build silent chunk                                   
                CF_MFB = CF_MFB[1:] #remove coefficients for the first AM band (the first one is only used to compute to lower frontier of the second one)
                
                # get acoustic energy by extracting the amplitude of the Hilbert tranfrorm
                targetYlen=next_fast_len(np.size(d0)) #length to which bring the signal to make hilber transform fast 
                Ahil = abs(signal.hilbert(d0))#compute the amplitude of the hilbert transform and discard the values corresponding to the added chunk
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
                
            #     print(len(AMampT))
            #     xvals = np.linspace(0, AMampT[-1], 600)
            #     # AM0Inter=np.interp(xvals, AMampT, AMs[:,0])
            #     # AM1Inter=np.interp(xvals, AMampT, AMs[:,1])
            #     AM0Inter=np.interp(xvals, AMampT, AMs[:,0]).reshape([1,600])
            #     AM1Inter=np.interp(xvals, AMampT, AMs[:,1]).reshape([1,600])
            #     # plt.plot(xvals,AM0Inter)
            #     # plt.plot(xvals,AM1Inter)
                
            #     AM0array=np.concatenate((AM0array, AM0Inter),axis=0)
            #     AM1array=np.concatenate((AM1array, AM1Inter),axis=0)
                
            #     # ### plot signal
            #     # dT=(np.arange(np.size(d0))+1)/sr
            #     # fig, ax = plt.subplots(2, 1, sharex=True, sharey=False,figsize=(10, 15))
            #     # ax[0].plot(AMampT,AMs[:,0])
            #     # ax[0].plot(AMampT,AMs[:,1])
            #     # ax[1].plot(dT,d0)
            #     ###
        
            # meanAM0 = np.mean(AM0array[1:,], axis=0)
            # meanAM1 = np.mean(AM1array[1:,], axis=0)
            # plt.plot(meanAM0)
            # plt.plot(meanAM1)
             
            # ########################################################################
            # # the normalization procedure works only for portions of signals from the first peak or valley to the last one
            # #preceding and following values become NaN. We need to store the first and the last usable values 
            # #initilialize matrices containing first and last good indexes
            # firstNonNanIdx=np.empty(3)
            # lastNonNanIdx=np.empty(3)
            # firstNonNanIdx.fill(np.nan)
            # lastNonNanIdx.fill(np.nan)
        
            # #initilialize matrices containing peaks and valleys of the signals
            # allPcks=[None] * 2
            # allPcksT=[None]*2
            # allPckVals=[None]*2
            # allValls=[None] * 2
            
            # for sigN in range(0,2):#for each signal
            #     mySig=AMs[:,sigN]#get the signal
            #     newPcksLocs=signal.find_peaks(mySig)[0]
            #     newVallsLocs=find_valleys(mySig,newPcksLocs.T,1)# get valleys between peaks (and preceding/following the first/last peak)
                
            #     allValls[sigN]=newVallsLocs# store valleys locations
            #     allPcksT[sigN]=AMPT[newPcksLocs]#store times of peaks
            #     allPcks[sigN]=newPcksLocs#store peak locatins in frames
            #     allPckVals[sigN]=mySig[newPcksLocs]#store signal peak values
                
            #     theseSigs[:,sigN]=AMs[:,sigN]
            #     firstNonNanIdx[sigN]=0
            #     lastNonNanIdx[sigN]=np.shape(AMs[:,sigN])[0]
            
            # #get lastest first good index and first last good index
            # firstNonNanIdx=max(firstNonNanIdx).astype(int)+1
            # lastNonNanIdx=min(lastNonNanIdx).astype(int)
            
            # #extract portions of signals and time stamps corresponding to values correctly normalized
            # tmpSigs=theseSigs[firstNonNanIdx:lastNonNanIdx,]
            # shortAMT0=AMPT[firstNonNanIdx:lastNonNanIdx]
            # shortsyllT0=AMPT[firstNonNanIdx:lastNonNanIdx]
            # ########################################################################
        
                #### GET PHASE VALUES OVER TIME
                phVals=np.angle(hilbert(AMs, axis=0))+np.pi #phase vals via hilbert transform
                # phVals=np.angle(hilbert(tmpSigs, axis=0))+np.pi #phase vals via hilbert transform
                syllPh=phVals[:,1]#syllable AMphase
                
                #AM signals instantaneous frequency over time    
                phValsD=np.diff(np.unwrap(phVals),axis=0)
                syllPhD=phValsD[:,1] #syllable AM instantaneous freq.
                badGuys=np.where(np.sum(phValsD<0,axis=1)>0) #get indexes where syllabe AM od stress AM have negative frequencies
            
                #remove portions of signals where negative frequencies are observed
                phVals=np.delete(phVals,badGuys,0)#
                # shortAMT=np.delete(shortAMT0,badGuys)  
                normAMsigs=np.delete(AMs,badGuys,0)#get normalized AM signals without portion with negative frequencies
                
                #### GET PLVs
                PLV,PLVidx, n_theta1,m_theta2=get_PLV(phVals[:,0],phVals[:,1],np.shape(phVals)[0],np.shape(phVals)[0],maxRelFreqOrd)
                if n_theta1==m_theta2:
                    n_theta1=m_theta2=1
                #get the PLV in different time windows by using the obtained m and n values
                PLV,PLVidx, n_theta1,m_theta2=get_PLV(phVals[:,0],phVals[:,1],np.shape(phVals)[0],np.shape(phVals)[0],[n_theta1,m_theta2])
                
                #### GET surrogate PLV
                surrPLVlst=[]
                for r in range(nSurr):
                    minShift=winStep
                    maxShift=len(AMs[:,1])-minShift
                    myShift=np.random.randint(maxShift)+minShift
                    thisSyllSurrSigs=np.roll(AMs[:,1],int(myShift))
                    
                    surrSyllPhVals=np.angle(hilbert(thisSyllSurrSigs, axis=0))+np.pi
                    
                    surrSyllphValsD=np.diff(np.unwrap(surrSyllPhVals),axis=0)
                    surrSyllbadGuys=np.where(np.sum(surrSyllphValsD<0)>0)
                    
                    surrSyllPhVals=np.delete(surrSyllPhVals,surrSyllbadGuys,0)
                    
                    surrPLV,surrPLVidx, surr_n_theta1,surr_m_theta2=get_PLV(phVals[:,0],surrSyllPhVals,np.shape(phVals)[0],np.shape(phVals)[0],maxRelFreqOrd)
                    if surr_n_theta1==surr_m_theta2:
                        surr_n_theta1=surr_m_theta2=1
                    surrPLV,surrPLVidx, surr_n_theta1,surr_m_theta2=get_PLV(phVals[:,0],phVals[:,1],np.shape(phVals)[0],np.shape(phVals)[0],[surr_n_theta1,surr_m_theta2])
                    
                    surrPLVlst=surrPLVlst+list(surrPLV)
                
                meansurrPLV=np.mean(surrPLVlst)
                stdsurrPLV=np.std(surrPLVlst)
                
                #corect PLV
                PLVn=(PLV[0]-meansurrPLV)/stdsurrPLV
                newFileWriter.writerow([filename,sex,speaker,trial,word,lang,prom,intVerN,PLVn,level])
                
                #### RUN one sample t-test 
                # statsResults=stats.ttest_1samp(surrPLVlst, popmean=PLV[0])
                # pValue=statsResults[1]
                
                # for plvType in [0,1]:
                #     if plvType==0:
                #         plv=PLV[0]
                #     elif plvType==1:
                #         plv=meansurrPLV
                #     newFileWriter.writerow([filename,sex,speaker,trial,word,lang,prom,intVerN,plv,plvType,level])
                
            else:
                continue
        
#%% plot PLV
plt.rcParams['savefig.transparent'] = True
colors=["#DECBE4","#CCEBC5","#B3CDE3"]

csvfile1="plvTest.csv"
csvfile2="plvTest2.csv"
da1=pd.read_csv(path+csvfile1)
da2=pd.read_csv(path+csvfile2)

da=pd.concat([da1, da2])
da=da.reset_index()

da.loc[da.pattern == "HL","stimulus_pattern"]="initial"
da.loc[da.pattern == "LH","stimulus_pattern"]="final"
da.loc[da.level == 0,"level"]="acoustic"
da.loc[da.level == 1,"level"]="larynx"
da.loc[da.language == "fr","sp_language"]="French"
da.loc[da.language == "ge","sp_language"]="German"

da=da[da["PLV"] <= 2]

# da2.loc[da2.pattern == "HL","stimulus_pattern"]="initial"
# da2.loc[da2.pattern == "LH","stimulus_pattern"]="final"
# da2.loc[da2.level == 0,"level"]="acoustic"
# da2.loc[da2.level == 1,"level"]="larynx"
# da1.loc[da1.pattern == "HL","stimulus_pattern"]="initial"
# da1.loc[da1.pattern == "LH","stimulus_pattern"]="final"
# da1.loc[da1.level == 0,"level"]="acoustic"
# da1.loc[da1.level == 1,"level"]="larynx"

sns.catplot(
    data=da, x="sp_language", y="PLV", hue="stimulus_pattern",col="level",row="word",
    kind="box",
    palette=colors,
    #showfliers=False,
    height=4,aspect=0.6,linewidth=1.5,
    # inner="box",
    showfliers=False
    )
        
        