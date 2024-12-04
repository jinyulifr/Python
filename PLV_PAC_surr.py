#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:14:53 2024

@author: jinyuli
"""
#%%
import os
os.chdir('/Volumes/Jinyu LI/PASDCODE/Experiments/exp_SynchroSpeech/analyses/')
# os.chdir('D:/PASDCODE/Experiments/exp_SynchroSpeech/analyses/')
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
from getPhaseToolsM_ext import demodulateAmp
from phaseCouplinFuns import get_PAC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from parselmouth.praat import call
# import tgt

#%% General settings
part=2
# level=0 #0=acoustic level; 1=larynx level

path="/Volumes/Jinyu LI/PASDCODE/Experiments/exp_SynchroSpeech/"
# path="D:/PASDCODE/Experiments/exp_SynchroSpeech/"
outFileName="plvTestLoc.csv"
outTxtFile=path+"analyses/"+outFileName #Name of output file
# outTxtFile=path+"analyses/"+outFileName #Name of output file
newFile=open(outTxtFile,'w', newline='') # open output file for writing
newFileWriter = csv.writer(newFile) # open output file for writing
# write headers line to output file
newFileWriter.writerow(["filename","sex","spID","trial","word","language","pattern","level",
                        "PLVword","PLVwordn","PLVall","PLValln","PACword","PACwordn","PACall","PACalln",
                        "mnRatioWord","mnRatioAll","stdsurrPLVword","stdsurrPLVall","stdPACword","stdPACall"])

sr = 16000

# frontiers for AM filters
meanRecorDur=12
# segN=112
# VowN=28
# AccN=14
segN=144
VowN=36
AccN=18
segRate=segN/meanRecorDur #segment rate
syllRate=VowN/meanRecorDur #syllable rate
stressRate=AccN/meanRecorDur #accent rate
stressCut=(stressRate+syllRate)/2 #frontier between stress and syllables
syllCut=syllRate+(syllRate-stressCut) #frontier between syllables and segment
segCut=segRate+(segRate-syllCut) #upper limit of segment AM band
Fcor=np.array([0.1,0.9,stressCut,syllCut,segCut]) # vector of frontiers for AM filters
# Fcor=np.array([0.1,0.9,2.5,12,40]) # vector of frontiers for AM filters

nSurr=100

NSamp=1000 #sampling frequency of the AM signals
# winLenS=0 # window size for PLV computation (if word level)
winLenS=0.6 # window size for PLV computation (if recording level)
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
    for wav_file in glob.glob(path+"/recordings/part2_test/"+"*.wav"):
        
        #### activate when plot signals
        # level=0
        # filename = "F_12_6_mama_LH.wav"
        # wav_file=path+"recordings/part2/"+filename
        ####
        
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
        
        ############
        
        d0=d
                        
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
                
        ########################################################################
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
        ########################################################################
        
        ##### demodulation and get phase values over time
        stressPHI=np.angle(hilbert(demodulateAmp(tmpSigs[:,0])))
        syllPHI=np.angle(hilbert(demodulateAmp(tmpSigs[:,1])))
      
        ### plot signal
        # dT=(np.arange(np.size(d0))+1)/sr
        # fig, ax = plt.subplots(3, 1, sharex=True, sharey=False,figsize=(10, 15))
        # ax[0].plot(shortAMT0,demodulateAmp(tmpSigs[:,0]))
        # ax[1].plot(shortAMT0,demodulateAmp(tmpSigs[:,1]))
        # # ax[2].plot(AMampT,stats.zscore(AMs[:,1]+AMs[:,0])+2)
        # ax[2].plot(dT,stats.zscore(d0))
        ##################################################################
        
        #### GET PHASE VALUES OVER TIME
        # phVals=np.angle(hilbert(AMs, axis=0))+np.pi #phase vals via hilbert transform
        # phVals=np.angle(hilbert(tmpSigs, axis=0))+np.pi #phase vals via hilbert transform
        # syllPh=phVals[:,1]#syllable AMphase
        
        #AM signals instantaneous frequency over time    
        # phValsD=np.diff(np.unwrap(phVals),axis=0)
        stressphValsD=np.diff(np.unwrap(stressPHI),axis=0)
        syllphValsD=np.diff(np.unwrap(syllPHI),axis=0)
        # syllPhD=phValsD[:,1] #syllable AM instantaneous freq.
        # stressbadGuys=np.where(np.sum(stressphValsD<0)>0)
        stressbadGuys=np.where(stressphValsD<0)[0]
        # syllbadGuys=np.where(np.sum(syllphValsD<0)>0)
        syllbadGuys=np.where(syllphValsD<0)[0]
        badGuys=np.union1d(stressbadGuys, syllbadGuys)
        # badGuys=np.where(np.sum(phValsD<0)>0) #get indexes where syllabe AM od stress AM have negative frequencies
        # badGuys=np.where(np.sum(phValsD<0,axis=1)>0) #get indexes where syllabe AM od stress AM have negative frequencies
    
        #remove portions of signals where negative frequencies are observed
        stressPHI=np.delete(stressPHI,badGuys,0)
        syllPHI=np.delete(syllPHI,badGuys,0)
        shortAMT=np.delete(shortAMT0,badGuys)  
        # normAMsigs=np.delete(phVals,badGuys,0)#get normalized AM signals without portion with negative frequencies
        #################################################################
        
        #### get surrogate phases of syllAM
        minShift=int(NSamp/1.5)
        maxShift=len(syllPHI)-minShift
        surrSyllPHIArry=np.zeros((np.shape(syllPHI)[0],100))
        for r1 in range(nSurr):
            myShift=np.random.randint(maxShift)+minShift
            surrSyllPHI=np.roll(syllPHI,int(myShift))
            surrSyllPHIArry[:,r1]=surrSyllPHI
        
        #### get amplitude enveloppe of syllabic modulation and get surrogate syllabic modulations
        syllAMamp=np.abs(hilbert(tmpSigs[:,1]))
        syllAMamp=np.delete(syllAMamp,badGuys,0)
        
        # fig, ax = plt.subplots(2, 1, sharex=True, sharey=False,figsize=(10, 15))
        # ax[0].plot(stressPHI)
        # ax[1].plot(syllPHI)
        
        surrSyllAMampArry=np.zeros((np.shape(syllAMamp)[0],100))
        maxShift=len(syllAMamp)-minShift
        for r2 in range(nSurr):
            myShift=np.random.randint(maxShift)+minShift
            surrSyllAMamp=np.roll(syllAMamp,int(myShift))
            surrSyllAMampArry[:,r2]=surrSyllAMamp
            
        ################################################################
        
        #### get the portion of recording to analyze
        txtgrd=call("Read from file", os.path.splitext(wav_file)[0]+".textgrid")
        noi=call(txtgrd,"Get number of intervals",1)
        
        intNlist=[]
        for intN in range(1,noi+1):
            thisLab=call(txtgrd,"Get label of interval", 1, intN)
            if thisLab=="0":
                intNlist=intNlist+[intN]
            else:
                continue
        
        thisStartT=call(txtgrd,"Get start time of interval",1,intNlist[0])
        thisEndT=call(txtgrd,"Get end time of interval",1,intNlist[-1])   
        
        # thisStartWav=np.floor(thisStartT*sr).astype(int)
        # thisEndWav=np.floor(thisEndT*sr).astype(int)
        
        # d0=d[int(np.floor(thisStartWav)):int(np.floor(thisEndWav))]
        
        shortIdxs=np.where( (shortAMT>=thisStartT) & (shortAMT<=thisEndT) )[0] #get index of this word
        shortStressPhVals=stressPHI[shortIdxs]
        shortSyllPhVals=syllPHI[shortIdxs]
        shortSyllAMamp=syllAMamp[shortIdxs]
        
        shortSurrSyllAMamp=surrSyllAMampArry[shortIdxs,: ]
        shortSurrSyllPHI=surrSyllPHIArry[shortIdxs,: ]
    
        #### GET PAC (with amplitude enveloppe of syllabic modulation) and surrogate PAC
        PAC_all=get_PAC(shortStressPhVals, shortSyllAMamp, 0, 0)
        
        surrPAClst_all=[]
        for surrPACN_all in range(nSurr):
            surrPAC_all=get_PAC(shortStressPhVals, shortSurrSyllAMamp[:,surrPACN_all], 0, 0)
            surrPAClst_all=surrPAClst_all+[surrPAC_all[0][0]]
        
        meansurrPAC_all=np.mean(surrPAClst_all)
        stdsurrPAC_all=np.std(surrPAClst_all)
        
        #corect PAC
        PACn_all=(PAC_all[0][0]-meansurrPAC_all)/stdsurrPAC_all
        
        #### Get PLV of the whole recording (using defined winLen) and surrogate PLV
        PLV_all,PLVidx_all, n_theta1_all,m_theta2_all=get_PLV(shortStressPhVals,shortSyllPhVals,np.shape(shortStressPhVals)[0],np.shape(shortStressPhVals)[0],maxRelFreqOrd)
        if n_theta1_all==m_theta2_all:
            n_theta1_all=m_theta2_all=1
        #get the PLV in different time windows by using the obtained m and n values
        PLV_all,PLVidx_all, n_theta1_all,m_theta2_all=get_PLV(shortStressPhVals,shortSyllPhVals,winLen,winStep,[n_theta1_all,m_theta2_all])
        PLV_all=np.mean(PLV_all)
        mnRatio_all=np.mean(m_theta2_all)/np.mean(n_theta1_all)
        
        surrPLVlst_all=[]
        for surPLVN_all in range(nSurr):
            surrPLV_all,surrPLVidx_all, surr_n_theta1_all,surr_m_theta2_all=get_PLV(shortStressPhVals,shortSurrSyllPHI[:,surPLVN_all],np.shape(shortStressPhVals)[0],np.shape(shortStressPhVals)[0],maxRelFreqOrd)
            if surr_n_theta1_all==surr_m_theta2_all:
                surr_n_theta1_all=surr_m_theta2_all=1
            surrPLV_all,surrPLVidx_all, surr_n_theta1_all,surr_m_theta2_all=get_PLV(shortStressPhVals,shortSurrSyllPHI[:,surPLVN_all],winLen,winStep,[surr_n_theta1_all,surr_m_theta2_all])
            surrPLV_all=np.mean(surrPLV_all)
            
            surrPLVlst_all=surrPLVlst_all+[surrPLV_all]
        
        meansurrPLV_all=np.mean(surrPLVlst_all)
        stdsurrPLV_all=np.std(surrPLVlst_all)
        
        #corect PLV
        PLVn_all=(PLV_all-meansurrPLV_all)/stdsurrPLV_all
        ################################################################
        
        #### get word portions
        # AM0array=np.empty([1,600])
        # AM1array=np.empty([1,600])
        # intVerN=0
        for intN in intNlist:
            print("Processing interval {}...".format(intN))
            # thisLab=call(txtgrd,"Get label of interval", 1, intN)
            # if thisLab=='0':
            # intVerN=intVerN+1
            thisT1=call(txtgrd,"Get start time of interval",1,intN)
            thisT2=call(txtgrd,"Get end time of interval",1,intN)
        
            # thisStartFrWav=np.floor(thisT1*sr).astype(int)
            # thisEndFrWav=np.floor(thisT2*sr).astype(int)
            # d0=d[int(np.floor(thisStartFrWav)):int(np.floor(thisEndFrWav))]
            
            thisWordIdxs=np.where( (shortAMT>=thisT1) & (shortAMT<=thisT2) )[0] #get index of this word
            thisStressPhVals=stressPHI[thisWordIdxs]
            thisSyllPhVals=syllPHI[thisWordIdxs]
            thisSyllAMamp=syllAMamp[thisWordIdxs]
            
            thisSurrSyllAMamp=surrSyllAMampArry[thisWordIdxs,: ]
            thisSurrSyllPHI=surrSyllPHIArry[thisWordIdxs,: ]
            
            #### GET PAC and surrogate PAC
            PAC=get_PAC(thisStressPhVals, thisSyllAMamp, 0, 0)
            
            surrPAClst=[]
            for surrPACN in range(nSurr):
                surrPAC=get_PAC(thisStressPhVals, thisSurrSyllAMamp[:,surrPACN], 0, 0)
                surrPAClst=surrPAClst+[surrPAC[0][0]]
            
            meansurrPAC=np.mean(surrPAClst)
            stdsurrPAC=np.std(surrPAClst)
            
            #corect PAC
            PACn=(PAC[0][0]-meansurrPAC)/stdsurrPAC
            
            #### GET PLV and surrogate PLV
            PLV,PLVidx, n_theta1,m_theta2=get_PLV(thisStressPhVals,thisSyllPhVals,np.shape(thisStressPhVals)[0],np.shape(thisStressPhVals)[0],maxRelFreqOrd)
            if n_theta1==m_theta2:
                n_theta1=m_theta2=1
            #get the PLV in different time windows by using the obtained m and n values
            PLV,PLVidx, n_theta1,m_theta2=get_PLV(thisStressPhVals,thisSyllPhVals,winLen,winStep,[n_theta1,m_theta2])
            PLV=np.mean(PLV)
            mnRatio=np.mean(m_theta2)/np.mean(n_theta1)
            
            surrPLVlst=[]
            for surPLVN in range(nSurr):
                surrPLV,surrPLVidx, surr_n_theta1,surr_m_theta2=get_PLV(thisStressPhVals,thisSurrSyllPHI[:,surPLVN_all],np.shape(thisStressPhVals)[0],np.shape(thisStressPhVals)[0],maxRelFreqOrd)
                if surr_n_theta1==surr_m_theta2:
                    surr_n_theta1=surr_m_theta2=1
                surrPLV,surrPLVidx, surr_n_theta1,surr_m_theta2=get_PLV(thisStressPhVals,thisSurrSyllPHI[:,surPLVN_all],winLen,winStep,[surr_n_theta1,surr_m_theta2])
                surrPLV=np.mean(surrPLV)
                
                surrPLVlst=surrPLVlst+[surrPLV]
            
            meansurrPLV=np.mean(surrPLVlst)
            stdsurrPLV=np.std(surrPLVlst)
            
            #corect PLV
            PLVn=(PLV-meansurrPLV)/stdsurrPLV
            
            newFileWriter.writerow([filename,sex,sex+str(int(speaker)),trial,word,lang,prom,level,
                                    PLV,PLVn,PLV_all,PLVn_all,PAC,PACn,PAC_all,PACn_all,
                                    mnRatio,mnRatio_all,stdsurrPLV,stdsurrPLV_all,stdsurrPAC,stdsurrPAC_all])
            newFile.flush()
newFile.close()
    
#%% plot PLV
plt.rcParams['savefig.transparent'] = True
colors=["#DECBE4","#CCEBC5","#B3CDE3"]

filename="plvTest.csv"
da=pd.read_csv(path+"analyses/"+filename)

da.loc[da.pattern == "HL","stimulus_pattern"]="initial"
da.loc[da.pattern == "LH","stimulus_pattern"]="final"
da.loc[da.level == 0,"level"]="acoustic"
da.loc[da.level == 1,"level"]="larynx"
da.loc[da.language == "fr","sp_language"]="French"
da.loc[da.language == "ge","sp_language"]="German"

# da=da[da["PLV"] <= 2]


sns.catplot(
    data=da, x="sp_language", y="PLValln", hue="stimulus_pattern",col="level",#row="word",
    kind="box",
    palette=colors,
    #showfliers=False,
    height=4,aspect=0.8,linewidth=1.5,
    # inner="box",
    showfliers=False
    )

#dansity plot
sns.kdeplot(da["mnRatio"], fill=True, color="blue",alpha=0.3)



        