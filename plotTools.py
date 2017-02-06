#Nicole Neubarth: May 14, 2015.
#SciPy reference:http://www.tau.ac.il/~kineret/amit/scipy_tutorial/
#Neural data analysis reference: MATLAB for neuroscientists by Wallisch et al.
import seaborn as sns
import scipy.signal as signal
import pylab
import numpy as np
import traceRoutines as tm
from scipy.optimize import leastsq
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import glob
import ephusRoutines as ephus
#sns.set(style="white") 

def prepper():
    '''
    The prepper function finds all xsg files in a directory using glob and uses the parseXSG function from ephusRoutines to parse xsg files.
    Then it uses the epochComprehension function to separate xsg files by type and prints a breakdown of the number of traces per type. 
    param: none
    returns: dictionary with category names as keys and lists of xsg files in that category as values
    '''
    files = glob.glob('*.xsg')
    xsgs = [ephus.parseXSG(f) for f in files]
    catdict = epochComprehension(xsgs)
    return catdict

def invertSignal(files):
    '''
    function for inverting ephus traces for improved spike detection.
    This can be useful if the peak of the spike is a signal minimum 
    rather than a maximum, e.g.
    param: files is a list of ephus traces
    returns: new list of traces with ephys channel inverted, does not modify files
    '''
    filescopy = files[:]
    for f in filescopy:
        f['ephys']['chan0'] = -(f['ephys']['chan0'])
    return filescopy


def epochComprehension(xsgs):
    '''The epochComprehension function sorts a batch of ephus traces into stimulus categories.
    It does this by reading the epoch number of the file. Prints a breakdown of number of traces by type.
    param: files, batch of parsed xsg files to be sorted.
    returns: dictionary with category names as keys and lists of xsg files in that category as values
    '''
    #change epoch numbers as needed here
    #use list comprehensions to parse xsg files into categories
    ephys = [t for t in xsgs if t['epoch'] == 0]
    estim = [t for t in xsgs if t['epoch'] == 1]
    grid = [t for t in xsgs if t['epoch'] == 2]
    chr2 = [t for t in xsgs if t['epoch'] == 3]
    wnforce = [t for t in xsgs if t['epoch'] == 4]
    vfrey = [t for t in xsgs if t['epoch'] == 5]
    wnlength = [t for t in xsgs if t['epoch'] == 6]
    chirpforce = [t for t in xsgs if t['epoch'] == 7]
    chirplength = [t for t in xsgs if t['epoch'] == 8]
    iforce = [t for t in xsgs if t['epoch'] == 9]
    ilength = [t for t in xsgs if t['epoch'] == 10]
    #notify user of the number of traces per category 
    print 'Ephys only traces: ' + str(len(ephys))
    print 'Electrical stim traces: ' + str(len(estim))
    print 'Grid indent traces: ' + str(len(grid))
    print 'ChR2 stim traces: ' + str(len(chr2))
    print 'White noise (force) traces: ' + str(len(wnforce))
    print 'von Frey traces: ' + str(len(vfrey))
    print 'White noise (length) traces: ' + str(len(wnlength))
    print 'Chirp (force) traces: ' + str(len(chirpforce))
    print 'Chirp (length) traces: ' + str(len(chirplength))
    print 'Indentation (force) traces: ' + str(len(iforce))
    print 'Indentation (length) traces: ' + str(len(ilength))
    #return a list of lists containing traces divided into categories
    return {'ephys': ephys,'estim': estim,'grid': grid,'chr2': chr2,'wnforce': wnforce,'vfrey': vfrey, 'wnlength': wnlength,'chirpforce': chirpforce,'chirplength': chirplength,'iforce': iforce,'ilength': ilength}
    
def getCV(files,distance,threshold=500,selectedtraces=None):
    '''
    getCV plots a group of CV measurements, finds the electrically evoked spike, and calculates the latency and CV for the neuron
    param: files to analyze, threshold is the minimum peak-to-peak amplitude of signal modulation for a signal to be considered non-empty(default is 500 pA; prevents detection of spikes in empty traces).
    selectedtraces is a list of ints indicating the acquisition numbers to be analyzed (default is that all will be analyzed),
    distance is a float measurement of the distance between the stimulus and the measurement point (in meters).
    prints mean latency with standard devation and CV (in m/sec)
    return: list with elements mean latency and CV
    '''
    pylab.plt.figure(figsize=(15,30))
    #subplot index
    i = 1
    spiketimes = {}
    for f in files:
        #make sure this is a real file by checking that ephys trace exists
        #and only analyze selected traces
        if f['ephys'] is not None and ( int(f['acquisitionNumber']) in selectedtraces or selectedtraces is None):
            pylab.plt.subplot(math.ceil(len(files))/5.0, 5, i)
            #increment plot index
            i += 1
            #filter signal to remove low frequency noise
            b, a = signal.butter(8, .025, 'high')
            f_trace = signal.filtfilt(b, a, f['ephys']['chan0'][int(.095*f['sampleRate']):int(.11*f['sampleRate'])], padlen=100)
            #create time vector
            t = np.linspace(.095, .11, num=.015*f['sampleRate'])
            #create plot
            pylab.plt.title('Trial Number ' + f['acquisitionNumber'])

            pylab.plt.plot(t,f_trace)
            #pylab.plt.ylabel('Current (pA)')
            #pylab.plt.xlabel('seconds')
            #only analyze segment after artifact for spikes
            f_trace = f_trace[int(.006*f['sampleRate']):int(.013*f['sampleRate'])]
            t = np.linspace(.101, .108, num=.007*f['sampleRate'])
            #detect spikes in trace and mark with vertical black line
            #choose threshold
            #modified so threshold works for traces with no spikes
            if f_trace.max()-f_trace.min() >threshold:
                hp_thresh = f_trace.max() *.60
            else:
                hp_thresh = f_trace.max() * 1.1
            spike_boundaries = zip(tm.findLevels(f_trace, hp_thresh, mode='rising')[0], tm.findLevels(f_trace, hp_thresh, mode='falling')[0])
            try:
                spike_peaks = [np.argmax(f_trace[s[0]:s[1]])+s[0] for s in spike_boundaries]
                pylab.plt.vlines(t[spike_peaks], f_trace.max()*2 ,f_trace.max()*1.1,color = 'k')
            except ValueError:
                print 'unable to process acquisition number ' + str(f['acquisitionNumber'])
            pylab.plt.title(f['acquisitionNumber']);
            pylab.plt.ylim(-1000,1000)
            sns.despine()
            pylab.plt.axis('off')
            #save spike times
            spiketimes[int(f['acquisitionNumber'])] = t[spike_peaks]
    #get the first spike time for each trace and calculate the latency
    latencies = []
    for a,s in spiketimes.items():
        if len(s) > 0:
            latencies.append(s[0] - .10)
    print 'The mean latency is ' + str(np.mean(latencies)) + ' +/- ' + str(np.std(latencies))
    print 'The CV is ' + str(distance/np.mean(latencies))
    #return the mean latency and CV
    return [np.mean(latencies), distance/np.mean(latencies) ]



def plotTraces(files,force=False,length=False,threshold=500):
    '''
    plotTraces is a function for plotting ephus traces imported into python using the 
    parseXSG function.
    param: files: list of files, force: boolean indicating presence of force traces,
    length: boolean indicating presence of length traces, threshold: int indicating the 
    minimum peak-to-peak amplitude of a signal to be considered a spike (default is 500 pA).
    return: spiketimes: tuple(dict of acquisition numbers and spike times for files, force amplitude dictionary,length amplitude dictionary)
    '''
    
    #create dictionary with acquisition number as keys, spike times as values
    #force amps as values
    #and length amps as values
    spiketimes = {}
    force_amps = {}
    length_amps = {}
    for f in files:
        #make sure this is a real file by checking that ephys trace exists
        if f['ephys'] is not None:
            pylab.plt.figure(figsize=(15,2))
            #filter signal to remove low frequency noise
            b, a = signal.butter(8, .025, 'high')
            f_trace = signal.filtfilt(b, a, f['ephys']['chan0'], padlen=100)
            #create time vector
            t = np.linspace(0, len(f['ephys']['chan0']) / f['sampleRate'], num=len(f['ephys']['chan0']))
            #create plot
            pylab.plt.title('Trial Number ' + f['acquisitionNumber'])
            pylab.plt.plot(t,f_trace)
            pylab.plt.ylabel('Current (pA)')
            pylab.plt.xlabel('seconds')
            #detect spikes in trace and mark with vertical black line
            #choose threshold
            #modified so threshold works for traces with no spikes
            if f_trace.max()-f_trace.min() >threshold:
                hp_thresh = f_trace.max() *.60
            else:
                hp_thresh = f_trace.max() * 1.1
            spike_boundaries = zip(tm.findLevels(f_trace, hp_thresh, mode='rising')[0], tm.findLevels(f_trace, hp_thresh, mode='falling')[0])
            spike_peaks = [np.argmax(f_trace[s[0]:s[1]])+s[0] for s in spike_boundaries]
            pylab.plt.title(f['acquisitionNumber']);
            pylab.plt.vlines(t[spike_peaks], f_trace.max()*1.6 ,f_trace.max()*1.2,color = 'k')
            pylab.plt.ylabel('Current (pA)')
            sns.despine()
            #save spike times
            spiketimes[int(f['acquisitionNumber'])] = t[spike_peaks]
            #plot force traces if force traces are present
            if force:
                pylab.plt.figure(figsize=(15,2))
                t = np.linspace(0, len(f['acquirer']['Force Out']) / f['sampleRate'], num=len(f['acquirer']['Force Out']))
                pylab.plt.plot(t,f['acquirer']['Force Out'],'r')
                pylab.plt.ylabel('Force (V)')
                pylab.plt.tick_params(\
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off') # labels along the bottom edge are off
                pylab.plt.axis([0, max(t),f['acquirer']['Force Out'].min()-.2,f['acquirer']['Force Out'].max()+.2])
                #add maximum amplitude to plot
                maxforce = f['acquirer']['Force Out'][int(.050*f['sampleRate']):].max() - f['acquirer']['Force Out'][int(.050*f['sampleRate'])]
                pylab.plt.text(.2,f['acquirer']['Force Out'].max()+.02,'Max amplitude = ' + str(round(maxforce*50.9,3)) + 'mN')
                #save the max amplitude for this acquisition number
                force_amps[int(f['acquisitionNumber'])] = maxforce*50.9
            if length:
                pylab.plt.figure(figsize=(15,2))
                t = np.linspace(0, len(f['acquirer']['Length Out']) / f['sampleRate'], num=len(f['acquirer']['Length Out']))
                pylab.plt.plot(t,f['acquirer']['Length Out'],'r')
                pylab.plt.ylabel('Displacement (V)')
                pylab.plt.tick_params(\
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off') # labels along the bottom edge are off
                pylab.plt.axis([0, max(t),f['acquirer']['Length Out'].min()-.2,f['acquirer']['Length Out'].max()+.2])
                #add maximum amplitude to plot using conversion factor (.64mm 50.9mN)
                signal_cut =  f['acquirer']['Length Out'][int(.050*f['sampleRate']):]
                maxlength = signal_cut.max() - signal_cut[0]
                pylab.plt.text(.2,f['acquirer']['Length Out'].max()+.05,'Max amplitude = ' + str(round(maxlength*.64,3)) + 'mm')
                #save the max amplitude for this acquisition number
                length_amps[int(f['acquisitionNumber'])] = maxlength*.64
    return (spiketimes, force_amps, length_amps)
            
def spikeAnalysis(spiketimes):
    '''
    param: spiketimes is dictionary of acquisition number, spike times
    returns: dictionary with acquition number: (int number of spikes, list of instantaneous frequency of spikes over time)
    '''   
    spikeanalysis = {}
    for trace in spiketimes.keys():
        numspikes = len(spiketimes[trace])
        ISIs = []
        if numspikes > 1:
            for i in range(1,numspikes):
                ISI = spiketimes[trace][i] - spiketimes[trace][i-1]
                ISIs.append(ISI)
        spikeanalysis[trace] = (numspikes,ISIs)
    return spikeanalysis

def getSpikeFreq(ISIs):
    '''
    param: ISIs is a list of ISIs in seconds
    return: list of frequencies
    '''
    freqs = []
    for ISI in ISIs:
        freqs.append(1.0/ISI)
    return freqs
    
def intensityPlot(spikeanalysis,amps,tracenumbers,amptype):
    '''
    plots the number of spikes vs stimulus amplitude
    plots the instantaneous frequency of the response over time for different amplitudes
    plots the max instantaneous frequency of the response as a function of amplitude
    param: dictionary of spike information for different traces, which contains as values 
    tuples of number of spikes (int) and list of ISIs for a single trace, amps is a dictionary 
    with acquisition numbers as keys and stimulus amplitudes as values, tracenumbers allows the user
    to select which trace numbers are relevant and is a list of ints
    amptype a boolean 1 for force and 0 for length
    '''
    numspikesplot = []
    freqlist = []
    if amptype:
        ampstr = 'mN'
    else:
        ampstr = 'mm'
    for trace in spikeanalysis.keys():
        if trace not in tracenumbers:
            continue
        else:
            #numspikesplot is a list with tuples of (amp,numspikes)
            numspikesplot.append((amps[trace],spikeanalysis[trace][0]))
            #get numpy array of spike frequencies
            spikefreqs = np.array(getSpikeFreq(spikeanalysis[trace][1]))
            try:
            #freqlist contains list of spike frequencies, maximum spike frequency, and amp
            #of the trace
                freqlist.append((spikefreqs,spikefreqs.max(),amps[trace]))
            except ValueError:
                freqlist.append((0,0,amps[trace]))
            #Plot 1. instantaneous frequency over time for different amplitudes
            pylab.plt.plot(spikefreqs,marker='o', linestyle='--', label = [str(round(amps[trace],2)) + ampstr])
            pylab.plt.title('Instantaneous firing frequency over time')
            pylab.plt.ylabel('Firing Frequency (Hz)')
            pylab.plt.xlabel('Time (no units)')
            
    pylab.plt.legend()
    #sort the numspikesplot according to amplitude
    numspikesplot.sort(key=lambda tup: tup[0])
    amps_val = [x[0] for x in numspikesplot]
    numspikes_val = [x[1] for x in numspikesplot]
    #Plot 2.the number of spikes as a fxn of amplitude
    pylab.plt.figure()
    pylab.plt.scatter(amps_val,numspikes_val,marker='o')
    pylab.plt.title('Number of spikes as a function of stimulus amplitude')
    pylab.plt.ylabel('Number of spikes')
    pylab.plt.xlabel('Amplitude (' + ampstr + ')')
    #least-squares sigmoid fit to the data: convert values to numpy arrays
    numspikes_val = np.array(numspikes_val)
    amps_val = np.array(amps_val)
    sigfit = sigmoidFit(amps_val,numspikes_val)
    x_sig = np.linspace(min(amps_val)-2,max(amps_val)+2,num=100)
    pylab.plt.plot(x_sig,sigmoid(sigfit,x_sig),linestyle='--',color='r')
    pylab.plt.text(2,max(amps_val)+3,'Sigmoid fit (x0,y0,L,k): ' + str(sigfit))
    #Plot 3. Max instantaneous frequency as a function of amplitude
    freqlist.sort(key=lambda tup: tup[2])
    freq_val = [x[1] for x in freqlist]
    amps_val = [x[2] for x in freqlist]
    pylab.plt.figure()
    pylab.plt.plot(amps_val,freq_val,marker='o')
    pylab.plt.title('Maximum instantaneous frequency of discharge as a function of amplitude')
    pylab.plt.ylabel('Maximum Instantaneous Frequency (Hz)')
    pylab.plt.xlabel('Amplitude (' + ampstr + ')')

def sigmoid(p,x):
    '''
    params: p, list of values for the sigmoid function
    L, L + y0 is y-value of top asymptote
    x0, x-value of curve's midpoint
    k, steepness of curve
    y0, y-value of the bottom asymptote
    x, numpy array of x values from the data
    returns: y values of the fit for the data
    '''
    #L, L + y0 is y-value of top asymptote
    #x0, x-value of curve's midpoint
    #k, steepness of curve
    #y0, y-value of the bottom asymptote
    x0,y0,L,k=p
    y = L / (1 + np.exp(-k*(x-x0))) + y0
    return y

def residuals(p, x, y):  
    ''' 
    params: p, list of values for sigmoid function
    x and y are numpy arrays of x and y values respectively
    returns: residuals for use with leastsq function
    '''
    return y - sigmoid(p,x) 

def sigmoidFit(x,y):
    '''
    params: x and y are numpy arrays of x and y values respectively
    returns: the parameters p for use with the sigmoid function as a numpy array
    '''
    start_sigmoid=(np.median(x),min(y),max(y),1.0)
    solution= leastsq(residuals,start_sigmoid,args=(x,y)) 
    return solution[0]
##Section 2. frequency chirp analysis
    
def calculatePhase(file,spiketime):
    '''
    params:file from ephus, spiketime is a single spike time for that file
    returns: phase in radians
    '''
    phase = math.asin((file['acquirer']['Length Out'][spiketime]-min(file['acquirer']['Length Out']))/(max(file['acquirer']['Length Out'])-min(file['acquirer']['Length Out'])))
    return phase
    
import math
def calculatePhase(file,spiketime):
    '''
    params:file from ephus, spiketime is a single spike time for that file
    returns: phase in radians
    '''
    phase = math.asin((file['acquirer']['Length Out'][spiketime]-min(file['acquirer']['Length Out']))/(max(file['acquirer']['Length Out'])-min(file['acquirer']['Length Out'])))
    return phase
    
def getFrequency(spiketimes):
    '''
    param: spiketime is a single spike times for a file
    returns: frequency assigment for the spike time
    '''
    #known frequencies from stimulus file. change when change stimulus
    freqs=[0.5,.65,.84,1.1,1.4,1.86,2.43,3.16,4.12, \
       5.36,6.98,9.09,11.83,15.41,20.06,26.11,33.99,44.25,57.61,75] 
    #assign spike times to a frequency based on time
    if spiketimes > 19.6:
        frequency_assignment = freqs[-1]
    elif spiketimes < 19.6 and spiketimes > 19.3:
        frequency_assignment = freqs[-2]
    elif spiketimes < 19.3 and spiketimes > 18.9:
        frequency_assignment = freqs[-3]
    elif spiketimes < 18.9 and spiketimes > 18.5:
        frequency_assignment = freqs[-4]
    elif spiketimes < 18.5 and spiketimes > 18.0:
        frequency_assignment = freqs[-5]
    elif spiketimes < 18.0 and spiketimes > 17.5:
        frequency_assignment = freqs[-6]
    elif spiketimes < 17.5 and spiketimes > 16.7:
        frequency_assignment = freqs[-7]
    elif spiketimes < 16.7 and spiketimes > 16.0:
        frequency_assignment = freqs[-8]
    elif spiketimes < 16.0 and spiketimes > 14.9:
        frequency_assignment = freqs[-9]
    elif spiketimes < 14.9 and spiketimes > 13.6:
        frequency_assignment = freqs[-10]
    elif spiketimes < 13.6 and spiketimes > 12.0:
        frequency_assignment = freqs[-11]
    elif spiketimes < 12.0 and spiketimes > 10.0:
        frequency_assignment = freqs[-12]
    elif spiketimes < 10.0 and spiketimes > 7.5:
        frequency_assignment = freqs[-13]
    elif spiketimes < 7.5 and spiketimes > 4.15 :
        frequency_assignment = freqs[-14]
    elif spiketimes < 4.15 and spiketimes > 0 :
        frequency_assignment = freqs[-15]
    else:
        frequency_assignment = 0
    return frequency_assignment
    
def getPhaseAndFrequency(file,spiketimes):
    '''
    params: file from ephus, spiketimes is a list of spike times for that file
    return: spikephases and spikefreqs, lists of phases and freqs for this file
    '''
    spikephases = []
    spikefreqs = []
    for st in spiketimes:
        phase = calculatePhase(file,st)
        freq = getFrequency(st)
        spikephases.append(phase)
        spikefreqs.append(freq)
    return spikephases,spikefreqs

def plotPhaseAndFrequency(files,spiketimes,force_amps,length_amps,selectedtraces):
    '''
    plots (TODO:--phase and--) frequency information against stimulus amplitude
    '''
    pylab.plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    pylab.plt.title('Threshold amplitude')
    pylab.plt.ylabel('Amplitude')
    pylab.plt.xlabel('Frequency (Hz)')

    for f in files:
        if int(f['acquisitionNumber']) in selectedtraces:
            spikephases,spikefreqs = getPhaseAndFrequency(f,spiketimes[int(f['acquisitionNumber'])])
            pylab.plt.scatter(spikefreqs,np.ones(len(spikefreqs))* length_amps[int(f['acquisitionNumber'])],alpha=0.1)

#Section 3. Neural data analysis

def binSpikeTrain(file,spiketime,binwidth):
    '''
    params:file (a single ephus file), spiketime (a list of spike times for a particular trace),
    binwidth( float of bin size in seconds)
    returns: hist (a histogram of spike counts for each bin), bin_edges (edges)
    bins spike times according to the user-provided bin width in seconds
    '''
    #create a histogram of spike counts per time bin
    binarray = np.arange(0,len(file['ephys']['chan0']) / file['sampleRate'],binwidth)
    hist, bin_edges = np.histogram(spiketime,bins=binarray)
    #pylab.plt.plot(bin_edges[0:len(hist)],hist)
    return hist, bin_edges

def convolveKernel(files,spiketimes,sigma,selectedtraces):
    '''
    params: files (ephus file list), 
    spiketimes (a dictionary of acqusitionnumber,spike times for a file),
    sigma (standard deviation of gaussian kernel in seconds)
    
    convolves spike time histogram with gaussian kernel of user-defined width (sigma)
    '''
    #get the kernel
    edges = np.arange(-3*sigma,3*sigma,.001)
    kernel = matplotlib.mlab.normpdf(edges,0,sigma)
    kernel = kernel*.001
    pylab.plt.figure()
    pylab.plt.plot(edges,kernel)
    pylab.plt.figure()
    for f in files:
        if int(f['acquisitionNumber']) in selectedtraces:
        #get the hist data for each file
            hist,bin_edges = binSpikeTrain(f,spiketimes[int(f['acquisitionNumber'])],.001)
            s = np.convolve(kernel, hist, mode='full')
            #plot each spike density function on the same plot
            pylab.plt.plot(s)
            try:
                savg = np.vstack((savg,s))
            except NameError:
                savg = np.array(s)
    #get the average spike density function for this group of files
    savg = np.mean(savg, axis=0)
    #plot the resulting average
    pylab.plt.figure()
    pylab.plt.plot(savg)
        

def spikeTrigAvg(files,spiketimes,selectedtraces,window):
    '''
    Spike-triggered average for a particular group of files using spike times.
    '''
    for f in files:
         
        if int(f['acquisitionNumber']) in selectedtraces:

            for st in spiketimes[int(f['acquisitionNumber'])]:
                #get start and stop times in number of points

                timestart = (st - window) * f['sampleRate']
                timestop = (st + window) * f['sampleRate']

                timewindow = range(int(timestart),int(timestop),1)
                lengthconfactor =.64
                perispikestim = lengthconfactor* (f['acquirer']['Length Out'][timewindow] - np.mean(f['acquirer']['Length Out'][timewindow]))


                pylab.plt.plot(perispikestim)

                try:
                    avgstim = np.vstack((avgstim,perispikestim))
                except NameError:
                    avgstim = np.array(perispikestim)
    #get the average perispike stimulus for this group of files
    
    avgstim = np.mean(avgstim, axis=0)
    #plot the resulting average
    pylab.plt.figure()
    t = np.linspace(-window,window,num=f['sampleRate']*2*window)
    pylab.plt.plot(t,avgstim)

        
    
                
            
    