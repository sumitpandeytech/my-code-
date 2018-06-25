
import numpy as np
import serial
import matplotlib.pyplot as plt
# start taking reading from arduino 
ser = serial.Serial('COM3', 115200)
s=ser.read(70000) #reading up to 100 bytes

print(s)
# close the port 
ser.close()

# decode the byte 

a = s.decode('unicode_escape').encode('utf-8')
# change the str into list y using split 
b = a.split()
# change the list inot array
c = np.array(b)
# this array has the strings not floats or integers try this:
d = [float(i) for i in c]
####################################################DATA CLEANING#######################
IRRAW = []
RRAW = []
t = []
e = 0;
f = 11;
g = 12;

# values extracion ##

for e in range (0, int(len(d)/13)*13,13) :
    t.append(d[e])
print(t)

for f in range (11, int(len(d)/13)*13, 13):
    IRRAW.append(d[f])
print(IRRAW)    
    
for g in range (12, int(len(d)/13)*13,13):
    RRAW.append(d[g])
print(RRAW)
'''
plt.plot(t,IRRAW)
plt.plot(t,RRAW)
plt.show()'''
#################### PART 2 ######################## use low pass filter ###########
###############################################################################

import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


########################################################################
################ Filter requirements.###################################
########################################################################
    
order = 6
fs = 25       # sample rate, Hz
cutoff = 7 # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)


'''
# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 5.0         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
'''
datai = IRRAW
datar = RRAW
# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(datai, cutoff, fs, order)
x = butter_lowpass_filter(datar, cutoff, fs, order)

######################################################################################
######################### Plot the frequency response.#############################
'''
w, h = freqz(b, a, worN=8000)
plt.subplot(3, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, datai, 'b-', label='IRRAW')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, datar, 'r-', label='RRAW')

plt.plot(t, x, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()


'''
###################################### SPLINE - curve fitting ###############################
#############################################################################################
y = IRRAW
x = t


from scipy.interpolate import UnivariateSpline
s = UnivariateSpline(x, y, s=34000)
xs = t
ys = s(xs)
plt.subplot(2,1,1)
plt.plot(x, y, '.')
plt.plot(xs, ys)
plt.show()

yl = RRAW
x = t
s = UnivariateSpline(x, yl, s=34000)
xs = t
yls = s(xs)
plt.subplot(2,1,2)
plt.plot(x, yl, '.')
plt.plot(xs, yls)
plt.show()

################################################# CALCULATION SPO2###############################
#################################################################################################
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
r1 = 0 
r = 0
r3 = 0
r4 = 0
RH = []
RL = []
IRH = []
IRL = []
# for RL
for i in range(30):
    RL.append(min(RRAW[r:r+25]))
    r = r+25
# for IRL 
for i in range(30):
    IRL.append(min(IRRAW[r4:r4+25]))
    r4 = r4+25
# for RH 
for i in range(30):
    RH.append(max(RRAW[r3:r3+25]))
    r3 = r3+25

# for IRH 
for i in range(30):
    IRH.append(max(IRRAW[r1:r1+25]))
    r1 = r1+25
    
xx = np.divide(RL,RH)
yy = np.divide(IRL,IRH)
aw = np.log(xx)
sw = np.log(yy)
Ros = np.divide(aw,sw)
HBLR = 0.811
HBLIR = 0.1974
HBOLR = 0.098969
HBOLIR = 0.259896
qw = (HBLR - np.multiply(HBLIR,Ros))
rt = HBOLIR - HBLIR
wq = ((HBLR - HBOLR) + np.multiply(rt,Ros))
SaO2 = abs(np.divide(qw,wq))
print("the average SPO2 is: ", np.mean(SaO2))

