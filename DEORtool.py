from LGSM import LGSM
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class DEORtool(object):
    
    def power_law_plot(fs=50,fe=100,step=1,R=False): 
        #fs:start freq , fe: end freq , step: freq step , R : return or not
        freq = np.arange(fs,fe,step)
        L = len(freq)
        a = LGSM()
        b = a.generate(freq)
        T = []
        for i in range(L):
            t = np.mean(b[i])
            T.append(t)
            
        plt.plot(freq,T)
        plt.title("Global Average Foreground Temperature")
        plt.xlabel("Frequency/MHz")
        plt.ylabel("Global Average Temperature/K")
        plt.show()
        if R == True:
            return freq, T
        
    def time_series(lon, lat, elev,year,month,day, freq):
        
        a = LGSM()
        a.lon = lon
        a.lat = lat
        a.elev = elev
        a.date = datetime(year,month,day,0,0)
        
        a.generate(freq)
        t = np.arange(0,1440,30)
        T = []
        for i in t:
            t1 = i//60
            t2 = i%60
            b = a.mask_sky()
            avg = np.mean(b)
            T.append(avg)
            a.date = datetime(year,month,day,t1,t2)
            
        plt.plot(t,T)
        plt.title("Global Average Foreground Temperature with time series")
        plt.xlabel("time/min")
        plt.ylabel("Global Average Temperature/K")
        plt.show()           

class EoRmod(object):
            
    def Bowman2008(x=78.3,plot=False):
        #input freq return eor signal T
        a = -150/1000#k
        v21 = 78.3#mhz
        g = 5
        T = a*np.exp(-((x - v21)**2)/(2*(g**2)))
        if plot == True:
            f = np.arange(50,100,1)
            temp = a*np.exp(-((f - v21)**2)/(2*(g**2)))
            plt.plot(f,temp)
            plt.title("Bowman2008 EoR model")
            plt.xlabel("freq/MHz")
            plt.ylabel("Global Average Temperature/K")
            plt.show() 
        return T

    def EDGES2018(x=78.3,plot=False):
        a = -520/1000
        v = 78.3
        t = 7
        w = 20.7
        b = (4*((x-v)**2)/(w**2))*np.log10(-(np.log10((1+np.exp(-t))/2))/t)
        T = a*((1-np.exp(-t*np.exp(b)))/(1-np.exp(-t)))
        if plot == True:
            f = np.arange(50,110,1)
            b = (4*((f-v)**2)/(w**2))*np.log10(-(np.log10((1+np.exp(-t))/2))/t)
            temp = a*((1-np.exp(-t*np.exp(b)))/(1-np.exp(-t)))
            plt.plot(f,temp)
            plt.title("EDGES2018 EoR model")
            plt.xlabel("freq/MHz")
            plt.ylabel("Global Average Temperature/K")
            plt.show() 
        return T
    
    def EDGES_FIT(v,v0,t,r,w,A):
        B = (4*(v-v0)**2)*np.log(-np.log((1+np.exp(-t))/2)/r)/(w**2)
        T_21cm = -A((1-np.exp(-t*np.exp(B)))/(1-np.exp(-t)))
        return T_21cm
    
    def T_21cm(X_HI,z,T_S,T_CMB):
        #X_HI氢原子电导率，z红移，T_CMB cmb温度，T_S 自旋温度
        T_21cm = 27*X_HI*np.sqrt((1+z)/10)*((T_S-T_CMB)/T_S)
        return T_21cm
    
    