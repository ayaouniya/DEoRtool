#read far field from feko ffe file and calculate antenna beam factor

import csv
import numpy as np
from scipy.interpolate import griddata

class cbeam(object):
    
#设置参数，head1为ffe文件中前五行，head2指ffe文件中6-15行
    def __init__(self):
        self.data = None
        self.n_theta = None
        self.n_phi = None
        self.n_freq = None
        self.head1 = 5
        self.head2 = 10
        
#读取文件
    def read_file(self,filepath):
        csvFile = open(filepath,"r")
        f = csv.reader(csvFile)
        self.data = list(f)
        return self.data
    
#设置参数，分别为角度和频率的取样数目 
    def set_up(self,n_theta,n_phi,n_freq):
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.n_freq = n_freq
        self.n_line = self.n_theta*self.n_phi
        
#读取数据
    def read_data(self):
        L = len(self.data)
        self.freq = np.zeros(self.n_freq)
        self.theta = np.zeros([self.n_freq,self.n_line])
        self.phi = np.zeros([self.n_freq,self.n_line])
        self.Gain = np.zeros([self.n_freq,self.n_line])
        #read frequency
        for i in range(self.n_freq):
            s = 8 + i * (self.n_line+self.head2)
            self.freq[i] = self.data[s][1]
            
        #read theta,phi,Gain
        for i in range(self.n_freq):
            p1 = 0
            s = np.arange(15+i*(self.n_line+10),self.n_line+15+i*(self.n_line+10))
            for j in s:
                self.Gain[i][p1] = self.data[j][9]
                self.theta[i][p1] = self.data[j][1]
                self.phi[i][p1] = self.data[j][2]
                p1 = p1 + 1
        #计算天线效率
        self.eff = 10**(0.1*self.Gain)
        
        return self.Gain , self.theta , self.phi
    
    def beam_interpolate(self,coord):
        
        data = []
        gain = []
        for i in range(self.n_freq):
            for j in range(self.n_line):
                d = [self.theta[i][j],self.phi[i][j],self.freq[i]]
                g = self.eff[i][j]
                data.append(d)
                gain.append(g)
                
        self.points = data
        self.values = gain
        
        self.beam = griddata(self.points,self.values,coord,method = 'nearest')
 
        return self.beam