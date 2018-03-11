# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:37:24 2017

@author: User
"""

import corner
import pickle
import emcee
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from chainconsumer import ChainConsumer
from scipy.integrate import dblquad
from scipy.integrate import quad
from scipy.ndimage import filters
from scipy.special import erf, erfc
from scipy import stats
from Fisher import FisherbN, line
import matplotlib.patches as patches
import time


class timer:
    """
    class for timing options
    """
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def time(self, st=''):
        s = self.start
        self.start = time.time()
        print(st + ':', self.start - s)
        return self.start - s

    def get_time(self, st=''):
        end = time.time()
        print(st, self.start - end)
        return self.start - end

    def get_time_hhmmss(self, st):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        print(st, time_str)
        return time_str

def powerlaw(a, b, g, size=1):
    """
    Power-law gen for pdf(x)\propto x^g for a<=x<=b
    """
    #g -= 1
    r = np.random.random(size=size)
    ag, bg = a**(g+1), b**(g+1)
    return (ag + (bg - ag)*r)**(1./(g+1))

class pars(dict):
    """
    class of dict of parameters to be used by model:
    """
    def __init__(self):
        super().__init__()
        self.names = []

    def __setitem__(self, key, value):
        super().__setitem__(key, par(self, key, value))
        setattr(self, key, value)
        if key not in self.names:
            self.names.append(key)

    def __str__(self):
        s = '{'
        for k, v in self.items():
            s += k + ': ' + str(v.value) + '; '
        return s+'}'


class par():
    """
    class for single parameter:
    """
    def __init__(self, parent, name, value):
        self.parent = parent
        self.value = value
        #ranges = {'alpha': [0.01, 0.25], 'b0': [1.0, 1.3], 'turb': [10, 50], 'beta': [-1.6, -0.7]}
        #ranges = {'alpha': [-10, 1.], 'b0': [-3, 15], 'turb': [0, 200], 'beta': [-3, 3]}
        #ranges = {'alpha': [-3, 3], 'b0': [0.9, 1.3], 'turb_m': [0, 5], 'turb_s': [-3, 5], 'beta': [-3, 1]}
        ranges = {'alpha': [-3, 3], 'b0': [-3, 3], 'power': [-3, 10], 'beta': [-3, 3], 'Pb': [0,1], 'Yb': [-100, 100], 'log10Vb': [-5, 5]}
        #ranges = {'alpha': [-3, 3], 'b0': [-3, 3], 'power': [-3, 10]}
        #ranges = {'alpha': [-1, 1.], 'b0': [-3, 3], 'power': [-3, 3], 'beta': [-3, 3]}
        self.range = ranges[name]
        

    def limit(self, p=None):
        if p is None:
            p = self.value
        if self.range[0] <= p <= self.range[1]:
            return 1
        else:
            return 0

class model():
    """
    class for estimate, generate and check parameters of the probabilities in (N, b).
    """
    def __init__(self, x=None, y=None, xerr=None, yerr=None):
        if x is not None:
            self.x = x
            self.num = len(self.x)
        if y is not None:
            self.y = y
        if xerr is not None:
            self.xerr = xerr
        if yerr is not None:
            self.yerr = yerr
        self.pars = pars() #pars(alpha=0.15, b0=1.2, turb=20)
        self.set_pars()
        self.set_range()
        self.N0 = 12

    def set_pars (self, alpha=0.12, b0=1.02, power = 1.8, beta=-1.6, Pb = 0.1, Yb=15., log10Vb=2): #(self, alpha=0.17, b0=0.9, power = 2.): #(self, alpha=0.17, b0=0.9, power = 2., beta=-1.2): #(self, alpha=0.15, b0=1.07, power=1.3, beta=-1.2):  # #(self, alpha=0.15, b0=1.07, turb=12, beta=-1.2):    #(self, alpha=0.15, b0=1.07, turb=20, beta=-1.2):
        #p = ['alpha', 'b0', 'turb', 'beta']
        #p = ['alpha', 'b0', 'turb_m', 'turb_s', 'beta']
        p = ['alpha', 'b0', 'power', 'beta', 'Pb', 'Yb', 'log10Vb']
        #p = ['alpha', 'b0', 'power']
        for p in p:
            self.pars[p] = locals()[p]

    def set_range(self, xmin=13.0, xmax=14.5, ymin=10, ymax=30): #14.26
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        

    def generate_data(self, n):
        """
        generate (N, b) data
        parameters:
            - n        : number of the points in the sample
        """
        self.x = np.log10(powerlaw(10**13, 10**15, self.pars.beta, size=n))
        #self.yturb2 = (powerlaw(0.1, 1000, self.pars.power, size = n))
        #self.yturb2 = stats.powerlaw.rvs(self.pars.power, size = n)
        #self.yturb2 = stats.chi2.rvs(3, scale=self.pars.turb**2, size=n)  #b_turb^2
        #yturb = np.random.lognormal(mean = self.pars.turb_m, sigma = self.pars.turb_s, size=n)  #b_turb
        yturb = powerlaw(0, 30, self.pars.power, size = n)
        self.yturb2 = yturb**2
        self.y = np.sqrt(10**(((self.x-self.N0) * self.pars.alpha + self.pars.b0)*2) +
                         self.yturb2)
                           

#        data = np.genfromtxt('model_data.dat', unpack=True)
        data = np.transpose(np.genfromtxt('quasars with errors.dat'))
        self.x, self.y, self.sn, self.res = data[1], data[2], data[3], data[4]
        self.num = len(self.x)
        
        self.add_Fisher_errs()
#        self.x=np.random.normal(self.x, 1*self.xerr)
#        self.y=np.random.normal(self.y, 1*self.yerr)
        

        
        print(self.x, self.y)
        
    def add_Fisher_errs(self):
        """
        calculate errors of N and b using Fisher matrix
        """
        self.xerr = np.zeros_like(self.x)
        self.yerr = np.zeros_like(self.x)
        if 1:
            lines = []
            lines.append(line('Lya', 1215.6701, 0.41640, 6.265e8))
            for i, x, y, sn, res in zip(range(self.num), self.x, self.y, self.sn, self.res):
                self.xerr[i], self.yerr[i], F, tau0 = FisherbN(y, x, lines, ston=sn, cgs=0, res=res, verbose=0, plots=0)
        else:
            self.xerr, self.yerr =  0.05*np.ones_like(self.x), 2*np.ones_like(self.x)

    def save_data(self):
        """
        save the data of the model
        """
        print(np.array([self.x, self.xerr, self.y, self.yerr]).transpose())
        np.savetxt('model_data.dat', np.array([self.x, self.xerr, self.y, self.yerr]).transpose(), fmt='%.2f')
        
        dat = np.genfromtxt('quasars with errors.dat', dtype = (float, float, float, float, float, 'U100'), unpack=False)
        file = open('quasars with lines errors.dat', 'w')  # create file
        for i in range(len(dat)):
                file.write('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\n' % (dat[i][0], dat[i][1], self.xerr[i], dat[i][2], self.yerr[i], dat[i][3], dat[i][4], dat[i][5]))  # assume you separate columns by tabs
        file.close()  # close file

    def load_data(self, fname=None):
        """
        load the data of the model:
        parameters:
            - fname     : filename to load from the data, if None load from 'model_data.dat'
        """
#        data = np.genfromtxt('model_data.dat', unpack=True)
        data = np.genfromtxt('data.dat', unpack=True)
        print(len(data))
#        data = np.transpose(np.genfromtxt('qsos_sum.csv', delimiter=','))
#        data = pickle.load(open( "cut_qso.pkl", "rb" ))
        if len(data) == 2:
            self.x, self.y = data[0], data[1]
        else:
            x, xerr, y, yerr, z = data[1], data[2], data[3], data[4], data[0]
            x, xerr, y, yerr, z = x[x>13], xerr[x>13], y[x>13], yerr[x>13], z[x>13]
            x, xerr, y, yerr,z = x[y<30], xerr[y<30], y[y<30], yerr[y<30], z[y<30]
            x, xerr, y, yerr, z = x[y>10], xerr[y>10], y[y>10], yerr[y>10], z[y>10]
            self.x, self.xerr, self.y, self.yerr, self.z =x[x<14.5], xerr[x<14.5], y[x<14.5], yerr[x<14.5], z[x<14.5]
            #self.x, self.xerr, self.y, self.yerr = data[0], 0.05*np.ones_like(data[0]), data[2], 0.05*np.ones_like(data[0])
            #print('er x, y')
            #print(min(data[1]), max(data[1]), min(data[3]), max(data[3]) )
        #self.x, self.xerr, self.y, self.yerr = [13.1], [0.2], [18], [2]
    def gauss(self, Ni, bi):
        x, y, xerr, yerr = self.tr_data[:, 0, :][self.ind[:,0] ==1], self.tr_data[:, 1, :][self.ind[:,0] ==1], self.tr_data[:, 2, :][self.ind[:,0] ==1], self.tr_data[:, 3, :][self.ind[:,0] ==1]

        g = np.zeros((len(x), self.num))  
        g = (1/(xerr*yerr*2*np.pi))*np.exp(-((Ni - x)**2)/(2*xerr**2)) * np.exp(-((bi - y)**2)/(2*yerr **2))
        return g
        
    def lnprob(self, kind='conv', num=300):
        """
        calculate likelihood
        parameters:
            - kind      : type of calculation
        """
        s = 0

        if kind == 'conv':
            if 0:
                for x, y, xerr, yerr in zip(self.x, self.y, self.xerr, self.yerr):
                    self.calc_grid_gauss(num, x, y, xerr, yerr)
                    prob = np.sum(self.prob)/np.sum(self.prob_norm)
                    s += np.log(prob+1.e-100)
            else:

                                         
                #self.ind = np.zeros((len(self.tr_data), self.num))
                #e=[]
                #e = np.arange(len(self.tr_data[:, 1, 0]))[self.tr_data[:, 1, 0] <= 10**(self.pars.alpha*(self.tr_data[:, 0, 0]-self.N0) + self.pars.b0)]
                
                
                #for i in (e):
                #    self.ind[i,:]=1
                    
               
#                for i in range(len(self.tr_data)):
#                    if self.tr_data[i, 1] <= 10**(self.pars.alpha*(self.tr_data[i, 0]-self.N0) + self.pars.b0):
#                        ind[i,:]=1
                 
                #a_up = self.probab_for_array(self.arr_grid[:, 0, :][self.ind[:,0] !=1], self.arr_grid[:, 1, :][self.ind[:,0] !=1] )
                #a_down = self.probab_for_array(self.u_xy[:, 0, :][self.ind[:,0] ==1], self.u_xy[:, 1, :][self.ind[:,0] ==1]) * self.gauss(self.u_xy[:, 0, :][self.ind[:,0] ==1], self.u_xy[:, 1, :][self.ind[:,0] ==1])
                #a_norm_u = self.probab_for_array(self.arr_grid[:, 2, :][self.ind[:,0] !=1], self.arr_grid[:, 3, :][self.ind[:,0] !=1])
                #a_norm_d = self.probab_for_array(self.arr_grid[:, 2, :][self.ind[:,0] ==1], self.arr_grid[:, 3, :][self.ind[:,0] ==1])
                
                if 0:
                    a = np.vstack((a_up, a_down))
                    a_norm = np.vstack((a_norm_u, a_norm_d))
                else:
                    a = self.probab_for_array(self.arr_grid[:, 0, :], self.arr_grid[:, 1, :],  self.arr_grid[:, 4, :])
                    a_norm = self.probab_for_array(self.arr_grid[:, 2, :], self.arr_grid[:, 3, :],  self.arr_grid[:, 4, :])
                
                #print(np.shape(a_down))

#                a = self.probab_for_array(self.arr_grid[:, 0, :], self.arr_grid[:, 1, :]) * (1-self.ind[:,:]) + self.probab_for_array(self.u_xy[:, 0, :], self.u_xy[:, 1, :]) * self.gauss(self.u_xy[:, 0, :], self.u_xy[:, 1, :]) * self.ind[:,:]      
#                a_norm = self.probab_for_array(self.arr_grid[:, 2, :], self.arr_grid[:, 3, :])
        
                
 #               self.arr_prob = np.sum(a, axis=1)
 #               self.arr_prob_norm = np.sum(a_norm, axis=1)

                prob =(1-self.pars.Pb) * np.sum(a, axis=1)/(np.sum(a_norm, axis=1))
                prob = prob/(self.ymax- self.ymin)
                V2 = 10**self.pars.log10Vb + self.tr_data[:, 3]**2
                #prob_out = self.pars.Pb * np.sum(self.probab_out_for_array(self.arr_grid[:, 1, :]), axis=1)/np.sum(self.probab_out_for_array(self.arr_grid[:, 3, :]), axis=1)
                prob_out =self.pars.Pb * 2/(np.sqrt(2*np.pi * V2)) * np.exp((-(self.tr_data[:, 1] - self.pars.Yb)**2)/(2*V2)) / (-erfc((self.ymax - self.pars.Yb)/(np.sqrt(2 * V2)))+ erfc((self.ymin - self.pars.Yb)/(np.sqrt(2 * V2))))
                #prob_out = prob_out / (self.ymax - self.ymin)
               
              
#                print(np.sum(a, axis=1))
#                print(np.sum(a_norm, axis=1))
                #print(prob[-10:])
                #print(prob_out[-10:])
                
 #               s = np.sum(np.log(prob+1.e-100))
#                s = np.sum(np.log(prob))
                s = np.sum(np.log((prob + prob_out)))
                
#                print(s)
                
            
        if kind == 'integ':
            t = timer()
            self.make_grid(n=num)
            self.calc_grid()
            for x, y, xerr, yerr in zip(self.x, self.y, self.xerr, self.yerr):
                if self.xmin < x < self.xmax and self.ymin < y < self.ymax:
                    #print(x, y, xerr, yerr)
                    if 0:
                        ymin, ymax = y - 10*yerr, y + 10*yerr
                        xmin, xmax = lambda y: x - 10*xerr, lambda y: x + 10*xerr
                        prob = dblquad(self.probab_gauss, ymin, ymax, xmin, xmax, args=(x, y, xerr, yerr))
                        prob = prob[0]
                    else:
                        #self.calc_grid(self.probab_gauss, args=(x, y, xerr, yerr))
                        prob = np.sum(self.prob * self.gauss_kern2d(x, y, xerr, yerr))
                    #print(np.log10(prob))
                    s += np.log10(prob+1.e-100)
        #input()
        if 0 and -s == np.inf:
            print(self.pars)
            for x, y, xerr, yerr in zip(self.x, self.y, self.xerr, self.yerr):
                pass #print(np.log10(self.probab(x, y)))
            input()
        #input()
       
        return s

    def calc_rejection_prob(self):
        dat = pickle.load(open("all_o_an2.pkl", "rb"))
        pars = self.pars.names
        ndim = len(pars)
        s = np.array(dat)[3900:4000, 90:, :].reshape((-1, ndim + 1))
        self.chain_len = len(s[:, 0])
        prob = 0
        self.num = len(self.arr_grid[:, 0, :])
        #a = np.zeros((len(self.x), self.num, len(s[:,0])))
        #a_norm = np.zeros((len(self.x), self.num, len(s[:, 0])))
        if 0:
            pass

        else:
            prob_out1 = np.zeros((len(self.arr_grid[:, 0, 0]), len(s[:,0])))
            for i in range(len(s[:,0])):
                print(i)
                a = self.probab_for_array_chain(self.arr_grid[:, 0, :], self.arr_grid[:, 1, :], self.arr_grid[:, 4, :], s[i, 1], s[i, 2], s[i, 3], s[i, 4])
                a_norm = self.probab_for_array_chain(self.arr_grid[:, 2, :], self.arr_grid[:, 3, :], self.arr_grid[:, 4, :], s[i, 1], s[i, 2], s[i, 3], s[i, 4])

                prob += (1 - s[i, 5]) * np.sum(a, axis=1) / (np.sum(a_norm, axis=1))


                V2 = 10 ** s[i, 7] + self.tr_data[:, 3] ** 2

                prob_out1[:, i] = s[i, 5] * 2 / (np.sqrt(2 * np.pi * V2)) * np.exp((-(self.tr_data[:, 1] - s[i, 6]) ** 2) / (2 * V2)) / (
                       -erfc((self.ymax - s[i, 6]) / (np.sqrt(2 * V2))) + erfc(
                   (self.ymin - s[i, 6]) / (np.sqrt(2 * V2))))
        prob_out = np.sum(prob_out1, axis=1)
        prob = prob / (self.ymax - self.ymin)

        self.rej_prob = prob/prob_out
        pickle.dump(self.rej_prob, open("rej_prob11.pkl", "wb"))
        print(len(self.rej_prob))
        print(len(self.x))
        print(self.rej_prob)


        

    def lnprior(self, p=None):
        """
        Check if parameters in range
        """
        prior = 1
        if p is None:
            p = [self.pars[i].value for i in self.pars.names]

        pars = [self.pars[i] for i in self.pars.names]
        for par, pi in zip(pars, p):
            prior *= par.limit(pi)
        if prior>0:   
         return np.log(prior)
        else:
         return -np.inf

        
        
    def probab(self, Ni, bi):
        """
        return pdf of the model
        parameters:
            - bi      : doppler parameter
            - Ni      : column density
        """
        
        def f(b, N):
            bth = 10**(self.pars.alpha*(N-self.N0) + self.pars.b0)
            scale = self.pars.turb**2
            x = (b**2 - bth**2)/(2*scale)
            if x > 0:
                F = (b/scale) * np.sqrt(x) * np.exp(-x) *  np.power(10, N*(self.pars.beta+1))
            else:
                F = 0  
            return F
                       
        Nmin = 12
        Nmax = 15
        bmin = lambda N: min(10**(self.pars.alpha*(N-self.N0) + self.pars.b0), 40)
        bmax = lambda N: 40
    

        integral = dblquad(f, Nmin, Nmax, bmin, bmax)

        #c1 = (self.pars.beta+1)/(10**(14*(self.pars.beta+1)) - 10**(12*(self.pars.beta+1)))
        c2 = 1/integral[0]
        #c2=1

        #print(Ni)
        
        N, b = np.meshgrid(Ni, bi)
        
        bth = 10**(self.pars.alpha*(N-self.N0) + self.pars.b0)
        scale = self.pars.turb**2
        x = (b**2 - bth**2)/(2*scale)
        
        z = np.zeros_like(N)
        m = x > 0

        z[m] =   c2 * np.sqrt(x[m]) * np.exp(-x[m]) * (b[m]/self.pars.turb**2) * np.power(10, N[m]*(self.pars.beta+1))
        return z
        
    def probab_for_array(self, Ni, bi, zi):
        """
        return pdf of the model
        parameters:
            - bi      : doppler parameter
            - Ni      : column density
        """        
#        for i in range(len(Ni)):
#            N1, b1 = np.meshgrid(Ni[i, :], bi[i, :])
#            if i==0:
#                N = N1
#                b = b1
#            else:    
#                N = np.append(N, N1, axis=0)
#                b = np.append(b, b1, axis=0)

        N=Ni
        b=bi
        zred=zi
        
        bth = 10**(self.pars.alpha*(N + 4.5*np.log10((1+zred)/3.4)-self.N0) + self.pars.b0)
        #scale = 0.5*self.pars.turb**(-2)
        scale = 1
        x = (b**2 - bth**2)*scale
        
        
        z = np.zeros_like(N)
        m = x > 0

        if 1:
            #z[m] =  np.sqrt(x[m]) * np.exp(-x[m]) * (b[m]*scale) * np.power(10, N[m]*(self.pars.beta+1))
            z[m] =  (b[m] / np.sqrt(x[m])) * np.power(np.sqrt(x[m]), self.pars.power) * np.power(10, N[m]*(self.pars.beta+1))
            #z[m] =  (b[m] / np.sqrt(x[m])) * np.power(np.sqrt(x[m]), self.pars.power) * np.power(10, N[m]*(-1.2+1))


        else:
            z[m] =  (b[m] / np.sqrt(x[m])) * (1/(self.pars.turb_s * np.sqrt(x[m]) * np.sqrt(2*np.pi))) * np.exp(-((np.log(np.sqrt(x[m])) - self.pars.turb_m)**2)/ (2 * self.pars.turb_s**2)) * np.power(10, N[m]*(self.pars.beta+1))

        return z

    def probab_out_for_array(self, bi):
            """
            return pdf of the model
            parameters:
                - bi      : doppler parameter
                - Ni      : column density
            """
            #        for i in range(len(Ni)):
            #            N1, b1 = np.meshgrid(Ni[i, :], bi[i, :])
            #            if i==0:
            #                N = N1
            #                b = b1
            #            else:
            #                N = np.append(N, N1, axis=0)
            #                b = np.append(b, b1, axis=0)

            #N = Ni
            b = bi
            #zred = zi

            #bth = 10 ** (self.pars.alpha * (N + 4.5 * np.log10((1 + zred) / 3.4) - self.N0) + self.pars.b0)
            # scale = 0.5*self.pars.turb**(-2)
            #scale = 1
            #x = (b ** 2 - bth ** 2) * scale

            z = np.zeros_like(b)


            if 1:
                # z[m] =  np.sqrt(x[m]) * np.exp(-x[m]) * (b[m]*scale) * np.power(10, N[m]*(self.pars.beta+1))
                V2=2*(10**self.pars.log10Vb)
                z = 1/np.sqrt(np.pi*V2) * np.exp(-((b-self.pars.Yb)**2)/V2)

                # z[m] =  (b[m] / np.sqrt(x[m])) * np.power(np.sqrt(x[m]), self.pars.power) * np.power(10, N[m]*(-1.2+1))


#            else:
 #               z[m] = (b[m] / np.sqrt(x[m])) * (1 / (self.pars.turb_s * np.sqrt(x[m]) * np.sqrt(2 * np.pi))) * np.exp(
 #                   -((np.log(np.sqrt(x[m])) - self.pars.turb_m) ** 2) / (2 * self.pars.turb_s ** 2)) * np.power(10, N[
 #                   m] * (self.pars.beta + 1))

            return z

    def probab_for_array_chain(self, Ni, bi, zi, alpha, b0, power, beta):

        N = Ni
        b = bi
        zred = zi

        bth = 10 ** (alpha * (N + 4.5 * np.log10((1 + zred) / 3.4) - self.N0) + b0)

        scale = 1
        x = (b ** 2 - bth ** 2) * scale

        z = np.zeros_like(N)
        #z = np.zeros((len(self.x), self.num, self.chain_len))
        m = x > 0

        z[m] = (b[m] / np.sqrt(x[m])) * np.power(np.sqrt(x[m]), power) * np.power(10, N[m] * (beta + 1))

        return z

    def probab_gauss(self, xs, ys, x, y, xerr, yerr):

        return self.probab(xs, ys) * self.gauss_kern2d(x, y, xerr, yerr)

    def gauss_kern2d(self, x, y, xerr, yerr):
        """Returns a 2D Gaussian kernel array."""

        kern_x = stats.norm.pdf((self.x_grid - x) / xerr)
        kern_y = stats.norm.pdf((self.y_grid - y) / yerr)
        kernel_raw = np.sqrt(np.outer(kern_y, kern_x))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def make_grid(self, n):
        """
        make grid for x, y parameters in the model
        parameters:
            - n       : number of points in grid
        """
        self.x_grid = np.linspace(self.xmin, self.xmax, n)
        self.y_grid = np.linspace(self.ymin, self.ymax, n)
        
    def calc_grid(self, func=None, args=()):
        """
        calculate probability on the grid
        """
        if func == None:
            func = self.probab_for_array

        if 1:
            z = np.zeros((len(self.x_grid), len(self.y_grid)))
            for k, x in enumerate(self.x_grid):
                for i, y in enumerate(self.y_grid):
                    z[i, k] = func(x, y, *args)
        else:
            z = func(self.x_grid, self.y_grid, *args)
        self.prob = z

    def load_grid(self):
        self. arr_grid = pickle.load(open("arr_grid_z_out.pkl", "rb"))
        self.tr_data = pickle.load(open("tr_data_z_out.pkl", "rb"))
        self.num = len(self.arr_grid[0,0,:])
        
    def make_and_calc_grid_gauss(self, n):  
        """
        make grid for normal distr. x, y parameters in the model
        parameters:
            - n       : number of points in grid
        and
        make grid for convolution of normal distr. and unoform distr. 
        x, y parameters in the model
        parameters:
            - n       : number of points in grid
        and
        calculate probability on the grid
        """
        save_grid =1

        self.area = (self.xmax - self.xmin)*(self.ymax - self.ymin)
        self.num = n
#        q = len(self.x[self.x < self.xmax, self.y < self.ymax, self.x > self.xmin, self.y > self.ymin])
        self.arr_grid = np.zeros((len(self.x), 6, n))
        self.arr_prob = np.zeros(len(self.x))
        self.arr_prob_norm = np.zeros(len(self.x))
        self.tr_data = np.zeros((len(self.x), 4))

        self.unif_x = np.random.uniform(self.xmin, self.xmax, n)
        self.unif_y = np.random.uniform(self.ymin, self.ymax, n)  
        
        u_x = np.random.uniform(self.xmin-3*max(self.xerr), self.xmax+3*max(self.xerr), n)
        u_y = np.random.uniform(self.ymin-3*max(self.yerr), self.ymax+3*max(self.yerr), n) 
#

        
         
        
        i=0
        for x, y, xerr, yerr, z in zip(self.x, self.y, self.xerr, self.yerr, self.z):
          if self.xmin < x < self.xmax and self.ymin < y < self.ymax:
            
            self.tr_data[i, 0] = x
            self.tr_data[i, 1] = y
            self.tr_data[i, 2] = xerr
            self.tr_data[i, 3] = yerr
            norm_x = np.random.normal(x, xerr, n)
            norm_y = np.random.normal(y, yerr, n)
   
            self.arr_grid[i, 0, :] = norm_x
            self.arr_grid[i, 1, :] = norm_y
            self.arr_grid[i, 4, 0:] = z

            self.arr_grid[i, 2, :] = np.random.normal(0, xerr, n) + self.unif_x
            self.arr_grid[i, 3, :] = np.random.normal(0, yerr, n) + self.unif_y
            self.arr_grid[i, 5, 0:] = 2.4
            
            
            #self.arr_prob[i] = np.sum(func(self.arr_grid[i, 0, :], self.arr_grid[i, 1, :]))
            #self.arr_prob_norm[i] = np.sum(func(self.arr_grid[i, 2, :], self.arr_grid[i, 3, :]))
            i+=1

        self.arr_grid=self.arr_grid[:i]
        self.tr_data=self.tr_data[:i] 
        
        self.u_xy =  np.zeros((i, 2,n))
        
        self.u_xy[0:, 0, :] = u_x
        self.u_xy[0:, 1, :] = u_y
        if save_grid:
            pickle.dump(self.arr_grid, open("arr_grid_z_out.pkl", "wb"))
            pickle.dump(self.tr_data, open("tr_data_z_out.pkl", "wb"))

        
#        print(np.shape(self.u_xy))   
#        input()
#        print(len(self.arr_grid))
#        a = func(self.arr_grid[:, 0, :], self.arr_grid[:, 1, :])
#        a_norm = func(self.arr_grid[:, 2, :], self.arr_grid[:, 3, :])
#        
#        self.arr_prob = np.sum(a, axis=1)
#        self.arr_prob_norm = np.sum(a_norm, axis=1)
        #for i in range(len(self.x)):
           # self.arr_prob[i] = np.sum(a[n*i:n*(i+1)])
           # self.arr_prob_norm[i] = np.sum(a_norm[n*i:n*(i+1)])
            
   
    def calc_grid_gauss(self, n, x, y, xerr, yerr, func=None):
        """
        calculate probability on the grid
        """
        if func == None:
            func = self.probab

        unif_x = np.random.uniform(self.xmin, self.xmax, n)
        unif_y = np.random.uniform(self.ymin, self.ymax, n)
        z = func(np.random.normal(x, xerr, n), np.random.normal(y, yerr, n))
        z_norm = func(np.random.normal(0, xerr, n) + unif_x, np.random.normal(0, yerr, n) + unif_y)
        self.prob = z
        self.prob_norm = z_norm



    def plot_prob(self, ax=None):
        """
        plot probability contour of the model
        parameters:
            - n       : number of points in grid
            - ax      : axis to plot, if None - create
        """
        if ax is None:
            fig, ax = plt.subplots()

        self.prob = np.log(self.prob)
        self.prob = self.prob - np.max(self.prob)
        x_grid, y_grid = np.meshgrid(self.x_grid, self.y_grid)
        cax = ax.contourf(x_grid, y_grid, self.prob, 100, cmap='viridis')
        ax.contour(x_grid, y_grid, self.prob, 40, colors='k')
        cbar = plt.colorbar(cax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_xlim(self.xmin, self.xmax)

    def plot_data(self, ax=None):
        """
        plot the data of the model
        parameters:
            - ax      : axis to plot, if None - create
        """
        if ax is None:
            fig, ax = plt.subplots()

        if 1:
            self.rej_prob = pickle.load(open("rej_prob11.pkl", "rb"))
            print(self.rej_prob)

        bz = [16.82, 16.86, 17.7, 17.42]
        nz = [13.008, 13.015, 13.033, 13.033]
        bbz = [16.387, 16.83, 17.786, 16.711]



        x_b, xerr_b, y_b, yerr_b = self.x[self.rej_prob<1], self.xerr[self.rej_prob<1], self.y[self.rej_prob<1], self.yerr[self.rej_prob<1]
        print(x_b)
        print(y_b)
        print(self.z[self.rej_prob<1])
        #ax.scatter(nz, bbz, marker='.', color='m')
        ax.errorbar(self.x, self.y, xerr=[self.xerr, self.xerr], yerr=[self.yerr, self.yerr],
                    fmt='o', marker='o', color='b', markersize=2)
        ax.errorbar(x_b, y_b, xerr=[xerr_b, xerr_b], yerr=[yerr_b, yerr_b],
                    fmt='o', marker='o', color='r', markersize=2)
        #ax.errorbar(x_b, bz, xerr=[xerr_b, xerr_b], yerr=[yerr_b, yerr_b],
        #           fmt='o', marker='o', color='m', markersize=1)
        #ax.errorbar(nz, bbz, xerr=0, yerr=0,
        #            fmt='o', marker='o', color='r', markersize=2)

        x=np.arange(self.xmin, self.xmax+1,0.01)
        #bth = 10 ** (0.075 * (N + 4.5 * np.log10((1 + zred) / 3.4) - 12) + 1.15)
        zred = [2.2364591, 2.1456858, 2.2272154, 2.0287573]
        ax.plot(x, 10**((0.075*(x-self.N0) +1.15)), color='g',  lw='2')
        #a1, =ax.plot(x, 10 ** ((0.075 * (x + 4.5 * np.log10((1 + zred[0]) / 3.4)- self.N0) + 1.15)), color='black', lw='1', label=zred[0])
        #a2, =ax.plot(x, 10 ** ((0.075 * (x + 4.5 * np.log10((1 + zred[1]) / 3.4)- self.N0) + 1.15)), color='c', lw='1', label=zred[1])
        #a3,=ax.plot(x, 10 ** ((0.075 * (x + 4.5 * np.log10((1 + zred[2]) / 3.4)- self.N0) + 1.15)), color='m', lw='1', label=zred[2])
        #a4,=ax.plot(x, 10 ** ((0.075 * (x + 4.5 * np.log10((1 + zred[3]) / 3.4)- self.N0) + 1.15)), color='r', lw='1', label=zred[3])
        #plt.legend(handles=[a1, a2, a3, a4])

        #ax.plot(x, 10**((0.61*(x-self.N0) - 0.16)), color='m', lw='1')
        #ax.plot(x, 10**((self.pars.alpha*(x-self.N0) + self.pars.b0)), color='c',  lw='1')          
        #ax.scatter(self.x, 10**((self.pars.alpha*(self.x-self.N0) + self.pars.b0)), marker='.', color='g')
        #ax.scatter(self.x, 10**((self.pars.alpha*(self.x-self.N0) + 1.08)), marker='.', color='r')
        ax.add_patch(patches.Rectangle((self.xmin, self.ymin), self.xmax - self.xmin, self.ymax - self.ymin, 
                                       fill = False, edgecolor = 'red'))            
        plt.xlabel('$\log N$')
        plt.ylabel('$b$')
        ax.set_ylim(self.ymin-1, self.ymax+1)
        ax.set_xlim(self.xmin-0.03, self.xmax+0.03)
        ax.set_yscale("log", nonposy='clip')

    def mcmc(self, ax=None, nwalkers=100, nsteps=20000):
        """
        calculate parameters using MCMC
        parameters:
            - ax         : axis to plot, if None - create
            - nwalkers   : number of walkers
            - nsteps     : number of steps to be preformed
        """
        pars = self.pars.names
        ndim = len(pars)
        init = [getattr(self.pars, p) for p in pars]
        init_range = [(self.pars[p].range[1] - self.pars[p].range[0])/300 for p in pars]
        pos0 = np.array([init + init_range * np.random.randn(ndim) for i in range(nwalkers)])
#        print(pos)

     
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.mcmclnprob, args=([pars]))
        if 0:
            save = 1
            
            # Run 5000 steps as a burn-in.
            #pos, prob, state = sampler.run_mcmc(pos0, 250)
            
            # Reset the chain to remove the burn-in samples.
            #sampler.reset()
            
            # Starting from the final position in the burn-in chain, sample for 100000
            # steps.
            #sampler.run_mcmc(pos, nsteps, rstate0=state)
            #sampler.run_mcmc(pos0, nsteps)

            #samples = sampler.flatchain
            #data = []

            data = np.zeros((nsteps, nwalkers, len(self.pars.names)))
            data2 = np.zeros((nsteps, nwalkers))
            dat = np.zeros((nsteps, nwalkers, len(self.pars.names)+1))
            data1 = pickle.load(open( "chain_o_an.pkl", "rb" ))
            pos0 = data1[-1]
            #sampler.run_mcmc(pos, nsteps)
            
            #samples = sampler.chain[:, (nsteps // 2):, :] #.reshape((nwalkers, -1, ndim))
            

            for i, result in enumerate(sampler.sample(pos0, iterations=nsteps, storechain=False)):
                    if (i+1) % 10 == 0:
                        print("{0:5.1%}".format(float(i) / nsteps))
                    
#                    print(result[0])
#                    print(np.shape(result))
                    data2[i] = result[1]
                    data[i] = result[0]

                    dat[:, :, 0] = data2
                    dat[:, :, 1:] = data
                    pickle.dump(data, open("chain_o_an2.pkl", "wb"))
                    pickle.dump(data2, open("chain_lik_o_an2.pkl", "wb"))
                    pickle.dump(dat, open("all_o_an2.pkl", "wb"))
                    
                
#                    print(np.shape(data))
#                    print(np.array(data)[:,9, :])
                
#                    input()
            #dat[:, :, 0] = data2
            #dat[:, :, 1:] = data
            print((np.array(dat)[:, :, :]))
            if save == 1:
                
                pass
                #pickle.dump(data, open("chain.pkl", "wb"))
                #pickle.dump(data2, open("chain_lik.pkl", "wb"))
                #pickle.dump(dat, open("all.pkl", "wb"))
            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            #print("autocorrelation time")
            #print(sampler.get_autocorr_time())
        else:
            load = 1
            dat = []
            #data =[]
            #data2 =[]

            if load:
                #dat = pickle.load(open( "chain.pkl", "rb" ))
                #data2 = pickle.load(open( "chain_lik.pkl", "rb" ))
                dat = pickle.load(open( "all_o_an2.pkl", "rb"))
                print(np.shape(np.array(dat)))
                #print(np.shape(np.array(data)))
                #print(np.shape(np.array(data2)))
                #print(np.array(dat))
                if 0:
                    fig, ax = plt.subplots()
#                    aList = ['l']
#                    
#                    aList = np.append(aList, self.pars.names)
                    
                    c = ChainConsumer()
                    for i in range(10):
                        c.add_chain(np.array(dat)[:, i, :])
                    fig = c.plot_walks()
                    fig.savefig("chain_zz.png")

                                                      
                   
        if 1:
            
            samples = np.array(dat)[4000:, :, :].reshape((-1, ndim+1))
            #samples = np.array(dat)[:, :, :].reshape((-1, ndim))
            #samples = np.array(samples)[:, 1:]
            #samples=samples[:,][samples[:,0]>510]
#            s1 =samples[:,1][samples[:,1]<1.250]
#            s2 = samples[:,2][samples[:,2]>15]
#            plt.hist(samples[:,0], 100)
           
            #plt.show()
        if 1:
            aList = ['l']
            #aList = []
            aList = np.append(aList, self.pars.names)
            #aList = self.pars.names
            print(aList)
            fig = corner.corner(samples, labels = aList,
                                quantiles=[0.16, 0.5, 0.84],
                                verbose=True,
                                no_fill_contours=False,
                                draw_datapoints=True)
            fig.savefig("triangle_o_an.png")
        if 0:
            c = ChainConsumer()
            aList = ['l']
            aList = np.append(aList, self.pars.names)
            samples=samples[:,1:]
            
            c.add_chain(samples, parameters=self.pars.names)
            #c.add_chain(samples)
            #c.configure(smooth=0, cloud=True, sigmas=np.linspace(0, 2, 10))
            #fig = c.plot(figsize=(12,12), truth=[0.17, 1.07, 1.3, -1.2])
            fig = c.plot(figsize=(12,12))


            fig.savefig("triangle201_zz.png")
            
            

    def mcmclnprob(self, p, names):
        for name, x in zip(names, p):
            self.pars[name] = x
        lp = self.lnprior()

        if not np.isfinite(lp):
            return -np.inf
        lpr= self.lnprob()
        if not np.isfinite(lpr):
            return -np.inf
        
        return lp +lpr

    def onedimfit(self, name, num=400, ax=None):
        if ax == None:
            fig, ax = plt.subplots()
        a = np.linspace(self.pars[name].range[0], self.pars[name].range[1], num)
        ln = []
        a1 = []
        for ai in a:
            if 1:
            #if self.mcmclnprob([ai], [name]) > 0:
                ln.append(self.mcmclnprob([ai], [name]))
                a1.append(ai)
                print(ai, ln[-1])
#                if self.mcmclnprob([ai], [name]) < 0:
#                    if ai>1.07:
#                        print('ho')
#                        print(10**((self.pars.alpha*(self.x[0]-self.N0) + ai)), self.y[0] + 3*self.yerr[0])
#                        
        print(self.pars)
        #ax.plot(a1, np.exp((ln - max(ln))))

        ax.plot(a1, ln)
        #plt.xlabel(r'b0')
        #plt.savefig('b0_30points.png', format = 'png', dpi=300)

    def twodimfit(self, names, num=40, ax=None):
        if ax == None:
            fig, ax = plt.subplots()
        x = np.linspace(self.pars[names[0]].range[0], self.pars[names[0]].range[1], num)
        y = np.linspace(self.pars[names[1]].range[0], self.pars[names[1]].range[1], num)
        ln = np.zeros([num, num])
        for i, xi in enumerate(x):
            for k, yk in enumerate(y):
                ln[i, k] = self.mcmclnprob([xi, yk], names)
        #ln = np.exp(-(ln - np.min(ln)))
        x, y = np.meshgrid(x, y)
        cax = ax.contourf(x, y, ln, 100)
        cbar = plt.colorbar(cax)

if __name__ == '__main__':
    lyman = model()
#    lyman.pars['alpha'] = 0.15
#    lyman.pars['b0'] = 1.2
#    lyman.pars['turb'] = 15
    if 0:
        lyman.generate_data(300)
        lyman.save_data()
    else:
        lyman.load_data()
        if 0:
            lyman.make_and_calc_grid_gauss(500)
        if 1:
            lyman.load_grid()
    if 0:
        lyman.calc_rejection_prob()

        
    if 1:
        fig, ax = plt.subplots()
        #lyman.make_grid(n=200)
        #lyman.calc_grid()
        #lyman.plot_prob(ax=ax)
        lyman.plot_data(ax=ax)
        #lyman.plot_const(ax=None)
        

    if 1:
        pass
#        lyman.onedimfit('b0')
        #print(10**((lyman.pars.alpha*(lyman.x[0]-lyman.N0) + 1.088)), lyman.y[0])
#        print(lyman.mcmclnprob([-0.01, 1.01, 20, 1, -1.2], lyman.pars.names))
        
#        print(lyman.mcmclnprob([0.12, 1.02, 1.8, -1.76, 0.0, 15, 2], lyman.pars.names))
#        print(lyman.mcmclnprob([0.12, 1.02, 1.8, -1.76, 0.01, 15, 2], lyman.pars.names))
#        lyman.onedimfit('alpha')
#        lyman.onedimfit('turb_m')
#        lyman.onedimfit('turb_s')
#        lyman.onedimfit('power')
#        lyman.onedimfit('beta')
#        lyman.twodimfit(['b0', 'alpha'])
    else:
        lyman.mcmc()
    plt.show()
        
        