# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:02:49 2016

@author: Serj

Puprose:

Fisher matrix calculation for single absorption line with Voigt profile

"""
from astropy import constants as const
import astropy.convolution as conv 
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz
from scipy import interpolate
from scipy import optimize
from numpy import genfromtxt
from scipy.stats import norm

#==============================================================================
# Voigt Real and Imaginery fucntion and partial derivatives
#==============================================================================

class Voigt():
    def __init__(self, n):
        self.h = np.zeros((n,2))
        self.k = np.zeros((n,2))
        
    def set(self, a, x, n):
        z = wofz(x + 1j*a)
        if n > -1:
            self.H = z.real
            self.K = z.imag
        if n > 0:
            self.H1a = 2*(z.real*a + z.imag*x - 1/np.sqrt(np.pi))
            self.K1a = 2*(z.imag*a - z.real*x)
            self.H1x = self.K1a
            self.K1x = -self.H1a
        if n > 1:
            self.H2a = 4*(z.real*(a**2-x**2+.5) + 2*z.imag*a*x - a/np.sqrt(np.pi))
            self.K2a = 4*(z.imag*(a**2-x**2+.5) - 2*z.real*a*x - x/np.sqrt(np.pi))
            self.H2x = -self.H2a
            self.K2x = -self.K2a
        if n > 2:
            self.H3a = 8*(z.real*(a**3-3*a*x**2+1.5*a) - z.imag*(x**3-3*a**2*x-1.5*x) + (x**2-a**2-1)/np.sqrt(np.pi))
            self.K3a = 8*(z.imag*(a**3-3*a*x**2+1.5*a) + z.real*(x**3-3*a**2*x-1.5*x) + 2*x*a/np.sqrt(np.pi))
            self.H3x = -self.K3a
            self.K3x = self.H3a
    
#==============================================================================
# 
#==============================================================================


def voigt(a, x, calc='spec'):
    """
    Returns voigt function

    parameters:
        - a       : a parameter
        - x       : x parameter
        - calc    : type of calculation

    return:
        voigt     : voigt function at x positions
    """
    if calc == 'spec':
        v = Voigt(0)
        v.set(a, x, 0)
        return v.H


# ==============================================================================
#
# ==============================================================================

class line:
    def __init__(self, name, l, f, g):
        line.name = name
        line.f = f
        line.l = l
        line.g = g

def funct(x, a, tau_0):
    return (wofz(x + 1j*a).real*tau_0-0.001)**2


def FisherbN(b, N, lines, ston=1, cgs=0, convolve=1, res=50000, z=2.67, verbose=1, plots=1):
    """
    calculate the Fisher matrix for a given b and logN, the parameters of line profile
    
    parameters:
        - b         :  b parameter in km/s
        - N         :  column density in log10[, cm^-2] units
        - lines     :  list of lines
        - ston      :  Signal to Noise ratio, inverse of dispersion
        - cgs       :  if 0 then derivative for N in cm^-2 and b in cm/s
        - convolve  :  if 1 convolve data else not convolve
        - res       :  resolution of the spectrograph (assuming 3 pixels in FWHM)
        - z         :  redshift of line   
        
    return:
        - db        :  uncertainty for b parameter in km/s
        - dN        :  uncertainty for column density in log10[, cm^-2] 
        - F         :  fisher matrix
    """
    
    V = Voigt(3)
    n = 200
    gauss_kernel = conv.Gaussian1DKernel(const.c.cgs.value/res/1e5)

    F_unc = np.zeros((6, n))
    F_con = np.zeros((6, n))
    F_con_extr = [0]*6
    F = np.zeros((2,2))
    
    if plots:
        fig, ax = plt.subplots(6, 1, figsize=(8,20))
        
    for line in lines:
        bin_width = line.l * (1+z) / res / 2.5
        #print(bin_width)
        if verbose:
            print('bin width=', bin_width)
            
        dl = line.l * b / const.c.to('km/s').value
        if verbose:
            print(r'dl =', dl)        
        
        a = line.g * line.l / 1e8 / 4 / np.pi / b / 1e5
        if verbose:
            print('a=', a)        

        tau_0 = 0.014983 * line.l * line.f / 1e8 * np.power(10.0,N) / b / 1e5
        
        x_lim = optimize.fmin(funct, 2, args=(a, tau_0), disp=0)
        
        l = np.linspace(line.l - dl * x_lim * 3, line.l + dl * x_lim * 3, n)
        x = (l - line.l) / dl
       
        if verbose:
            print('tau_0=', tau_0)
        
        
        for i, xi in enumerate(x):
            V.set(a, xi, 3)
            tau = tau_0 * V.H
            F_unc[0, i] = 1
            if not cgs:
                F_unc[1, i] = (tau**2-tau) * np.log(10)**2
                F_unc[2, i] = tau_0 * V.H2x / 2 / b * (tau-1) * np.log(10)
                F_unc[3, i] = tau_0 / 2 / b / b * (tau_0/2*V.H2x**2 - a*V.H3a - xi*V.H3x + 2*V.H2x)
                F_unc[4, i] = tau*np.log(10)
                F_unc[5, i] = tau_0 / 2 / b * V.H2x
            else:
                F_unc[1, i] = (tau / np.power(10.0,N))**2
                F_unc[2, i] = tau_0 * V.H2x / 2 / np.power(10.0,N) / b / 1e5 * (tau-1)
                F_unc[3, i] = tau_0 / 2 / b / b / 1e10 * (tau_0/2*V.H2x**2 - a*V.H3a - xi*V.H3x + 2*V.H2x)
                F_unc[4, i] = tau / 10**N
                F_unc[5, i] = tau_0 / 2 / b / 1e5 * V.H2x

            F_unc[:,i] *= np.exp(-tau)
        
        if convolve:
            for i in range(6):
                F_con[i,:] = conv.convolve(F_unc[i,:], gauss_kernel, boundary='extend')
        
        colors = ['k', 'r', 'b', 'g', 'r', 'r']
        if plots:
            for i in range(6):
                ax[i].plot(x, F_unc[i,:], '--', color=colors[i])
                ax[i].plot(x, F_con[i,:], '-', color=colors[i])
            
        for i in range(6):
            F_con_extr[i] = interpolate.interp1d(x, F_con[i,:])
        
        x1 = np.linspace(-x_lim*1, x_lim*1, 2*x_lim/(bin_width/(1+z)/dl))
        if verbose:
            print('number of points = ', len(x1))
            
        if plots:
            for i in range(6):
                ax[i].plot(x1, F_con_extr[i](x1), 'o')
        
        if 0:
            F[0,0] = np.sum(F_con_extr[1](x1))
            F[1,0] = np.sum(F_con_extr[2](x1))
            F[0,1] = F[1,0]            
            F[1,1] = np.sum(F_con_extr[3](x1))
        else:
            F[0,0] = np.sum(F_con_extr[4](x1)**2)
            F[1,0] = np.sum(F_con_extr[4](x1)*F_con_extr[5](x1))
            F[0,1] = F[1,0]
            F[1,1] = np.sum(F_con_extr[5](x1)**2)
            F *= 2
        
        cov = np.abs(np.linalg.inv(F))
        
        if verbose:
            print('Fisher matrix:', F)
            print('Covariance matrix', cov)            
        
        if not cgs:
            dN = np.sqrt(np.abs(cov))[0,0]/ston
            db = np.sqrt(np.abs(cov))[1,1]/ston
        else:
            print(np.sqrt(np.abs(cov))[0,0]/ston)
            dN = N - np.log10(np.power(10, N) - np.sqrt(np.abs(cov))[0,0]/ston)
            db = np.sqrt(np.abs(cov))[1,1]/1e5/ston
        
        return dN, db, F, min(F_con[0,:])
        
if __name__ == '__main__':
    lines = []
    lines.append(line('Lya', 1215.6701, 0.41640, 6.265e8))
    #lines.append(line('Lyb', 1025.7223, 0.07912, 1.897e8))
    #lines.append(line('Lyg',  972.5368, 0.02900, 8.127e7))
    #lines.append(line('Lyd',  949.7431, 0.01394, 4.204e7))
    
    ston = 80
    
    if 1:
        b = 26
        N = 12.74
        
        #FisherbN(b, N, lines)
        dN, db, F, tau = FisherbN(b, N, lines, ston=ston, plots=1, cgs=0)
        print('tau(0) =', tau)
        print('N = ', N)
        print('dN = ', dN)
        print('b = ', b)
        print('ddb =', db)
        #data = genfromtxt('test1.csv', delimiter=',')
        #data = np.transpose(np.genfromtxt('log_res_sum.csv', delimiter=','))
        #print(data[1])
    if 0:
        #f_in = open(r'C:\university\NIR\work\summer_tests\test1.dat', 'r')
        #data = np.transpose(np.loadtxt(f_in))
        data = np.genfromtxt('J1444_sum_er.csv', delimiter=',')
        #data = np.genfromtxt('log_res_2_5_blend.csv', delimiter=',')
        print(data[1])
        def b(N):
            return 5*N-45
            #return ((5*N-45 + 3.0*norm.rvs())**2)**0.5
        fig, ax = plt.subplots()
        #b_grid = b(data[0])#np.linspace(20, 25, 20)
        b_grid = data[3]
        N_grid = data[0]#np.linspace(13, 14, 20)
        db, dN = np.meshgrid(N_grid, b_grid)
        tau_0 = np.zeros_like(db)
        er=np.zeros([4, len(data[0])])
        for ib, b in enumerate(b_grid):
            print(ib)
            for iN, N in enumerate(N_grid):
                if ib == iN:
                    dN[ib, iN], db[ib, iN], F, tau_0[ib, iN] = FisherbN(b, N, lines, ston=ston, verbose=0, plots=0, cgs=0)
                    er[3][iN] = db[ib, iN]
                    er[1][iN] = dN[ib, iN]
                    er[0][iN] = N
                    er[2][iN] = b                        

        #er=np.zeros([4, len(data[0])])    
        
        if 0:
            CS = ax.contourf(N_grid, b_grid, db, 100, cmap=plt.cm.jet, alpha=0.4)
            #CS = ax.contour(N_grid, b_grid, tau_0, 30, cmap=plt.cm.bone, alpha=1)
            cbar = plt.colorbar(CS)
        if 0:
            for iN, x in enumerate(N_grid):
                for ib, y in enumerate(b_grid):
                    if ib == iN:
                        plt.scatter(x, y, s=0)
                        ax.text(x, y, '{0:.2f}'.format(db[ib, iN]), va='center', ha='center')
                        er[3][iN] = db[ib, iN]
                        er[1][iN] = dN[ib, iN]
                        er[0][iN] = x
                        er[2][iN] = y
                        #er[4][iN] = ((5*x-45 + 3.0*norm.rvs())**2)**0.5
      

        for r in range(1, len(er)):
            for c in range(1, len(er[0])):
                er[r][c] = float(er[r][c])
        results = open('J1444_sum_er1.csv', 'w')
        writer = csv.writer(results)
        for row in er:
            writer.writerow(row)
        results.close()                  
    if 0:
        csvfile = open('output.csv', 'w', newline='')
        output = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        i = 0
        with open('log_res_sum_data_w_s_n.csv', newline='') as csvfile:
            for line in csv.reader(csvfile, delimiter=',', quotechar='|'):
                if ''.join(line).strip():
                    N, b, ston = float(line[1]), 10**float(line[2]), float(line[3])*3
                    print(N, b, ston)
                    i += 1
                    cov = np.abs(np.linalg.inv(FisherbN(b, N, lines, verbose=0, plots=0))/ston)
                    dN = np.log10(10**N + np.sqrt(cov)[0,0])-N
                    db = (np.sqrt(cov)[1,1])/1e5
                    if i in [190,191]:
                        cov = np.abs(np.linalg.inv(FisherbN(b, N, lines, verbose=1, plots=1))/ston)
                        print(dN, db, np.dot(np.linalg.inv(FisherbN(b, N, lines, verbose=0, plots=0)), FisherbN(b, N, lines, verbose=0, plots=0)))
                        input()
                    print(N, dN, b, db)
                    output.writerow([line[0], line[1], line[2], line[3], dN, db])         
        csvfile.close()