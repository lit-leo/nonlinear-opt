import os
from os.path import join

import numpy as np
from numpy import flatnonzero as find
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve

import pypower
from pypower.api import case9, loadcase, ext2int, makeYbus, makeSbus, bustypes
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS
from pypower.dSbus_dV import dSbus_dV

def case_as_data(casename):
    '''
    Loads the case according to its name.

    Possible names are: 
    ['case4gs', 'case6ww', 'case9',
    'case9Q', 'case14', 'case24_ieee_rts',
    'case30', 'case30Q', 'case30pwl',
    'case39', 'case57', 'case118', 'case300',
    'case30_userfcns'].
    '''

    # list of available cases
    cases = ['case4gs', 'case6ww', 'case9',
    'case9Q', 'case14', 'case24_ieee_rts',
    'case30', 'case30Q', 'case30pwl',
    'case39', 'case57', 'case118', 'case300',
    'case30_userfcns']

    if casename not in cases:
        print('Case not found, defaults to case9')
        casename_ = 'case9'
    else:
        casename_ = casename


    case_path = join('pypower', casename_)
    data = ext2int(loadcase(case_path))

    return data


class PowerflowData():
    def __init__(self, data):
        """Initialize internal structure with loaded case file"""
        self.baseMVA = data['baseMVA']
        self.bus = data['bus']
        self.branch = data['branch']
        self.gen = data['gen']

        self.ref, self.pv, self.pq = bustypes(self.bus, self.gen)
        self.pvpq = np.r_[self.pv, self.pq]
        self.npv = len(self.pv)
        self.npq = len(self.pq)
        self.j1 = 0;         self.j2 = self.npv           
        self.j3 = self.j2;        self.j4 = self.j2 + self.npq      
        self.j5 = self.j4;        self.j6 = self.j4 + self.npq
        ## j1:j2 - V angle of pv buses
        ## j3:j4 - V angle of pq buses
        ## j5:j6 - V mag of pq buses
        
        self.Y, _, _ = makeYbus(self.baseMVA, self.bus, self.branch)
        self.S = makeSbus(self.baseMVA, self.bus, self.gen)
        self.V0 = self.get_V0()
        
    def forward(self, V):
        """Calculate loss/mismatch given parameters vector V"""
        def mismatch(Y, V, S):
            return V * np.conj(Y * V) - S
        
        mis = mismatch(self.Y, V, self.S)
        F = np.r_[  mis[self.pv].real,
                 mis[self.pq].real,
                 mis[self.pq].imag  ]
        return F
    
    def grad(self, V):
        """
        Calculate Jacobi matrix given parameters vector V
        Name is such due to compatability with Conjugated Gradients Class
        """
        dS_dVm, dS_dVa = dSbus_dV(self.Y, V)
        J11 = dS_dVa[self.pvpq[:, None], self.pvpq].real
        J12 = dS_dVm[self.pvpq[:, None], self.pq].real
        J21 = dS_dVa[self.pq[:, None], self.pvpq].imag
        J22 = dS_dVm[self.pq[:, None], self.pq].imag

        J = vstack([
                hstack([J11, J12]),
                hstack([J21, J22])
            ], format="csr")

        return J
    
    def get_V0(self):
        on = find(self.gen[:, GEN_STATUS] > 0)      ## which generators are on?
        gbus = self.gen[on, GEN_BUS].astype(int) 
        V0  = self.bus[:, VM] * np.exp(1j * np.pi/180 * self.bus[:, VA])
        V0[gbus] = self.gen[on, VG] / abs(V0[gbus]) * V0[gbus]
        return V0
    
    def updatedV(self, V, dx):
        Vm = np.abs(V)
        Va = np.angle(V)
        if self.npv:
            Va[self.pv] = Va[self.pv] + dx[self.j1:self.j2]
        if self.npq:
            Va[self.pq] = Va[self.pq] + dx[self.j3:self.j4]
            Vm[self.pq] = Vm[self.pq] + dx[self.j5:self.j6]
        V = Vm * np.exp(1j * Va)
        return V
    
    def V2real(self, V):
        return np.r_[V[self.pv].real, V[self.pq].real, V[self.pq].imag]
    

class PowerflowOptimData(PowerflowData):
    def __init__(self, data):
        s = super().__init__(data)
    
    def forward(self, x):
        return np.power(super().forward(x), 2).sum()
    
    def mismatch(self, x):
        return super().forward(x)
    
    def jacobian(self, x):
        return super().grad(x)
    
    def grad(self, x):
        return 2 * super().grad(x).T @ super().forward(x)
    
    def normed_forward(self, x):
        """
        Calculate norm of the forward vector.
        Implemented for the comparison with the Newton method.
        """
        return np.linalg.norm(super().forward(x), 2)