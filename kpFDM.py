import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh, lobpcg
#from scipy.sparse.linalg.eigen.arpack import eigsh
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class UniversalConst(object):

    def __init__(self):
        """
        """
        self.RY = 13.6
        self.A0 = 0.529167
        self.eVAA2 = 3.67744

class DrawingFunctions(object):

   def __init__(self):
      """
      Parameters
      ----------
      dimen : integer
              system's dimensionality, i.e., quantum well, wire and dot
      """
 
   def square(self, params, profile, value, idx):
      """

      """

      if params['dimen'] == 1:
  
        profile[params['startPos'][idx]:params['endPos'][idx]] = value
 
      if params['dimen'] == 2:
         
        profile[params['startPos'][idx]:params['endPos'][idx],params['startPos'][idx]:params['endPos'][idx]] = value

      if params['dimen'] == 3:
         print 'Not implemented yet'


      return profile

class Potential(object):

    def __init__(self, params):
      """
      """
      
      # Set internal parameters
      
      if params['dimen'] == 1:
        if params['model'] == 'ZB2x2':
            self.pot = np.zeros(len(params['x']))
        if params['model'] == 'ZB6x6':
            self.pot = np.zeros(3*len(params['x'])).reshape((3,len(params['x'])))
      
      if params['dimen'] == 2:
        if params['model'] == 'ZB2x2':
            self.pot = np.zeros(len(params['x'])**2).reshape(len(params['x']),len(params['x']))
        if params['model'] == 'ZB6x6':
            self.pot = np.zeros(3*len(params['x'])**2).reshape((3,len(params['x']),len(params['x'])))

    def buildPot(self, params, flag):
      """
      """
      drawFunc = DrawingFunctions()
      
      
      if params['potType'] == 'square':

          if params['model'] == 'ZB2x2':
              
              for i in range(params['nmat']):
                  #print 'constructing layer %d potential from %d(%f) to %d(%f)'%(i,params['startPos'][i],params['x'][params['startPos'][i]],
                  #params['endPos'][i],params['x'][params['endPos'][i]])

                  if flag == 'het':
                    value = params['gap'][0] + (params['gap'][i]-params['gap'][0])*(1.-params['bshift'])
                  if flag == 'kin':
                    value = params['elecmassParam'][i]

                  self.pot = drawFunc.square(params, self.pot, value, i) 
          
          if params['model'] == 'ZB6x6':

              value = {}
              value[2] = 0

              for i in range(params['nmat']):
                  #print 'constructing layer %d potential from %d(%f) to %d(%f)'%(i,params['startPos'][i],params['x'][params['startPos'][i]],
                  #params['endPos'][i],params['x'][params['endPos'][i]])

                  if flag == 'het':
                    value[0] = (params['gap'][0]-params['gap'][i])*params['bshift']
                    value[1] = value[0]
                    value[2] = value[0]-params['deltaSO'][i]
                  if flag == 'kin':
                    value[0] = params['gamma1'][i]
                    value[1] = params['gamma2'][i]
                    value[2] = params['gamma3'][i]
                  
                  for j in range(3):
                    self.pot[j,:] = drawFunc.square(params, self.pot[j,:], value[j], i)
                  
      else:
          print 'Not implemented yet'

      return self.pot


    def plotPot(self, params):
      """
      """
      UniConst = UniversalConst()
      fig = plt.figure()
      
      if params['model'] == 'ZB2x2':
        if params['dimen'] == 1:
          plt.plot(params['x']*UniConst.A0,self.pot)
          plt.show()
        if params['dimen'] == 2:
          ax = fig.gca(projection='3d')
          X, Y = np.meshgrid(params['x']*UniConst.A0, params['x']*UniConst.A0)
          surf = ax.plot_surface(X, Y, self.pot, rstride=1, cstride=1, cmap=cm.coolwarm,
                  linewidth=0, antialiased=False)
          
          ax.zaxis.set_major_locator(LinearLocator(10))
          ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
          
          fig.colorbar(surf, shrink=0.5, aspect=5)
          
          plt.show()
      if params['model'] == 'ZB6x6':
        plt.plot(params['x']*UniConst.A0,self.pot[0,:])
        plt.plot(params['x']*UniConst.A0,self.pot[1,:])
        plt.plot(params['x']*UniConst.A0,self.pot[2,:])
        plt.show()

class IO(object):

    def __init__(self):
        """
        """

        self.const = UniversalConst()
        self.parser = argparse.ArgumentParser(description='kp in real space by FDM')

        self.parameters = {}

        self.parse()

    def parse(self):
        """
        """
        self.parser.add_argument('-n','--NMAT',action='store', dest='nmat',
                                  type=int, help='Type: integer. Number of material which composes the system.')

        self.parser.add_argument('-d','--DIMEN',action='store', dest='dimen',
                                  type=int,
                                  help="""
                                  Type: integer. Dimensionality of the system, i.e.,
                                    1 -> quantum well,
                                    2 -> quantum wire,
                                    3 -> quantum dot.
                                    """)
        self.parser.add_argument('-m','--MODEL',action='store', dest='model',type=str,
                                 choices=['ZB2x2', 'ZB6x6', 'ZB8x8'], help="""
                                 Hamiltonian model: ZB2x2, ZB6x6, ZB8x8, WZ2x2, ....
                                 """)

        self.parser.add_argument('-p','--GAPS',action='store', dest='gap',
                                  type=float, nargs='+', help="""
                                  Material's gaps in the form: gap_material1, gap_material2, gap_material3, ... .
                                  Units of eV.
                                  """)

        self.parser.add_argument('-t','--type',action='store',dest='potType',
                                 type=str, help="""
                                 Potential type, i.e.,
                                 square, hexagon, circle, sphere, etc
                                 """)

        self.parser.add_argument('-s','--STARTPOS',action='store',dest='startPos',
                                  type=float, nargs='+', help="""
                                  Starting position of material: v1, v2, v3, ... .
                                  Units of Angstroms
                                  """)

        self.parser.add_argument('-e','--ENDPOS',action='store',dest='endPos',
                                  type=float, nargs='+', help="""
                                  Ending position of the band mismatch in the form: v1, v2, v3, ... .
                                  Units of Angstroms
                                  """)

        self.parser.add_argument('-st','--STEP',action='store',dest='step',type=float, help="""
                                 Discretization step, usually values like: 0.5, 0.25, ...
                                """)

        self.parser.add_argument('-dr','--DIRECTION',action='store',dest='direction',type=str, help="""
                                 Direction to compute the band structure, ex: kx, ky ou kz
                                """)

        self.parser.add_argument('-me','--ELECMASS',action='store',dest='elecmass',
                                  type=float, nargs='+', help="""
                                  Electron effective masses in the form: v1, v2, v3, ... .
                                  Units of m0. If WZ input e1 and e2 for all materials.
                                  """)
                                  
        self.parser.add_argument('-g1', '--gamma1',action='store',dest='gamma1',
                                  type=float, nargs='+', help="""
                                  Holes mass parameters in the form: gamma1, gamma1, ...
                                  """)
        
        self.parser.add_argument('-g2', '--gamma2',action='store',dest='gamma2',
                                  type=float, nargs='+', help="""
                                  Holes mass parameters in the form: gamma2, gamma2, ...
                                  """)
                                  
        self.parser.add_argument('-g3', '--gamma3',action='store',dest='gamma3',
                                  type=float, nargs='+', help="""
                                  Holes mass parameters in the form: gamma3, gamma3, ...
                                  """)                          

        self.parser.add_argument('-dso', '--DELTASO', action='store', dest='deltaSO',
                                  type=float, nargs='+', help="""
                                  Delta SO in the form: deltaso1, deltaso2, ...
                                  """)


        self.parser.add_argument('-lp','--LATPAR', action='store',dest='latpar',
                                  type=float, help="""
                                  General lattice parameter. Units of Angstrom
                                  """)

        self.parser.add_argument('-np', '--NPOINTS', action='store', dest='npoints',
                                  type=int, help="""
                                  Number of k points to compute
                                  """)

        self.parser.add_argument('-pc', '--PERCENTAGE', action='store', dest='percent',
                                  type=float, help="""
                                  Percentage of the k-mesh to compute
                                  """)

        self.parser.add_argument('-bs', '--BSHIFT', action='store', dest='shift',
                                  type=float, help="""
                                  Band shift in percentage for valance band, ex: 0.4
                                  """)

        self.parser.add_argument('-ncb', '--NUMCB', action='store', dest='numcb',
                                  type=int, help="""
                                  Number of conduction bands to compute
                                  """)

        self.parser.add_argument('-nvb', '--NUMVB', action='store', dest='numvb',
                                  type=int, help="""
                                  Number of valance bands to compute
                                  """)
                                

        self.args = self.parser.parse_args()

        self.verification()

        self.buildParamDict()

    def verification(self):
        """
        """
        assert self.args.nmat > 1
        assert self.args.dimen >= 1
        assert (self.args.potType == 'square')

        assert self.args.model in ['ZB2x2', 'ZB6x6', 'ZB8x8']

        if self.args.model == 'ZB2x2':
          assert self.args.numcb > 0
        else:
          if self.args.model in ['ZB6x6', 'ZB8x8']:
            assert self.args.numvb > 0
            assert len(self.args.gamma1) == self.args.nmat
            assert len(self.args.gamma2) == self.args.nmat
            assert len(self.args.gamma3) == self.args.nmat
            assert len(self.args.deltaSO) == self.args.nmat


        if self.args.model in ['ZB2x2', 'ZB6x6', 'ZB8x8']:
            assert len(self.args.gap) == self.args.nmat
            assert len(self.args.elecmass) == self.args.nmat
        else:
            assert len(self.args.gap)/2 == self.args.nmat
            assert len(self.args.elecmass)/2 == self.args.nmat

        assert self.args.direction in ['kx', 'ky', 'kz']


        #self.args.potValue = [x/const.RY for x in self.args.potValue]
        #self.args.startPos = [x/const.A0 for x in self.args.startPos]
        #self.args.endPos = [x/const.A0 for x in self.args.endPos]
        #self.args.elecmass = [x/const.eVAA for x in self.args.elecmass]
        #self.args.latpar = self.args.latpar/const.A0

    def buildParamDict(self):
        """
        """

        # Primary parameters

        self.parameters['nmat'] = self.args.nmat
        self.parameters['dimen'] = self.args.dimen
        self.parameters['model'] = self.args.model
        self.parameters['gap'] = self.args.gap
        self.parameters['potType'] = self.args.potType
        self.parameters['startPos'] = self.args.startPos
        self.parameters['endPos'] = self.args.endPos
        self.parameters['step'] = self.args.step
        self.parameters['direction'] = self.args.direction
        self.parameters['elecmass'] = self.args.elecmass
        self.parameters['gamma1'] = self.args.gamma1
        self.parameters['gamma2'] = self.args.gamma2
        self.parameters['gamma3'] = self.args.gamma3
        self.parameters['deltaSO'] = self.args.deltaSO 
        self.parameters['latpar'] = self.args.latpar
        self.parameters['npoints'] = self.args.npoints
        self.parameters['percentage'] = self.args.percent
        self.parameters['bshift'] = self.args.shift
        self.parameters['numcb'] = self.args.numcb
        self.parameters['numvb'] = self.args.numvb


        # Secondary parameters

        self.parameters['N'] = (self.parameters['endPos'][0]-self.parameters['startPos'][0]+1)/self.parameters['step']

        if self.parameters['dimen'] in [1, 2]:
            self.parameters['x'] = np.linspace(self.parameters['startPos'][0],self.parameters['endPos'][0],self.parameters['N'])
            self.parameters['x'] /= self.const.A0
        else:
            print 'Not implemented yet'

        self.parameters['elecmassParam'] = [self.const.eVAA2/x for x in self.parameters['elecmass']]

        # Converting startPos and endPos to vector index

        startPos = np.zeros(self.parameters['nmat'])
        endPos = np.zeros(self.parameters['nmat'])

        if self.parameters['dimen'] in [1,2]:
            for i in range(self.parameters['nmat']):
                startPos[i] = abs((self.parameters['startPos'][0]-self.parameters['startPos'][i])*self.parameters['N']/(self.parameters['endPos'][0]-self.parameters['startPos'][0]))
                endPos[i] = startPos[0]+abs(self.parameters['endPos'][0]+self.parameters['endPos'][i])*self.parameters['N']/(self.parameters['endPos'][0]-self.parameters['startPos'][0])
        else:
            print 'Not implemented yet'

        self.parameters['startPos'] = [int(x) for x in startPos]
        self.parameters['endPos'] = [int(x) for x in endPos]

        return self.parameters



class ZincBlend(object):

   def __init__(self, params):
      """
      """
      
      self.UniConst = UniversalConst()
      self.symmetry = 'cubic'

      if params['dimen'] == 1:
          self.directbasis = np.array([params['latpar']/self.UniConst.A0, params['latpar']/self.UniConst.A0, (params['N']-1)*params['step']/self.UniConst.A0])
      else:
          print 'Not implemented yet'

      # set Bravais basis in reciprocal space
      self.reciprocalbasis = 2.*np.pi/self.directbasis

      if params['dimen'] == 1:
        if params['direction'] == 'kx':
            kxlim = self.reciprocalbasis[0]*params['percentage']/100
            self.kmesh = np.linspace(0,kxlim,params['npoints'])
        if params['direction'] == 'ky':
            kylim = self.reciprocalbasis[1]*params['percentage']/100
            self.kmesh = np.linspace(0,kylim,params['npoints'])
      else:
          print 'Not implemented yet'


class ZBHamilton(ZincBlend):

    def __init__(self, params, potHet, Kin):
      """
      """
    
      ZincBlend.__init__(self,params)
      self.potHet = potHet
      self.Kin = Kin
         
      
    def buildHam(self, params, kpoints):
      """
      """
      
      kx = kpoints[0]
      ky = kpoints[1]
      kz = kpoints[2]
      
      
      HT = []
      
      if params['model'] == 'ZB2x2':

        if params['dimen'] == 1:
      
          A = np.diagflat(self.Kin)
          ksquare = kx**2 + ky**2
          
          # derivatives related terms
          
          nonlocal_diag = np.convolve(self.Kin,[1,2,1],'same')
          nonlocal_off = np.convolve(self.Kin,[1,1],'valid')
          
          nonlocal = (1./(2.*params['step']**2))*(np.diagflat(nonlocal_diag) - np.diagflat(nonlocal_off,1) - np.diagflat(nonlocal_off,-1))
          
          HT = A*ksquare + nonlocal + np.diagflat(self.potHet)

        if params['dimen'] == 2:
      
          A = np.diagflat(self.Kin)
          ksquare = kz**2
          
          # derivatives related terms
          
          nonlocal_diag = np.convolve(self.Kin,[1,2,1],'same')
          nonlocal_off = np.convolve(self.Kin,[1,1],'valid')
          
          nonlocal = (1./(2.*params['step']**2))*(np.diagflat(nonlocal_diag) - np.diagflat(nonlocal_off,1) - np.diagflat(nonlocal_off,-1))
          
          HT = local + nonlocal + np.diagflat(self.potHet)
  
      if params['model'] == 'ZB6x6':
        
        gamma1 = np.diagflat(self.Kin[0,:])
        gamma2 = np.diagflat(self.Kin[1,:])
        gamma3 = np.diagflat(self.Kin[2,:])
      
    
        POT = linalg.block_diag(np.diagflat(self.potHet[0,:]), np.diagflat(self.potHet[1,:]), 
                   np.diagflat(self.potHet[1,:]), np.diagflat(self.potHet[0,:]),
                   np.diagflat(self.potHet[2,:]), np.diagflat(self.potHet[2,:]))
        
        nonlocal_diag = -np.convolve(self.Kin[0,:]-2.*self.Kin[1,:],[1,2,1],'same')
        nonlocal_off = -np.convolve(self.Kin[0,:]-2.*self.Kin[1,:],[1,1],'valid')
        nonlocal = (1./(2.*params['step']**2))*(np.diagflat(nonlocal_diag) - np.diagflat(nonlocal_off,1) - np.diagflat(nonlocal_off,-1))
      
        Q = -(gamma1+gamma2)*kx**2 - (gamma1+gamma2)*ky**2 + nonlocal
        
        nonlocal_diag = -np.convolve(self.Kin[0,:]+2.*self.Kin[1,:],[1,2,1],'same')
        nonlocal_off = -np.convolve(self.Kin[0,:]+2.*self.Kin[1,:],[1,1],'valid')
        nonlocal = (1./(2.*params['step']**2))*(np.diagflat(nonlocal_diag) - np.diagflat(nonlocal_off,1) - np.diagflat(nonlocal_off,-1))
      
        T = -(gamma1-gamma2)*kx**2 - (gamma1-gamma2)*ky**2 + nonlocal
        
        nonlocal_off = np.convolve(self.Kin[2,:],[1,1],'valid')
        nonlocal = (1./(4.*params['step']))*(np.diagflat(nonlocal_off,1) - np.diagflat(nonlocal_off,-1))
      
        S = 2.*np.sqrt(3.)*complex(kx,ky)*nonlocal
        SC = np.conjugate(S)
        
        R = -2.*np.sqrt(3.)*(gamma2*(kx**2 + ky**2)-complex(0,1.)*gamma3*kx*ky)
        RC = np.conjugate(R)
      
        ZERO = np.zeros(params['N']**2).reshape((params['N'],params['N']))
        
        
        HT = np.bmat([[Q   , S   , R   , ZERO, complex(0,1./np.sqrt(2.))*S    , -complex(0,np.sqrt(2.))*R      ],
                      [SC  , T   , ZERO, R   , -complex(0,1./np.sqrt(2))*(Q-T), complex(0,np.sqrt(3./2.))*S    ],
                      [RC  , ZERO, T   , -S  , -complex(0,np.sqrt(3./2.))*SC  , -complex(0,1./np.sqrt(2))*(Q-T)],
                      [ZERO, RC  , -SC , Q   , -complex(0,np.sqrt(2.))*RC     , -complex(0,1./np.sqrt(2.))*SC  ],
                      [-complex(0,1./np.sqrt(2.))*SC, complex(0,1./np.sqrt(2))*(Q-T), complex(0,np.sqrt(3./2.))*S,
                        complex(0,np.sqrt(2.))*R, (1./2.)*(Q+T), ZERO ],
                      [complex(0,np.sqrt(2.))*RC, -complex(0,np.sqrt(3./2.))*SC, complex(0,1./np.sqrt(2))*(Q-T),
                        complex(0,1./np.sqrt(2.))*S, ZERO, (1./2.)*(Q+T) ]
                      ])
        
        HT = np.asarray(HT)
        
        #np.savetxt('HT.dat',HT)
        
        HT += POT
        
        #np.savetxt('pot.dat',POT)
        
      return HT


    def solve(self,params):
      """
      """
      
      if params['model'] == 'ZB2x2':
        va = np.zeros(len(self.kmesh)*int(params['numcb'])).reshape((int(params['numcb']),len(self.kmesh)))
        ve = np.zeros(len(self.kmesh)*int(params['numcb'])*int(params['N'])).reshape((int(params['N']),int(params['numcb']),len(self.kmesh)))
      
      if params['model'] == 'ZB6x6':
        va = np.zeros(len(self.kmesh)*int(params['numvb'])).reshape((int(params['numvb']),len(self.kmesh)))
        ve = np.zeros(len(self.kmesh)*int(params['numvb'])*6*int(params['N'])).reshape((6*int(params['N']),int(params['numvb']),len(self.kmesh)))
        #X = np.zeros(int(params['numvb'])*6*int(params['N'])).reshape((6*int(params['N']),int(params['numvb'])))
      
      #cb_va = 'cb_values'+params['direction']+'.txt.gz'
      #cb_ve = 'cb_vector'+params['direction']+'.txt.gz'
      
      #np.savetxt(cb_va,header=params['direction']+', cb1, cb2, ... , cb'+str(cbnum))
      #np.savetxt(cb_ve,header=params['direction']+', cb1, cb2, ... , cb'+str(cbnum))
      
      for i in range(len(self.kmesh)):
        
        if params['direction'] == 'kx':
          kpoints = np.array([self.kmesh[i],0,0])
        if params['direction'] == 'ky':
          kpoints = np.array([0,self.kmesh[i],0])
        
        print "Solving k = ",kpoints 
        
        HT = self.buildHam(params,kpoints)
        
        if params['model'] == 'ZB2x2':
          va[:,i], ve[:,:,i] = eigsh(HT, int(params['numcb']), which='SM')
        
        if params['model'] == 'ZB6x6':
          va[:,i], ve[:,:,i] = eigsh(HT, int(params['numvb']), which='LA')
          #va[:,i], ve[:,:,i] = lobpcg(HT**2, X, M=None, tol=10e-6, largest=True, verbosityLevel=1)
          
        if params['model'] == 'ZB8x8':
          print 'Not implemented yet'
          
        #va[:,i] = va_aux
        #ve[:,:,i] = ve_aux
        
      return self.kmesh, va, ve
        

#===============================================


UniConst = UniversalConst()
ioObject = IO()

#print ioObject.parameters


# Building potential
P = Potential(ioObject.parameters)
pothet = P.buildPot(ioObject.parameters,'het')

# Building effective mass variation
#K = Potential(ioObject.parameters)
#kin = K.buildPot(ioObject.parameters,'kin')

P.plotPot(ioObject.parameters)
#K.plotPot(ioObject.parameters)

"""
# Set parameters to Hamiltonian
ZB = ZBHamilton(ioObject.parameters, pothet, kin)


k, va, ve = ZB.solve(ioObject.parameters)

#print va

if ioObject.parameters['model'] == 'ZB2x2':

  fig = plt.figure()
  for i in range(ioObject.parameters['numcb']):
    plt.plot(k, np.sqrt(va[i,:]))
  plt.show()

if ioObject.parameters['model'] == 'ZB6x6':

  fig = plt.figure()
  for i in range(ioObject.parameters['numvb']):
    plt.plot(k, np.sqrt(va[i,:]))
  plt.show()
"""
