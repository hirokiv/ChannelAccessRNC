import warnings
warnings.simplefilter("ignore",UserWarning)
import numpy as np
import pandas as pd
import sys
import time
from scipy import linalg

# Import EPICS channel access interface
from CaChannel import CaChannel, CaChannelException


#####################
# begin CaAccess class
#####################
class UserChannelAccess: # CHannel Access

  def __init__(self, caname, T=200, timeout=0.5, KFSET='OFF', sigv2=1, sigw2=2):
    self.caname = caname
    self.chan = CaChannel(caname)
    self.chan.setTimeout(timeout)
    self.error_flag = []
    #  chan = CaChannel('akito12Host:xxxExample')
    print("CA init:  " + str(self.caname))
    self.val = self.read_ca()
    self.T = T  # length of buffer
    self.buff = [np.nan]*self.T # buffer having T components
    self.buff[-1] = self.val


    # apply Kalman filter if KFSET is ON
    self.KFSET = KFSET
    if self.KFSET == 'ON':
      self.initialize_KF(sigv2, sigw2 )

  def initialize_KF(self, sigv2=1, sigw2=2):
    ## Kalman filter setup
    # for kalman filters later being introduced
    self.sigv2 = sigv2 # white noise variance
    self.sigw2 = sigw2 # observation variance

    self.estval = self.val # estimation value. initialize by the initial value fetched
    self.buff_est = np.array([np.nan]*self.T) # kalman filtered buffer having T components
    self.buff_est[-1] = self.estval
  

  def __del__(self):
        # body of destructor
    return

  def buffering(self):
    self.read_ca()
    self.buff = self.buff[1:]
    self.buff.append( self.fetch() )

    if self.KFSET == 'ON':
      self.buffering_KF()

  def buffering_ave(self,num,delay,verbose=0):
    self.read_ca_ave(num,delay,verbose)
    self.buff = self.buff[1:]
    self.buff.append( self.fetch() )

    if self.KFSET == 'ON':
      self.buffering_KF()



  # Add function of fetching kalman filter
  def buffering_KF(self):
    if (self.sigv2 == 1) and (self.sigw2 == 2):
      # the result of riccati equation for A=b=c=1 is p= 2 and -1 (Refer to S. Adachi etc.)
      p = 2
    else:
#      print('Not implemented yet. Use quadratic programing for solving riccati equation')
#      print('-p^2 + sigv2*(p+sigw2) = 0')
      p = linalg.solve_discrete_are(1,1,self.sigv2, self.sigw2)
#      print('p = '+  str(p))

    # kalman gain
    g = p / (p + self.sigw2)
    self.estval = self.estval + g * (self.val - self.estval)
    self.buff_est = self.buff_est[1:]
    #self.buff_est.append(  self.fetch_est()  )
    self.buff_est = np.append( self.buff_est,  self.fetch_est()  )




  def read_ca(self):

    try:
      if isinstance(self.caname,str):
        self.chan.searchw()
        self.chan.pend_io()
        a=self.chan.getw()
        self.val = a
        self.error_flag = 0
        return a
      else:
        print('Channel definition false : specify string')
        sys.exit()
    except CaChannelException as e:
       print(e)
       print('Error in reading Channel ' + self.caname)
       self.error_flag = 1
       self.val = np.nan

  # Read ca several times and average
  def read_ca_ave(self,num,delay,verbose=0):
    try:
      if isinstance(self.caname,str):
        a = np.empty_like(range(num),dtype=np.float)
        self.chan.searchw()
        self.chan.pend_io()

        for i in range(num):
          a[i]=self.chan.getw()
          time.sleep(delay)

        a_ave = np.average(a)
        self.val = a_ave
        self.error_flag = 0

        if verbose==1:
          print('\na_array')
          print(a)
          print('\na_ave')
          print(a_ave)
          print('\n\n')

        return a
      else:
        print('Channel definition false : specify string')
        sys.exit()
    except CaChannelException as e:
       print(e)
       print('Error in reading Channel ' + self.caname)
       self.error_flag = 1
       self.val = np.nan


  def put_ca(self,putval):

    try:
      if isinstance(self.caname,str):
        self.chan.searchw()
        self.chan.putw(putval)
        self.chan.pend_io()
        self.val = self.chan.getw()
#        print('self.val = ' + str(self.val))
#        print('putval   = ' + str(putval))
#        if self.val != putval:
#          print('Error in putting value to ' + self.caname)
#          sys.exit()
      else:
        print('Channel definition false : specify string')
    except CaChannelException as e:
       print(e)
       print('Error in Putting Channel ' + self.caname)
       sys.exit()


  def show_caname(self):
    return self.caname

  def fetch(self):
    return self.val

  def fetch_est(self):
    return self.estval


  def show(self):
    print(self.val)

  def plot_buffer(self,ax):
#    c1,c2,c3,c4 = "blue","green","red","black"
  #  l1,l2,l3,l4 = "","","",""
    ax.plot(range(self.T), self.buff,  label=self.caname, marker="o")

    if self.KFSET == 'ON':
      ax.plot(range(self.T), self.buff_est.flatten(), label=self.caname+'_est', marker="o")

    ax.set_xlabel('BUFF_INDEX')
    ax.set_ylabel('')
    ax.legend(loc=0,frameon=False)
  #  ax1 = fig.add_subplot(2,1,1)

    return ax

