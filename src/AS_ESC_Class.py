# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 08:53:08 2021

@author: 236141 
modified by Hiroki Fujii, Nishina accelrator center, Riken
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import ParameterHandler as PH

font_size = 16
plt.rcParams.update({'font.size': font_size})

class ES_Algorithm_Parent:

  def __init__(self,ES_steps,p_max,p_min,kES, oscillation_size=0.0, decay_rate=0.999, observation = 0):

    # Total number of ES steps to take
    self.ES_steps = ES_steps
    self.p_max = p_max
    self.p_min = p_min
     # ES feedback gain kES
    self.kES = kES
    # This takes some trial and error to find a good value
  
    # Number of parameters being tuned
    self.nES = len(p_min)
    
    # Average values for normalization
    self.p_ave = (p_max + p_min)/2.0
    
    # Difference for normalization
    self.p_diff = self.p_max - self.p_min
    
    # Now we define some ES parameters
    
    # This keeps track of the history of all of the parameters being tuned
    self.pES = np.zeros([self.ES_steps+1,self.nES])
    self.pES[:] = np.nan
    
    # Start with initial conditions inside of the max/min bounds
    # In this case I will start them near the center of the range
    self.pES[0] = self.p_ave
    
    # This keeps track of the history of all of the Normalized parameters being tuned
    self.pES_n = np.zeros([self.ES_steps+1,self.nES])
    self.pES_n[:] = np.nan
    
    # Calculate the mean value of the initial condtions
    self.pES_n[0] = self.p_normalize(self.pES[0])
    
    # This keeps track of the history of the measured cost function
    self.cES = [np.nan]*(self.ES_steps+1) # np.zeros(self.ES_steps+1)

    # Calculate the initial cost function value based on initial conditions
    self.cES[0] = self.f_ES_minimize(self.pES[0],0,observation)    

    # ES dithering frequencies
    self.wES = np.linspace(1.0,1.75,self.nES)
    
    # ES dt step size
    self.dtES = 2*np.pi/(10*np.max(self.wES))
      
    # ES dithering size
    # In normalized space, at steady state each parameter will oscillate
    # with an ampltidue of \sqrt{aES/wES}, so for example, if you'd like 
    # the parameters to have normalized osciallation sizes you 
    # choose the aES as:
    self.oscillation_size = oscillation_size
    self.aES = self.wES*(self.oscillation_size)**2
    # Note that each parameter has its own frequency and its own oscillation size
   
    # Decay rate of dithering signal
    self.decay_rate = decay_rate
    
    # Decay amplitude
    self.amplitude = 1.0
    
    # parameter handler to edit amplitude
    amp_range = [0.05,1]
    self.ph_amp   = PH.ParameterHandler('amplitude', self.amplitude, prange=amp_range)
    drate_range = [0.9,1.0]
    self.ph_decay = PH.ParameterHandler('decay_rate', self.decay_rate, drate_range)



  # Function that normalizes paramters
  def p_normalize(self,p):
      p_norm = 2.0*(p-self.p_ave)/self.p_diff
      return p_norm
  
  # Function that un-normalizes parameters
  def p_un_normalize(self,p):
      p_un_norm = p*self.p_diff/2.0 + self.p_ave
      return p_un_norm
    
    # Normalization allows you to easily handle a group of parameters
    # which might have many orders of magnitude difference between their values
    # with this normalization the normalized values live in [-1,1]
  
  # This function defines one step of the ES algorithm at iteration i
  def ES_step(self,p_n,i,cES_now,amplitude):
      
      # ES step for each parameter
      p_next = np.zeros(self.nES)
      
      # Loop through each parameter
      for j in np.arange(self.nES):
          p_next[j] = p_n[j] + self.amplitude*self.dtES*np.cos(self.dtES*i*self.wES[j]+self.kES*cES_now)*(self.aES[j]*self.wES[j])**0.5
      
          # For each new ES value, check that we stay within min/max constraints
          if p_next[j] < -1.0:
              p_next[j] = -1.0
          if p_next[j] > 1.0:
              p_next[j] = 1.0
              
      # Return the next value
      return p_next
    
  def ES_main_loop(self, iter, observation=0):
    # Now we start the ES loop
        
    # Normalize previous parameter values
    self.pES_n[iter] = self.p_normalize(self.pES[iter])
    
    # Take one ES step based on previous cost value
    self.pES_n[iter+1] = self.ES_step(self.pES_n[iter],iter,self.cES[iter],self.amplitude)
    
    # Un-normalize to physical parameter values
    self.pES[iter+1] = self.p_un_normalize(self.pES_n[iter+1])
    
    # Calculate new cost function values based on new settings
    self.cES[iter+1] = self.f_ES_minimize(self.pES[iter+1],iter+1, observation) 
    
    # Decay the amplitude
    self.amplitude = self.amplitude * self.decay_rate
    # read and write parameter to file s.t. you can change amplitude by editing the amplitude.txt file
    self.amplitude = self.ph_amp.renew_parameter(self.amplitude) 
    self.decay_rate = self.ph_decay.renew_parameter(self.decay_rate) 
    # prevent amplitude goes up more than the initial setup
    if (self.amplitude > 1.0):
      self.amplitude = 1

    return self.pES[iter+1]
      
  def Plot_absolute_params(self, fig):

#    # These are the unknown optimal values (just for plotting)
#    p_opt = np.zeros([self.ES_steps,self.nES])
#    p_opt[:,0] = 0.99e6 -  0.99e6*0.5*(1-np.exp(-np.arange(self.ES_steps)/1000))
#    p_opt[:,1] = 1.5 + np.zeros(self.ES_steps)
#    p_opt[:,2] = 300 + np.zeros(self.ES_steps)
#    p_opt[:,3] = -5  + np.zeros(self.ES_steps)
 
    # Plot some results
    plt.subplot(5,1,1)
    plt.title(f'$k_{{ES}}$={self.kES}, $a_{{ES}}$={self.aES}')
    plt.plot(self.cES)
    plt.ylabel('ES cost')
    plt.xticks([])

    
    plt.subplot(5,1,2)
    plt.plot(self.pES[:,0],label='$p_{ES,1}$')
#    plt.plot(p_opt[:,0],'k--',label='$p_{ES,1}$ opt')
    plt.legend(frameon=False)
    plt.ylabel('ES parameter 1')
    plt.xticks([])
    
    plt.subplot(5,1,3)
    plt.plot(self.pES[:,1],label='$p_{ES,2}$')
#    plt.plot(p_opt[:,1],'k--',label='$p_{ES,2}$ opt')
    plt.legend(frameon=False)
    plt.ylabel('ES parameter 2')
    plt.xticks([])
    
# show just 2
#       plt.subplot(5,1,4)
#       plt.plot(self.pES[:,2],label='$p_{ES,3}$')
#   #    plt.plot(p_opt[:,2],'k--',label='$p_{ES,3}$ opt')
#       plt.legend(frameon=False)
#       plt.ylabel('ES parameter 3')
#       plt.xticks([])
#       
#       plt.subplot(5,1,5)
#       plt.plot(self.pES[:,3],label='$p_{ES,4}$')
#   #    plt.plot(p_opt[:,3],'k--',label='$p_{ES,4}$ opt')
#       plt.legend(frameon=False)
#       plt.ylabel('ES parameter 4')
#       plt.xlabel('ES step')
#       
    plt.tight_layout()

    return fig


  def Plot_normalized_params(self, fig):
    ax1 = plt.subplot(2,1,1)
    ln1 = ax1.plot(np.multiply(self.kES, self.cES),label='$k_{{ES}} C_{{ES}}$ ')
    ax1.set_title(f'$k_{{ES}}$={self.kES}, $a_{{ES}}$={self.aES}')
    ax1.set_ylabel('$k_{{ES}} C_{{ES}}$')
#    ax1.xticks([])

# show all the input parameters to be tuned
    ax2 = plt.subplot(2,1,2)
    
    for idx in np.arange(self.nES):
      label_str = '$p_{ES,n}$' + '[{}]'.format(str(idx))
      ax2.plot(self.pES_n[:,idx],label=label_str)
#       ax2.plot(self.pES_n[:,1],label='$p_{ES,2,n}$')
#       ax2.plot(self.pES_n[:,2],label='$p_{ES,3,n}$')
#       ax2.plot(self.pES_n[:,3],label='$p_{ES,4,n}$')

    ax2.plot(1.0+0.0*self.pES_n[:,0],'r--',label='bounds')
    ax2.plot(-1.0+0.0*self.pES_n[:,0],'r--')
#    ax2.legend(frameon=False)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
    ax2.set_ylabel('Normalized Parameters')
    ax2.set_xlabel('ES step')
    
    plt.tight_layout()
    
    return fig, ax1, ax2



  # This is the unknown time-varying function being minimized
  def f_ES_minimize(self,p,i):
#       f_val = 1e-4*np.abs(p[0]- (0.99e6-0.99e6*0.5*(1-np.exp(-i/1000))))+(p[1]-1.5)**2+2*p[2]-10.0*np.exp(-2*(p[3]+5.0)**2)
      f_val = 0
      return f_val
 


#########################################################################################
  #  design your own ES cost function
  # Effect of Barrier function included
#########################################################################################
class ES_Algorithm( ES_Algorithm_Parent ):

  def __init__(self,ES_steps,p_max,p_min,kES, oscillation_size=0.0, decay_rate=0.999, observation= 0):
    super(ES_Algorithm, self).__init__(ES_steps,p_max,p_min,kES, oscillation_size, decay_rate, observation) #super()
    self.Bx = 0
    self.Bx_list = [np.nan]*(self.ES_steps+1)

  # this is simple sum of barrier functions
  def Set_Barrier(self, val, iter):
    self.Bx = val
    self.Bx_list[iter+1] = val

  ## function below activates adaptive weightings of barrier functions
  def DefineBarrierWeights(self, Blist):
    self.w = {} # empty weight for baffle values
    for Blist_key in Blist.keys():
        self.w[Blist_key] = {}
        for RLUD in Blist[Blist_key].keys():
          # temporaliry define all weights to be 1
          self.w[Blist_key][RLUD] = 1.0

  def DefineBxlistBuffer(self, Blist, Bx_buffsize):
     self.Blist_buff = {}
     for Blist_key in Blist.keys():
        self.Blist_buff[Blist_key] = {}
        for RLUD in Blist[Blist_key].keys():
          # temporaliry define all weights to be 1
          # generate Bx buffer
          self.Blist_buff[Blist_key][RLUD] = np.ones(Bx_buffsize) * Blist[Blist_key][RLUD]


  def UpdateWeights(self, iter, bfname, reg_sig2=0.001, maven=20):
    # define moving average
    # def moving_average(data_set, periods):
    #     weights = np.ones(periods) / periods
    #     return np.convolve(data_set, weights, mode='valid')
    def moving_average(data, N):
      y_padded = np.pad(data, (N//2, N-1-N//2), mode='edge')
      y_smooth = np.convolve(y_padded, np.ones((N,))/N, mode='valid')
      return y_smooth

    # initialize 
    sig2 = {}
    wnorm_post = 0
    wnorm_pre  = 0
    # fetch prior w
    weight_pre = self.w[bfname].copy() 
    alpha_reg = reg_sig2; # regulation factor

    for key in weight_pre.keys(): # should be RLUD
      Bmave = moving_average(self.Blist_buff[bfname][key], maven)
      Bfluct = self.Blist_buff[bfname][key] - Bmave
      sig2[key] = np.var(Bfluct) 
      self.w[bfname][key] = np.sqrt( 0.5*0.5 / (sig2[key] + alpha_reg) ) ;  

      wnorm_post = wnorm_post + self.w[bfname][key]
      wnorm_pre  = wnorm_pre +  weight_pre[key]
    
    # regulate length of w after update less than 2.0 times the one before. 
    # too high weights will be reduced by SanityCheck function later
    if (wnorm_post > (1.5 * wnorm_pre) ):
      for key in weight_pre.keys():
        self.w[bfname][key] = self.w[bfname][key] * 1.5 * wnorm_pre / wnorm_post


  def BufferBx(self, Blist):
    for Blist_key in Blist.keys():
      for RLUD in Blist[Blist_key].keys():
          self.Blist_buff[Blist_key][RLUD] = np.roll(self.Blist_buff[Blist_key][RLUD], -1)
          self.Blist_buff[Blist_key][RLUD][-1] = Blist[Blist_key][RLUD]

  # check if adaptively set weights are OK to be applied by judging if each update \Delta*wB exceeds 0.5 or not
  def SanityCheck(self, bfname, thre=1.0):
    # temporary factor value to be updated
    factor = thre

    for RLUD in self.w[bfname].keys():
      temp = self.w[bfname][RLUD] * (self.Blist_buff[bfname][RLUD][-1] - self.Blist_buff[bfname][RLUD][-2])
      if (abs(temp) > thre) :
        print('ESC: Sanity check function detected erronous weight')
        factor = max(temp, factor)

    for RLUD in self.w[bfname].keys():
      self.w[bfname][RLUD] = self.w[bfname][RLUD] * thre / factor
      # print(str(RLUD + ':' + format(self.w[bfname][RLUD], '02.02f') ) )


  # set adaptive weighting for each B values
  def Set_weightedBarrier(self, Blist, iter):
    # define noise magnitude which is used as regularization factor
    self.Bx = 0

    for Blist_key in Blist.keys():
        for RLUD in Blist[Blist_key].keys():
          # temporaliry define all weights to be 1
          self.Bx = self.Bx + self.w[Blist_key][RLUD] * Blist[Blist_key][RLUD]

    self.Bx_list[iter+1] = self.Bx


  ###########################################################################
  # This function defines one step of the ES algorithm at iteration i
  ###########################################################################
  def ES_step(self,p_n,i,cES_now,amplitude):
      # ES step for each parameter
      p_next = np.zeros(self.nES)
      # Loop through each parameter
      for j in np.arange(self.nES):
          p_next[j] = p_n[j] + self.amplitude*self.dtES*np.cos(self.dtES*i*self.wES[j]+self.kES*cES_now+self.Bx)*(self.aES[j]*self.wES[j])**0.5
      
          # For each new ES value, check that we stay within min/max constraints
          if p_next[j] < -1.0:
              p_next[j] = -1.0
          if p_next[j] > 1.0:
              p_next[j] = 1.0
              
      #  phase rotation due to evaluation functions
      self.phase = self.kES*cES_now + self.Bx 
      # Return the next value
      return p_next

  def Plot_normalized_params(self, fig):
    fig, ax11, ax2 = super(ES_Algorithm, self).Plot_normalized_params(fig) # super 
    ax12 = ax11.twinx()
    ln121 = ax12.plot(self.Bx_list, 'C1', label='$B(x)$ ')
    ln122 = ax12.plot(np.multiply(self.kES, self.cES) + self.Bx_list, 'C2', label='$k_{{ES}} C_{{ES}} + B(x)$ ')
 
    l1 = ax11.get_ylim()
    l2 = ax12.get_ylim()
    h_l1 = l1[1] - l1[0]
    h_l2 = l2[1] - l2[0]
    if h_l1 > h_l2 : 
      ax12.set_ylim([l2[0], l2[0]+h_l1])
    elif h_l1 < h_l2 : 
      ax11.set_ylim([l1[0], l1[0]+h_l2])


    ax11.set_ylabel('Phase shift due to $k_{{ES}} C_{{ES}}$ and B(x)')
    h1, l1 = ax11.get_legend_handles_labels()
    h2, l2 = ax12.get_legend_handles_labels()
#    leg = ax11.legend(h1+h2, l1+l2, loc='lower right',frameon=False)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
    plt.tight_layout()




  # dump input values, esc amplitude, phase shift
  def write2df(self, df,tstep):
    # self.esc_input 
    # self.damplitude
    # self.Bx
    # self.phase
    df.loc[tstep,'Phase:k_ES*C+B'] = self.phase
    df.loc[tstep,'Phase:Barrier:B(x)'] = self.Bx
    for idx in range(self.nES):
      df.loc[tstep,str('ESC_input'+ format(idx, '03'))] = self.pES[tstep, idx]
    df.loc[tstep,'ESC_amplitude'] = self.amplitude
    df.loc[tstep,'ESC_decay_rate'] = self.decay_rate

    w_lastkey = list(self.w.keys())[-1]
    for RLUD in self.w[w_lastkey].keys():
      df.loc[tstep,str('weight:' + w_lastkey + RLUD )] = self.w[w_lastkey][RLUD]

    return df




if __name__=='__main__':
  import time

  # number of steps
  ES_steps = 2000
  # Upper bounds on tuned parameters
  p_max = np.array([1e6,20,400,-1e-3])
  # Lower bounds on tuned parameters
  p_min = np.array([-1,-1,-1,-1])
  # kES needs some trial and error
  # kES times B should be around the order of 100
  kES = 1e-8
  # the parameters to have normalized osciallation sizes you 
  # choose the aES as:
  oscillation_size = 0.1


  class ES_Algorithm_User(ES_Algorithm):

    def f_ES_minimize(self,p,i,observation=0):
      f_val = (np.linalg.norm(p, ord=2)-1)**2
      return f_val

 
  es_class = ES_Algorithm_User(ES_steps,p_max,p_min,kES,oscillation_size)

  for iter in np.arange(ES_steps-1):
#      Bx = np.exp(1-iter/ES_steps)
      Bx = 0
      es_class.Set_Barrier(Bx, iter)
      es_input = es_class.ES_main_loop(iter)
      time.sleep(0.1)

  fig1 = plt.figure(2,figsize=(10,15))
  es_class.Plot_normalized_params(fig1)
  fig2 = plt.figure(1,figsize=(10,15))
  es_class.Plot_absolute_params(fig2)
  fig1.savefig('./Image/ESC_NormalizedInput.png')
  fig2.savefig('./Image/ESC_AbsoluteInput.png')


