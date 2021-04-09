from UserChannelAccess import  UserChannelAccess  as CHA
from PSClass import PSClass as PSClass
from FCClass import FCClass as FCClass
from BaffleClass import BaffleClass as BaffleClass
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == '__main__':
  mgname = 'A_Q20'
  fcname = 'FCs_A11b'
#  bfname = 'BF_R_MDC1e'
  bfname = 'SL_U10'

  T = 100
#  ps_on = CHA('ps:A_Q2P:OnButton')
#  ps_off = CHA('ps:A_Q2P:OffButton')
#  ps_on.put_ca('enabled')
  time.sleep( 0.5 ) # sleep for 2 seconds
  
  for delay in [0.0]:
    diffmax = 1.0
    mg1 = PSClass(mgname, diffmax, 'dim', T) # right
    fc1 = FCClass(fcname, T) #
    # Kalman filter configuration 


    for sigv2 in [ 20]: # nA faraday cup sensitivity corresponds to R
#      for sigw2 in [60, 120, 200]: # nA observation noise corresponds to Q
      for sigw2 in [60]: # nA observation noise corresponds to Q
        fc1.initialize_nACur_KF( sigv2, sigw2 )
    
        bf1 = BaffleClass(bfname, T)
    
    #    for cur in np.sin(np.linspace(0,5,T)):
    #      print('Current value : ' + str(cur))
    #      mg1.apply_current(4. + 0.5 * cur)
        for step in range(81):
          
          print('\n')
          print('Step = ' +  str(step))
          print('\n')
          time.sleep( delay )
          mg1.buffering()
          fc1.buffering()
          bf1.buffering()
          mg1.show_current()
          fc1.show_current()
          
      
      
          if (step % 40 == 0):
      
            fig, ax = plt.subplots(3,1,figsize=(15,10))
    #        plt.title("delay = {0}".format(delay))
        #    mg1.aiCnv.plot_buffer(ax[0])
            mg1.aiCnv.plot_buffer(ax[0])
        #    mg1.dacCnv.plot_buffer(ax[0])
            mg1.dacCnv.plot_buffer(ax[0])
            fc1.nACur.plot_buffer(ax[1])
            bf1.Plot_buffer(ax[2])
            plt.savefig('../Image_delay_test/TestDelayMgFc_Tdelay_{0}_sigv2_{1}_sigw2_{2}.png'.format(delay,sigv2,sigw2))

          plt.close()
      
      
      
