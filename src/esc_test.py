# Delete default warnings when importing CaChannel as they are tadius
import warnings
import datetime
warnings.simplefilter("ignore",UserWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
from sklearn import svm
import sys
import signal
import time
import user_pickle

# User defined packages
from UserChannelAccess import UserChannelAccess as CHA
# from BaffleClass import BaffleClass as BaffleClass
# from PSClass import PSClass as PSClass
from FCClass import FCClass as FCClass
import CBF  # describing control barrier function 
import gauss_fit as gf # Original function. should be in the same folder
from SVM_regression import SVM_regression  # original function. should be in the same folder
from AS_ESC_Class import ES_Algorithm
from ComponentsList import PSesList, BafflesList
 
# Import EPICS channel access interface
from CaChannel import CaChannel, CaChannelException

# Set division warning to error
np.seterr(divide='raise')

 
###################################################
## profile monitor inherit parent channel class
###################################################
class CHA_Profile_Current(CHA):

  def process_data(self):
    middle = int("8000",16)
    self.val_length = len(self.val)
    self.val = np.array(self.val) - middle
    
  # process data of position for profile monitor case
  def process_data_pos(self,stroke,vout,vin,vcenter):
    self.process_data()
    dV = 10 / (int("FFFF",16) - int("8000",16)) # 10 V range within stroke
    val_V = self.val * dV - vcenter # in volt
    self.val = val_V / (vout-vin) * stroke # in mm

  # process position for diagonal analysis. subtract 10mm from the actual position
  def fetch4dpos(self):
    return self.val - 10 # subtract 10mm as diagonal wire ahead of x y wires


def profile_set(string):
  # profiling operation duration is 2 second
  pfl_instate  = CHA('pfl:' + string + ':byte_r1.B00')  # check insertion
  pfl_outstate = CHA('pfl:' + string + ':byte_r1.B04')  # check insertion
  # pfl_outstate = CHA('pfl' + string + 'byte_w1')  # check insertion


def ProfileClass(string,idx):
  # read out functions
  pf_pos = CHA_Profile_Current('pf:' + string + ':DATA') # position
  pf_xpf = CHA_Profile_Current('pf:' + string + ':DATA.XPF') # x 
  pf_ypf = CHA_Profile_Current('pf:' + string + ':DATA.YPF') # y 
  pf_dpf = CHA_Profile_Current('pf:' + string + ':DATA.ZPF') # diagonal
  pf_dir = CHA('pf:' + string + ':ORIENTATION') # insertion direction 1n 3w 5s 7e

  # profile position definition
  pf_vin  = CHA('pf:' + string + ':VLTIN') # insertion voltage initial
  pf_vct  = CHA('pf:' + string + ':VLTCEN') # insertion voltage center
  pf_vout = CHA('pf:' + string + ':VLTOUT') # insertion voltage output
  pf_str  = CHA('pf:' + string + ':STROKE') # STROKE distance in mm
#  pf_str.show()
 
  # Subtract 0x8000
  pf_xpf.process_data() # x 
  pf_ypf.process_data() # y 
  pf_dpf.process_data() # z 
  pf_pos.process_data_pos(pf_str.fetch(),pf_vout.fetch(),pf_vin.fetch(),pf_vct.fetch()) # pos

#  pfA2_xpf = CaChannel('epics_iocmydaq2:aiExample1')
#  pfA2_xpf.search()
#  pfA2_xpf.pend_io()
#  b=pfA2_xpf.getw()
#  print(b)

#  read_ca('pf:PFb_U10a:DATA')
  pos = pf_pos.fetch()
  dpos = pf_pos.fetch4dpos() # position for diagonal wire
  x = pf_xpf.fetch()
  y = pf_ypf.fetch()
  z = pf_dpf.fetch()
#  data_y = pf_pos.read_ca('pf:PFb_A1:DATA')
  xsvm, amp, svmerror_x = SVM_regression(pos,x)
  ysvm, amp, svmerror_y = SVM_regression(pos,y)

  sigma_x,mu_x = gf.main(pos, xsvm, x, string + '_x', amp, idx)
  sigma_y,mu_y = gf.main(pos, ysvm, y, string + '_y', amp, idx)

  l1,l2,l3,l4 = "DATA.XPF","DATA.YPF","DATA.ZPF","SVM DATA.XPF"
  c1,c2,c3,c4 = "blue","green","red","black"

  # plot configuration
  plt.rcParams['font.size'] = 14
  
  # show scales internally plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
 
  fig = plt.figure()
  ax = fig.gca()
  ax.grid()
#  ax.plot(data1, color=c1, label=l1)
#  ax.plot(data2, color=c2, label=l2)
  ax.plot(pos,x, color=c1, label=l1)
  ax.plot(pos,y, color=c2, label=l2)
  ax.plot(dpos,z, color=c3, label=l3)
  ax.plot(pos,xsvm, color=c4, label=l4)
#  ax.plot(a,color = c2)
  ax.set_xlabel('.DATA')
  ax.set_ylabel('y')
  ax.legend(loc=0)
  fig.tight_layout()

  fig.savefig('../Image/esc_test_' + 'IDX' + "{:0>4}".format(idx) + '_' + string + '.png')

  plt.clf()
  plt.close(fig)

  return sigma_x,sigma_y,mu_x,mu_y,svmerror_x,svmerror_y

def main_profile(pf_list):

  idx_list = range(0,len(pf_list))
 
  sigx_list = np.zeros_like(pf_list,dtype=float)
  sigy_list = np.zeros_like(pf_list,dtype=float)
  mux_list = np.zeros_like(pf_list,dtype=float)
  muy_list = np.zeros_like(pf_list,dtype=float)
  svmerror_x = np.zeros_like(pf_list,dtype=bool)
  svmerror_y = np.zeros_like(pf_list,dtype=bool)


  idx = 0
  for pf in pf_list:
#  for idx in idx_list:
    sigx_list[idx],sigy_list[idx],mux_list[idx],muy_list[idx],svmerror_x[idx],svmerror_y[idx] = ProfileClass(pf_list[idx],idx)
    # increment idx by 1
    idx = idx + 1

  df = pd.DataFrame({ 'IDX' : idx,
		     'COMP' : pf_list,
		     'SIGX' : sigx_list,
		     'MUX' : mux_list,
		     'XERR' : svmerror_x,
		     'SIGY' : sigy_list,
		     'MUY' : muy_list,
		     'YERR' : svmerror_y,
                      })

  print(df.head())


  # print centroid and radial evolution
  fig = plt.figure()
  ax1 = fig.add_subplot(2,1,1)
  Plot_RadiusEvolution(ax1,df)
  ax2 = fig.add_subplot(2,1,2)
  Plot_CentroidEvolution(ax2,df)
  fig.tight_layout()
  fig.savefig('../Image/esc_test_' + 'BeamParams' + '.png')

  return df

def Plot_RadiusEvolution(ax,df):
  # Plot pf radius info along course list
  XShow = df[df["XERR"].isin([False])]
  YShow = df[df["YERR"].isin([False])]
    
  l1,l2,l3,l4 = "SIGX","SIGY","",""
  c1,c2,c3,c4 = "blue","green","red","black"
  ax.plot(XShow['IDX'],XShow['SIGX'], color=c1, label=l1, marker="o")
  ax.plot(YShow['IDX'],YShow['SIGY'], color=c2, label=l2, marker="o")
  ax.set_xlabel('INDEX')
  ax.set_ylabel('y')
  ax.legend(loc=0, frameon=False)

def Plot_CentroidEvolution(ax,df):
  # Plot pf centroid info along course list
  XShow = df[df["XERR"].isin([False])]
  YShow = df[df["YERR"].isin([False])]
 
  c1,c2,c3,c4 = "blue","green","red","black"
  l1,l2,l3,l4 = "MUX","MUY","",""
  ax.plot(XShow['IDX'],XShow['MUX'], color=c1, label=l1, marker="o")
  ax.plot(YShow['IDX'],YShow['MUY'], color=c2, label=l2, marker="o")
  ax.set_xlabel('INDEX')
  ax.set_ylabel('y')
  ax.legend(loc=0, frameon=False)
 
 
######################################################
# Define ES cost function and use this class
######################################################
class ES_Algorithm_User(ES_Algorithm):

    def f_ES_minimize(self,p,i,observation):
      # f_val = (np.linalg.norm(p, ord=2)-1)**2
#      f_val = 0
      f_val = (15200.0 - observation )**2
      if (observation < 13500.0): 
        print('Process terminated automatically')
        print('Observation value reached lower limit')
        sys.exit()
      return f_val

######## End User defined ESC class    ###############


###############################################################
## main function
###############################################################
 
if __name__ == '__main__':


###############################################################
## Treat channels as class
###############################################################
  # Generate baffle classs
#   bf_list = ['BF_S_EDC','BF_S_EIC', 'BF_S_MIC2', 'BF_S_MIC2x', 'BF_S_MIC1', 'BF_S_MIC1x', 'BF_S_MDC1', 'BF_S_MDC1x', 'BF_S_MDC2', 'BF_S_MDC2x', 'BF_S_MDC3', 'BF_S_MDC3x', 'BF_S_EBM']
#  bf_list = ['BF_C03', 'BF_F_EBM']

  # specify baffles to be processed
  baffle_course = 'BF_RRC_REDUCED'
#  baffle_course = '28G_TEST'
  # Generate power supply classs
  PS_course = 'PS_RRC_REDUCED'
  # FC course
#  FC_course = '28G_TEST'
  FC_course = 'FCh_A02a'

  # all baffles
  if baffle_course == 'BF_RRC_ALL':

    bf_list = ['BF_R_MIC2e', 'BF_R_MIC1e', 'BF_R_EICe', 'BF_R_MIC1i', 'BF_R_MDC1i', 'BF_R_MDC2i','BF_R_VALi', 'BF_R_MDC1e', 'BF_R_MDC1x', 'BF_R_MDC2e', 'BF_R_MDC2x']
  # reduced model

  elif baffle_course == 'BF_RRC_REDUCED':

    bf_list = ['BF_R_MIC2e', 'BF_R_MIC1e', 'BF_R_MDC1e', 'BF_R_MDC2e']
 
  elif baffle_course == '28G_TEST':
    bf_list = ['SL_U10']


  # allowable current for each baffles in nA
  bf_curr_lim_list = [40.0]*len(bf_list)

  bafflesList = BafflesList(bf_list, bf_curr_lim_list)

  # all PSes
  if PS_course == 'PS_RRC_ALL':
#    ps_list = ['A_Q19'] #, 'A_Q20', 'A_Q21', 'A_Q22']
    ps_list = ['A_ST21', 'A_ST22', 'A_ST23', 'A_ST24', 'A_Q30', 'A_Q31', 'A_Q32']
#    ps_list = ['A_ST24', 'A_Q30', 'A_Q31', 'A_Q32', 'A_ST26', 'BM2', 'BM1_1', 'BM1_2', 'MIC1', 'EICV']
    # MIC2 requires 5 A precision,  seems unpredictable. Eliminate from the list
    # Steerer should be around or less than 0.5 A
    ps_allowable_diff_list = [1.0]*len(ps_list)
    pstype_list = ['dim']*len(ps_list)
 
  # reduced model
  elif PS_course == 'PS_RRC_REDUCED':
    ps_list = [ 'A_Q30', 'A_Q31', 'A_Q32']
#    ps_list = ['A_ST21', 'A_ST22', 'A_ST23', 'A_ST24', 'A_Q30', 'A_Q31', 'A_Q32']
#    ps_list = ['A_ST24', 'A_Q30', 'A_Q31', 'A_Q32', 'A_ST26', 'BM2', 'BM1_1', 'BM1_2', 'MIC2', 'MIC1']
    ps_allowable_diff_list = [ 10.0, 10.0, 10.0]

    pstype_list = ['dim']*len(ps_list)
    pstype_list.extend(['ndim'])
    pstype_list.extend(['dim']*4)
    print(ps_list)
    print(pstype_list)
    print(ps_allowable_diff_list)
  # limit applyable input difference from the initial input 
  # power supplier type {dim, ndim, etc}, 
  # currently only dim type is defined in PSesClass.py

  psesList = PSesList(ps_list, ps_allowable_diff_list, pstype_list)
  esc_input_dict = psesList.return_esc_input_list()



  # Faracay cup course list
  if FC_course == '28G_TEST':
    fc_list = 'FC_U10'

  elif FC_course == 'FCh_A02a':
    fc_list = 'FCh_A02a'

  fc1 =  FCClass(fc_list) 
  # Kalman filter configuration 
  sigv2 = 20 # nA faraday cup sensitivity corresponds to R
  sigw2 = 400 # nA observation noise corresponds to Q
  fc1.initialize_nACur_KF( sigv2, sigw2 )



  # Profile monitors
  pf_list = [ 'PFb_U10a', 'PFb_U10b'] # , 'PFb_B12a', 'PFb_B12b']  #, 'PFb_B22', 'PFb_B31', 'PFb_B41', 'PFb_B50', 'PFb_B61', 'PFb_B71', 'PFb_C21a', 'PFb_S23', 'PFb_S31a', 'PFb_S31b', 'PFb_S41', 'PFb_S61', 'PFb_S64', 'PFb_S71', 'PFb_RRC_BM2', 'PFb_RRC_EBM', 'PFb_A01a', 'PFb_A02a', 'PFb_A02b', 'PFb_A1', 'PFb_A11b', 'PFb_D13', 'PFb_D14', 'PFb_D15', 'PFb_D16a', 'PFb_D16b', 'PFb_D17', 'PFb_D18', 'PFb_D50', 'PFb_5A1', 'PFb_5B1']
#  pf_list = [ 'PFb_S64', 'PFb_S71', 'PFb_RRC_BM2', 'PFb_RRC_EBM', 'PFb_A01a']
#  main_profile(pf_list)

###############################################################
#### Design Barrier function specification
###############################################################
  cbf_mu_init = 30 # barrier function gain C' = kC + muB
  cbf_mu_end = 30 # barrier function gain C' = kC + muB
 # cbf_mu_init = 0
 # cbf_mu_end = 0
  cbf_t = 10.0 # slope of barrier function
  

###############################################################
#### Initialize ESC control class
###############################################################
  ES_steps = 500
  cbf_mu_list = np.linspace(cbf_mu_init, cbf_mu_end, ES_steps)
  cbf_mu0 = cbf_mu_list[0]
  # Upper bounds on tuned parameters
  p_max = np.array(list(esc_input_dict.values())) + ps_allowable_diff_list
  # Lower bounds on tuned parameters
  p_min = np.array(list(esc_input_dict.values())) - ps_allowable_diff_list
  # kES needs some trial and error,  kES times B should be around the order of 100
  kES = 1e-5
  # the parameters to have normalized osciallation sizes you choose the aES as:
  oscillation_size = 0.2
  # decay_rate of input. signal decays as exp( -at) 
  decay_rate = 0.999
  # initiate ES algorithm class instance
  es_class = ES_Algorithm_User(ES_steps,p_max,p_min,kES,oscillation_size,decay_rate,observation=fc1.nACur.fetch_est())
  Bx_bf_est = bafflesList.Calc_CBFs(cbf_mu0, cbf_t)
  es_class.Set_Barrier(Bx_bf_est, -1) 
   

  #######################################################################################
  # Save the initial input state as a function of time so that you can retrieve the original state
  #######################################################################################
  dt_now = datetime.datetime.now()
  print('\n############################################')
  print('Would you like to store the initial input values? ')
  print('############################################\n')
  SaveInputStatus = ''
  while SaveInputStatus not in ['yes', 'no']:
    SaveInputStatus = input("Save input status.\n ''yes'' for saving, ''no'' for skipping this process \n\n " )
 

  init_time_str = dt_now.strftime('%Y-%m-%d:%H:%M:%S') 
  if SaveInputStatus == 'yes':
    print('save called')
    psesList.save_init_ESC_input(dt_now)


  print('\n############################################')
  print('Selected time step is ' + str(ES_steps))
  print('The ESC process doesn''t stop unless stopped by user')
  print('############################################\n')

  StartESCOperation = 'no'
  while StartESCOperation != 'yes':
    StartESCOperation = input("Please recheck the machine state : ''yes'' and begin \n\n " )
 

###############################################################
## Make buffers for storing parameters
###############################################################
 
   # Pandad data frame 
   #  ES_steps = 500
   # time, [mgvalues], [BFs], [FCs], evalf, evalB 
#
#  time_history = np.zeros(ES_steps, 1)
#  mgvalues_history = np.zeros(ES_steps, psesList.num*2)
#  baffle_history   = np.zeros(ES_steps, bafflesList.num*4*2)
#  fc_history       = np.zeros(ES_steps, 2)
#  evalf_history    = np.zeros(ES_steps, 1)
#  evalB_history    = np.zeros(ES_steps, 1)
#  

  

###############################################################
## main loop
###############################################################
  tshow = 20 # show figures every tshow
  
  t_list = range(ES_steps)
  start = time.time() # measure time elapsed

  ## 1.Update baffle values
  ## 2.Calculate and sum all the barrier function values
  for tstep in t_list:
    cbf_mu = cbf_mu_list[tstep]

    ########################################################
    ## Process buffering through Channel Access
    ########################################################
    # ps readouts 
    psesList.buffering()
    # Baffle readouts 
    bafflesList.buffering()
    fc1.buffering()

    ########################################################
    ## Calculate control parameters
    ########################################################
    # Add baffle CBF results to Bx
    Bx_bf_est = bafflesList.Calc_CBFs(cbf_mu, cbf_t)
    # Store Bx value to the ESC algorithm
    es_class.Set_Barrier(Bx_bf_est, tstep) 
    esc_input = es_class.ES_main_loop(tstep, fc1.nACur.fetch_est())

    # convert array into dict 
    esc_input_dict = {key: val for key, val in zip(esc_input_dict.keys(), esc_input)}
 ######### ######### ######### ######### #########
    # apply esc values for each PS
    psesList.apply_currents(esc_input_dict)
      ######### when testing apply function, consider taking ES_step to be 1 #########
 ######### ######### ######### ######### #########


###########################################################
## Store values
###########################################################
#    time_history[tstep,0] = np.zeros(ES_steps, 1)
#    mgvalues_history[tstep,:] = psesList.num)
#    baffle_history[tstep,:]   = bafflesList.num*4)
#    fc_history[tstep,:]       = [fc1.nACur.val, fc1.nACur.estval]
#    evalf_history[tstep,0]    = np.zeros(ES_steps, 1)
#    evalB_history[tstep,0]    = np.zeros(ES_steps, 1)






###########################################################
## show and plot information every tshow steps
###########################################################
      
    if ((tstep % tshow) == 0 and (tstep!=0)) or  (tstep == t_list[-1]):

      print('Step = ' + str( tstep ) )

      # Clear all plots

      ## 1.plot all the value of baffles
      fig1 = plt.figure(1,figsize=(15,15))
      plt.clf()
      ax1 = fig1.add_subplot(2,1,1)
      # call plot buffers method in baffles list class
      bafflesList.Plot_buffers(ax1)
      plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
    
      ## 2.plot time varying Bx value
      ax2 =  fig1.add_subplot(2,1,2)
      label_cbf = 'Control barrier function'
      ax2.plot(t_list,es_class.Bx_list[:-1], label=label_cbf, marker="o")
      plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
      plt.tight_layout()
      fig1.savefig('../Image/esc_test_' + 'Buffer'+init_time_str+'.png')
      plt.close()

      ## 3. Plot normalized ESC input parameters and cost functions
      fig3 = plt.figure(2,figsize=(15,15))
      plt.clf()
      es_class.Plot_normalized_params(fig3)
   #   plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
      plt.tight_layout()
      fig3.savefig('../Image/esc_test_ESC_NormalizedInput'+init_time_str+'.png')
      plt.close()

#      ## 4. Plot absolute ESC input parameters
#      fig4 = plt.figure(3,figsize=(10,15))
#      plt.clf()
#      es_class.Plot_absolute_params(fig4)
#      fig4.savefig('../Image/esc_test_ESC_AbsoluteInput.png')
#
      ## 5. Plot magnetic value configured and readout
      fig5, ax5 = plt.subplots(figsize=(15,15))
      psesList.Plot_PS_buffers(ax5)
#      plt.title("delay = {0}".format(delay))
      plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
      plt.tight_layout()
      fig5.savefig('../Image/esc_test_PSoutput'+init_time_str+'.png')
      plt.close()
    
      ## 6. Plot FC values
      fig6, ax6 = plt.subplots(1,1,figsize=(15,10))
      fc1.nACur.plot_buffer(ax6)
      fig6.savefig('../Image/esc_test_FC'+fc1.nACur.show_caname()+'_'+init_time_str+'.png')



      plt.close()
      
      ## show time elapsed
      elapsed_time = time.time() - start
      print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")



#
#      # Pandad data frame 
#      # time, [mgvalues], [FCs], [BFs], evalf, evalB 
#    
#      df_save = pd.DataFrame([t_list  ],  columns=['time', esc_input_dict])
#      df_save.to_csv(('esc_input_output_dict'+dt_now+'.csv'))
