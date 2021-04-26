# Delete default warnings when importing CaChannel as they are tadius
import warnings
import datetime
warnings.simplefilter("ignore",UserWarning)
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
#from concurrent.futures import ProcessPoolExecutor
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
# from FCClass import FCClass as FCClass
import CBF  # describing control barrier function 
import gauss_fit as gf # Original function should be in the same folder
from SVM_regression import SVM_regression  # original function. should be in the same folder
from AS_ESC_Class import ES_Algorithm
from ComponentsList import PSesList, BafflesList, FCsList
# Import EPICS channel access interface
from CaChannel import CaChannel, CaChannelException
# Set division warning to error
np.seterr(divide='raise')

 
###################################################
## profile monitor inherit parent channel class
###################################################
###                                             ###
###                                             ###
###                                             ###
###  note All profile monitor related functions  
###  moved to ProfileMonitor.py
###                                             ###
###                                             ###
###                                             ###
###                                             ###
###################################################
 
######################################################
# Define ES cost function and use this class
######################################################
class ES_Algorithm_User(ES_Algorithm):

    def f_ES_minimize(self,p,i,observation):
      # f_val = (np.linalg.norm(p, ord=2)-1)**2
#      f_val = 0
#      f_val = (15200.0 - observation )**2
      f_val = (15000.0 - observation )
#      if (observation < 13500.0): 
#        print('Process terminated automatically')
#        print('Observation value reached lower limit')
#        sys.exit()
      return f_val

######## End User defined ESC class    ###############



######################################################
##  Show current time stamp     
######################################################

def printCurrentTime(current_time,begin_time,tstep):
    print('Step = %06d, Time = %s [sec]'%(tstep, str(current_time-begin_time)))
    return str(current_time-begin_time)

##   End show current time stamp function ##




###################################################
## initialize data frame and associated routines 
###################################################
# Write Component related values
def write_to_df( tstep,  compLists, time_now):
  global df
  
  datetime_now = datetime.datetime.fromtimestamp(time_now)
  # show in milliseconds
  df.loc[tstep,'time'] = str(datetime_now)[:-3]
  df.loc[tstep,'tstep'] = tstep
  for comp in compLists:
    df = comp.write2df(df,tstep)
#   print(df)
  

def initialize_data_frame(ES_steps):
  cols = ['time', 'tstep']
  df = pd.DataFrame(index = range(ES_steps), columns = cols)
  return df


# Write all component values to pickle file and CSV
def df_dump_to_files(df, filename):
  starttime = time.time()

#  output_dir = 'RW_Data'
  csv_wfilename = 'RW_Data/history_csv' + filename + '.csv'
  df.to_csv(csv_wfilename, float_format='%.6e')

  pickle_wfilename = 'RW_Data/history_pickle_'  + filename + '.pickle'
  df.to_pickle(pickle_wfilename)
  
  endtime = time.time()
  elapsed_time = str(endtime - starttime)
  print('\n Successfully dumped outputfile to RW_Data in ' + elapsed_time + '[sec]\n')
  return

## end pandas data frame related functions
###############################################################




###############################################################
## main function
###############################################################
 
if __name__ == '__main__':

###############################################################
## Treat channels as class
###############################################################
  # Generate baffle classs

  # specify baffles to be processed
  baffle_course = 'BF_RRC_REDUCED'

#  baffle_course = '28G_TEST'
  # Generate power supply classs
  PS_course = 'PS_RRC_REDUCED'

  # FC course
#  FC_course = '28G_TEST'
#  FC_course = 'FCh_A02a'
  FC_course = 'FCs_A11b'



  ###################
  ### 1. all baffles
  ###################
  if baffle_course == 'BF_RRC_ALL':
    bf_list = ['BF_R_MIC2e', 'BF_R_MIC1e', 'BF_R_EICe', 'BF_R_MIC1i', 'BF_R_MDC1i', 'BF_R_MDC2i','BF_R_VALi', 'BF_R_MDC1e', 'BF_R_MDC1x', 'BF_R_MDC2e', 'BF_R_MDC2x']
  # reduced model
  elif baffle_course == 'BF_RRC_REDUCED':
    bf_list = ['BF_R_MIC2e', 'BF_R_MIC1e', 'BF_R_MDC1e', 'BF_R_MDC2e']
  elif baffle_course == '28G_TEST':
    bf_list = ['SL_U10']


  ### allowable current for each baffles in nA
  # or determins slope of baffles in linear CBF case
  bf_curr_lim_list = [200.0]*len(bf_list)
  ### generate baffles class instance
  bafflesList = BafflesList(bf_list, bf_curr_lim_list)

  ###################
  ### 2. all PSes
  ###################
  if PS_course == 'PS_RRC_ALL':
#    ps_list = ['A_Q19'] #, 'A_Q20', 'A_Q21', 'A_Q22']
    ps_list = ['A_ST21', 'A_ST22', 'A_ST23', 'A_ST24', 'A_Q30', 'A_Q31', 'A_Q32']
    # MIC2 requires 5 A precision,  seems unpredictable. Eliminate from the list
    # Steerer should be around or less than 0.5 A
    # limit applyable input difference from the initial input 
    ps_allowable_diff_list = [0.02]*len(ps_list)
    # currently only dim & ndim type is defined in PSesClass.py
    pstype_list = ['dim']*len(ps_list)
 
  # reduced model
  elif PS_course == 'PS_RRC_REDUCED':
#    ps_list = [ 'A_Q30', 'A_Q31', 'A_Q32']
    ps_list = ['A_ST21', 'A_ST22', 'A_ST23', 'A_ST24', 'A_Q30', 'A_Q31', 'A_Q32']
#    ps_list = ['A_ST24', 'A_Q30', 'A_Q31', 'A_Q32', 'A_ST26', 'BM2', 'BM1_1', 'BM1_2', 'MIC2', 'MIC1']

  # limit applyable input difference from the initial input 
    ps_allowable_diff_list = [ 0.02]*len(ps_list)

  # power supplier type {dim, ndim, etc}, 
    # currently only dim & ndim type is defined in PSesClass.py
    pstype_list = ['dim']*len(ps_list)
    pstype_list.extend(['ndim'])
    pstype_list.extend(['dim']*4)

  psesList = PSesList(ps_list, ps_allowable_diff_list, pstype_list)
  esc_input_dict = psesList.return_esc_input_list()


  ###################
  # 3. Faracay cup (only 1 signal available)
  ###################
  if FC_course == '28G_TEST':
    fc_list = ['FC_U10']
  elif FC_course == 'FCh_A02a':
    fc_list = ['FCh_A02a']
  elif FC_course == 'FCs_A11b':
    fc_list = ['FCs_A11b']

  fc1 =  FCsList(fc_list, T = 1000, delay=0.1, ave_times = 8, buff_mode = 'Average') # averaging it 8 times
  # Kalman filter configuration 
#  sigv2 = 20 # nA faraday cup sensitivity corresponds to R
#  sigw2 = 40 # nA observation noise corresponds to Q
#  fc1.initialize_nACur_KF( sigv2, sigw2 )




###############################################################
#### Design Barrier function specification
###############################################################
  cbf_mu_init = 1 # barrier function gain C' = kC + muB
#  cbf_mu_end = 1 # barrier function gain C' = kC + muB
  cbf_t = 10.0 # slope of barrier function
  cbf_mu = cbf_mu_init
#  cbf_mu_list = np.linspace(cbf_mu_init, cbf_mu_end, ES_steps)
#  cbf_mu0 = cbf_mu_list[0]
  

###############################################################
#### Initialize ESC control class
###############################################################
  ES_steps = 2000
  # Upper bounds on tuned parameters
  p_max = np.array(list(esc_input_dict.values())) + ps_allowable_diff_list
  # Lower bounds on tuned parameters
  p_min = np.array(list(esc_input_dict.values())) - ps_allowable_diff_list
  # kES needs some trial and error,  kES times B should be around the order of 100
  kES = 1e-5
  # the parameters to have normalized osciallation sizes you choose the aES as:
  oscillation_size = 0.1
  # decay_rate of input. signal decays as exp( -at) 
  decay_rate = 0.999
  # initiate ES algorithm class instance
  es_class = ES_Algorithm_User(ES_steps,p_max,p_min,kES,oscillation_size,decay_rate,observation=fc1.fetch())
  Bx_bf_est = bafflesList.Calc_CBFs(cbf_mu, cbf_t)
  es_class.Set_Barrier(Bx_bf_est, -1) 
   

  #######################################################################################
  # Save the initial input state as a function of time so that you can retrieve the original state
  #######################################################################################
  dt_now = datetime.datetime.now()
  print(dt_now)
  print('\n############################################')
  print('Would you like to store the initial input values? ')
  print('############################################\n')
  SaveInputStatus = ''
  while SaveInputStatus not in ['yes', 'no']:
    SaveInputStatus = input("Save input status.\n ''yes'' for saving, ''no'' for skipping this process \n\n " )
 
  # show in milliseconds
#  init_time_str = str(dt_now)[:-3]
  init_time_str = dt_now.strftime('%Y-%m-%d-%H:%M:%S.%f')[:-3]
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

  ## initialize executor
  # Please limit the number of max_workers to be 4
  executor = ThreadPoolExecutor(max_workers=8)
  

  ######################################################################
  # Inline functions to be processed inside a loop function
  ######################################################################
  def loop_buffering(executor):
    global psesList, bafflesList, fc1
    ## Process buffering through Channel Access
    futures = []
    for kind in [ fc1, psesList, bafflesList]:
      # fc1 should come first as it averages out the signal
      comp_future =  kind.buffering( executor )
      futures += comp_future[:]
    return futures

    # just submit
#    if (step % 20 == 0):
#      async_future1 = executor.submit(plot_functions, fc1,fc2,bf1)
#      async_future2 = executor.submit(dump_functions, fc0,fc1,fc2)
   ###########################################################
   ## show and plot information every tshow steps
   ###########################################################
  def plot_ESC_figures():      
      global t_list, bafflesList, psesList, es_class, fc1

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
      fc1.Plot_FC_buffers(ax6)
      fig6.savefig('../Image/esc_test_FC'+fc1.show_caname()+'_'+init_time_str+'.png')



      plt.close()
 
  ######################################################################


###############################################################
## main loop
###############################################################
  # show figures every tshow
  tshow = 20   
  # execute loop every sampleT [s]
  sampleT = 1.5
  
  t_list = range(ES_steps)

  # store time at the begining of loop 
  begin_time = time.time()
  # initialize the curretn time_list
  current_time_list = np.array([np.nan]*ES_steps)
  current_time_list[0] = begin_time
  # initialize sleep_sec variable
  sleep_sec = 0


  # Prepare dataframe for fileoutput
  df = initialize_data_frame(ES_steps)


  ## 1.Update baffle values
  ## 2.Calculate and sum all the barrier function values
  for tstep in t_list:

    # Show current_time
    printCurrentTime(current_time_list[tstep], begin_time, tstep )
    # add iteration number for the information of the next step

    ################################################################
    # buffering function to be processed in different Threads
    ################################################################
    # execute buffering
    # returns submitted list of pooled jobs
#    print('Line 1 ' +  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
    futures = loop_buffering(executor)


    # following plot line can still process while executing the functions above
    # Plot figures every tshow steps
    if ((tstep % tshow) == 0 and (tstep!=0)) or  (tstep == t_list[-1]):
      executor.submit( plot_ESC_figures )
      executor.submit( df_dump_to_files(df, init_time_str) )


    # Wait for all the buffering process finishes
    for future in futures: # baffles, FC, PS
      future.result()

#    print('Line 2 ' +  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )

    #  Calculate Barrier functions Once the buffering process of baffles ends
    ########################################################
    ## Calculate control parameters
    ########################################################
    # Add baffle CBF results to Bx
    Bx_bf_est = bafflesList.Calc_CBFs(cbf_mu, cbf_t)
    # Store Bx value to the ESC algorithm
    es_class.Set_Barrier(Bx_bf_est, tstep) 
    esc_input = es_class.ES_main_loop(tstep, fc1.fetch())

    # convert array into dict 
    esc_input_dict = {key: val for key, val in zip(esc_input_dict.keys(), esc_input)}
 

    ####################################################################
    # in the following df writer, df is defined as GLOBAL function
    ####################################################################
    executor.submit( write_to_df(tstep,  [fc1, psesList, bafflesList, es_class ], current_time_list[tstep]) )

    ######### ######### ######### ######### #########
    # apply esc values for each PS
    ######### ######### ######### ######### #########
#   executor.submit(  psesList.apply_currents(esc_input_dict) )
    ######### when testing apply function, consider taking ES_step to be 1 #########

    if tstep == ES_steps: 
      break
    # Calculate the elapsed time
    current_time_list[tstep+1] = time.time()
    elapsed_sec = current_time_list[tstep+1] - begin_time # in second
    sleep_sec = sampleT - (elapsed_sec % sampleT)
    if sleep_sec > 0.5*sampleT: 
      sleep_sec = 0
    time.sleep(max(sleep_sec, 0))


  print('Full time steps successfully finished')
