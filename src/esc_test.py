# Delete default warnings when importing CaChannel as they are tadius
import warnings
import datetime
warnings.simplefilter("ignore",UserWarning)
import matplotlib
matplotlib.use('TkAgg')  # Or any other X11 back-end
import matplotlib.pyplot as plt
#matplotlib.use('GTK3Agg')  # Or any other X11 back-end
# 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo',
#matplotlib.use('TkCairo')

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
#      f_val = (20000.0 - observation )
      f_val = 0
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
  csv_wfilename = 'RW_Data/history_csv_' + filename + '.csv'
  df.to_csv(csv_wfilename, float_format='%.6e')

  pickle_wfilename = 'RW_Data/history_pickle_'  + filename + '.pickle'
  df.to_pickle(pickle_wfilename)
  
  endtime = time.time()
  elapsed_time = str(endtime - starttime)
  print('\n Successfully dumped outputfile to RW_Data in ' + elapsed_time + '[sec]\n')
  return

## end pandas data frame related functions
###############################################################



def save_script_myself(filename):
  # importing the modules
  import os
  import shutil
  
  # getting the current working directory
  src_dir = os.getcwd()
  src_dir = src_dir + '/RW_Data'
  
  # printing current directory
  print(src_dir) 
  
  # copying the files
  shutil.copyfile('esc_test.py', 'RW_Data/esc_test_history_' + filename + '.py') #copy src to dst
  
  # printing the list of new files
  # print(os.listdir(src_dir)) 


###############################################################
## main function
###############################################################
 
if __name__ == '__main__':

###############################################################
## Treat channels as class
###############################################################
  # Generate baffle classs

  COURSE = 'AVF_Shootout'
#  COURSE = 'RRC'
  #COURSE = 'AVF_Shootout_Test'

  if COURSE=='RRC' : 
    baffle_course = 'BF_RRC_REDUCED'
  #  baffle_course = '28G_TEST'
  #  PS_course = 'PS_RRC_MORE_REDUCED'
    PS_course = 'PS_RRC_REDUCED'
  #  FC_course = '28G_TEST'
    FC_course = 'FCh_A02a'
  #  FC_course = 'FCs_A11b'

  elif COURSE=='AVF_Shootout':
    baffle_course = 'BF_AVF_Shoot'
    #baffle_course = 'BF_AVF_SL_TEST'
    PS_course =     'PS_AVF_Shoot'
    FC_course = 'FCh_A02b'

  elif COURSE=='AVF_Shootout_Test':
    baffle_course = 'BF_AVF_Shoot_Test'
    #baffle_course = 'BF_AVF_SL_TEST'
    PS_course =     'PS_AVF_Shoot_Test'
    FC_course = 'FC_AVF_Shoot_Test'


  sampleT = 0.1

  ###################
  ### 1. all baffles
  ###################
  if baffle_course == 'BF_RRC_ALL':
    bf_list = ['BF_R_MIC2e', 'BF_R_MIC1e', 'BF_R_EICe', 'BF_R_MIC1i', 'BF_R_MDC1i', 'BF_R_MDC2i','BF_R_VALi', 'BF_R_MDC1e', 'BF_R_MDC1x', 'BF_R_MDC2e', 'BF_R_MDC2x']
  # reduced model
    Bx_buffsize = 200
    eval_bfname = bf_list[-1]
  elif baffle_course == 'BF_RRC_REDUCED':
    bf_list = ['BF_R_MIC2e', 'BF_R_MIC1e', 'BF_R_MDC1e', 'BF_R_MDC2e']
    ### allowable current for each baffles in nA
    # or determins slope of baffles in linear CBF case
    bf_curr_lim_list = [5.0]*len(bf_list)
    ### generate baffles class instance
    CBFTYPE = ['Linear']*len(bf_list)
    bafflesList = BafflesList(bf_list, bf_curr_lim_list, CBFTYPE)
    Bx_buffsize = 200
    eval_bfname = bf_list[-1]
  elif baffle_course == '28G_TEST':
    bf_list = ['SL_U10']
  elif baffle_course == 'BF_AVF_Shoot':
    # bf_list = ['SL_C01a','BF_C03']
    # bf_curr_lim_list = [0.0, 100.0]
    # ### generate baffles class instance
    # CBFTYPE = ['Linear', 'AbsLinear']
    bf_list = ['BF_C03']
    bf_curr_lim_list = [ 100.0]
    ### generate baffles class instance
    CBFTYPE = ['AbsLinear']
    bafflesList = BafflesList(bf_list, bf_curr_lim_list, CBFTYPE)
    eval_bfname = bf_list[-1]
  elif baffle_course == 'BF_AVF_Shoot_Test':
    bf_list = ['SL_C01a']
    bf_curr_lim_list = [300.0]
    ### generate baffles class instance
    CBFTYPE = ['AbsLinear']
    bafflesList = BafflesList(bf_list, bf_curr_lim_list, CBFTYPE)
    eval_bfname = bf_list[-1]



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
    ps_list = ['A_ST21', 'A_ST22', 'A_ST23', 'A_ST24', 'A_Q30', 'A_Q31', 'A_Q32']
    #    ps_list = ['A_ST24', 'A_Q30', 'A_Q31', 'A_Q32', 'A_ST26', 'BM2', 'BM1_1', 'BM1_2', 'MIC2', 'MIC1']

    # limit applyable input difference from the initial input 
#    ps_allowable_diff_list = [ 2.0]*4
    ps_allowable_diff_list = [ 0.01]*4
#    ps_allowable_diff_list.extend( [ 30.0]*3 )
    ps_allowable_diff_list.extend( [ 0.001]*3 )
    # power supplier type {dim, ndim, etc}, 
    # currently only dim & ndim type is defined in PSesClass.py
    pstype_list = ['dim']*len(ps_list)
    pstype_list.extend(['ndim'])
    pstype_list.extend(['dim']*4)

  # more reduced model
  elif PS_course == 'PS_RRC_MORE_REDUCED':
    ps_list = [ 'A_Q30', 'A_Q31', 'A_Q32']

    # limit applyable input difference from the initial input 
    #    ps_allowable_diff_list = [ 0.01]*4
    ps_allowable_diff_list = [ 30.0]*len(ps_list)
    # power supplier type {dim, ndim, etc}, 
    # currently only dim & ndim type is defined in PSesClass.py
    pstype_list = ['dim']*len(ps_list)
    #    pstype_list.extend(['ndim'])
    #    pstype_list.extend(['dim']*4)

  # AVF shooting course
  elif PS_course == 'PS_AVF_Shoot':
   # power supplier type {dim, ndim, etc}, 
#    ps_list = ['AVF_S1','AVF_Q1','AVF_Q2','AVF_Q3','AVF_S3','AVF_S4','AVF_Q4','AVF_Q5']
#    # limit applyable input difference from the initial input 
#    ps_allowable_diff_list = [0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0]
    ##### Omit S1 as it doesn't respond well
    # operation before 3:21
    ps_list = [ 'AVF_Q1','AVF_Q2','AVF_Q3','AVF_S3','AVF_S4','AVF_Q4','AVF_Q5']
#    # limit applyable input difference from the initial input 
#    ps_allowable_diff_list = [3.0, 3.0, 3.0, 0.5, 0.5, 3.0, 3.0]
    #ps_allowable_diff_list = [3.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0] # .    set 1
    ps_allowable_diff_list = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]# . set 2
#    ps_allowable_diff_list = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

#    ps_list = [ 'AVF_Q3','AVF_S3','AVF_S4','AVF_Q4','AVF_Q5']
#    ps_allowable_diff_list = [1.0, 2.0, 1.0, 1.0, 5.0]
    pstype_list = ['dim']*len(ps_list)
    Bx_buffsize = 200

  elif PS_course == 'PS_AVF_Shoot_Test':
   # power supplier type {dim, ndim, etc}, 
#    ps_list = ['AVF_S1','AVF_Q1','AVF_Q2','AVF_Q3','AVF_S3','AVF_S4','AVF_Q4','AVF_Q5']
#    # limit applyable input difference from the initial input 
#    ps_allowable_diff_list = [0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0]
    ##### Omit S1 as it doesn't respond well
    ps_list = ['AVF_S1', 'AVF_Q1','AVF_Q2','AVF_Q3']
    # limit applyable input difference from the initial input 
    ps_allowable_diff_list = [0.5, 5.0, 5.0, 5.0]
    pstype_list = ['dim']*len(ps_list)
    Bx_buffsize = 100


  psesList = PSesList(ps_list, ps_allowable_diff_list, pstype_list)
  esc_input_dict = psesList.return_esc_input_list()
  esc_input_dict0 = esc_input_dict


  ###################
  # 3. Faracay cup (only 1 signal available)
  ###################
  if FC_course == '28G_TEST':
    fc_list = ['FC_U10']
  elif FC_course == 'FCh_A02a':
    fc_list = ['FCh_A02a']
  elif FC_course == 'FCh_A02b':
    fc_list = ['FCh_A02b']
  elif FC_course == 'FCs_A11b':
    fc_list = ['FCs_A11b']
  elif FC_course == 'FC_AVF_Shoot':
    fc_list = ['FC_C03']
  elif FC_course == 'FC_AVF_Shoot_Test':
    fc_list = ['FC_C01a']
  elif FC_course == 'NONE':
    fc_list = []



  fc1 =  FCsList(fc_list, T = 1000, delay=0.1, ave_times = 2, buff_mode = 'Average') # averaging it 4 times
  # Kalman filter configuration 
  #  sigv2 = 20 # nA faraday cup sensitivity corresponds to R
  #  sigw2 = 40 # nA observation noise corresponds to Q
  #  fc1.initialize_nACur_KF( sigv2, sigw2 )




  ###############################################################
  #### Design Barrier function specification
  ###############################################################
  cbf_mu_init = 1 # barrier function gain C' = kC + muB
  #  cbf_mu_end = 1 # barrier function gain C' = kC + muB
  cbf_t = 100.0 # slope of barrier function
  #  cbf_t = 10.0 # slope of barrier function
  cbf_mu = cbf_mu_init
  #  cbf_mu_list = np.linspace(cbf_mu_init, cbf_mu_end, ES_steps)
  #  cbf_mu0 = cbf_mu_list[0]
  

###############################################################
#### Initialize ESC control class
###############################################################
  ES_steps = 4000
  # Upper bounds on tuned parameters
  p_max = np.array(list(esc_input_dict.values())) + ps_allowable_diff_list
  # Lower bounds on tuned parameters
  p_min = np.array(list(esc_input_dict.values())) - ps_allowable_diff_list
  # kES needs some trial and error,  kES times B should be around the order of 100
  kES = 1e-9
  # the parameters to have normalized osciallation sizes you choose the aES as:
  oscillation_size = 0.02
  # decay_rate of input. signal decays as exp( -at) 
  decay_rate = 1.0 # 0.999
  # initiate ES algorithm class instance
  es_class = ES_Algorithm_User(ES_steps,p_max,p_min,kES,oscillation_size,decay_rate,observation=fc1.fetch())
  bafflesList.Calc_CBFs(cbf_mu, cbf_t)
  # BF_result contains 1. simple sum of all Bs, 2. each B value in key 
  es_class.Set_Barrier(bafflesList.Fetch_Bx(), -1) 
  # define weight
  es_class.DefineBarrierWeights( bafflesList.Fetch_Blist() )
  # define Bxlist_buff
  es_class.DefineBxlistBuffer( bafflesList.Fetch_Blist(), Bx_buffsize )
  # Blist = BF_result[1]
  # es_class.Set_weightedBarrier(Blist, -1) 
   

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
 
  ######################################################################
  # make a copy of this file for the later reference
  ######################################################################
  save_script_myself(init_time_str)


  ######################################################################
  ## initialize executor
  ######################################################################
  # Please limit the number of max_workers to be 16
  executor = ThreadPoolExecutor(max_workers=4)
  

  ######################################################################
  # Inline functions to be processed inside a loop function
  ######################################################################
  def loop_buffering(executor):
    global psesList, bafflesList, fc1
    ## Process buffering through Channel Access
    futures = []
    for kind in [ fc1, psesList, bafflesList]:
      # fc1 should come first as it averages out the signal
      comp_future =  kind.buffering_pool( executor )
      futures += comp_future[:]
    return futures

  def monitor_CAs():
    global psesList, bafflesList, fc1
    for kind in [ fc1, psesList, bafflesList]:
      # fc1 should come first as it averages out the signal
      kind.monitor_ca()
    psesList.comp_cha_list[psesList.comp_list[0]].aiCnv.flush_io()
 
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
      plt.close()

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
#      plt.close()

      ## 3. Plot normalized ESC input parameters and cost functions
      fig3 = plt.figure(2,figsize=(15,15))
      plt.clf()
      es_class.Plot_normalized_params(fig3)
   #   plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
      plt.tight_layout()
      fig3.savefig('../Image/esc_test_ESC_NormalizedInput'+init_time_str+'.png')
#      plt.close()

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
#      plt.close()
    
      ## 6. Plot FC values
      fig6, ax6 = plt.subplots(1,1,figsize=(15,10))
      fc1.Plot_FC_buffers(ax6)
      fig6.savefig('../Image/esc_test_FC'+fc1.show_caname()+'_'+init_time_str+'.png')

      plt.show()


 
  ######################################################################


###############################################################
## main loop
###############################################################
  # show figures every tshow
  tshow = 20   
  # execute loop every sampleT [s]
  
  t_list = range(ES_steps)

  # store time at the begining of loop 
  begin_time = time.time()
  # initialize the curretn time_list
  current_time_list = np.array([np.nan]*ES_steps)
  current_time_list[0] = begin_time
  # initialize sleep_sec variable
  sleep_sec = 0

  ###############################################################
  ## Make buffer dataframe for storing parameters
  ###############################################################
  # Prepare dataframe for fileoutput
  df = initialize_data_frame(ES_steps)
  # activate CA monitors
  monitor_CAs()
  time.sleep(1.0)


  ## 1.Update baffle values
  ## 2.Calculate and sum all the barrier function values
  for tstep in t_list:

    # Show current_time
    printCurrentTime(current_time_list[tstep], begin_time, tstep )
    # add iteration number for the information of the next step

    ################################################################
    # buffering function replaced by CA monitor 
    ################################################################
    # execute buffering
    # returns submitted list of pooled jobs
    #    print('Line 1 ' +  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
    # futures = loop_buffering(executor)


    # following plot line can still process while executing the functions above
    # Plot figures every tshow steps
    if ((tstep % tshow) == 0 and (tstep!=0)) or  (tstep == t_list[-1]):
      executor.submit( plot_ESC_figures )
      executor.submit( df_dump_to_files(df, init_time_str) )


#    # Wait for all the buffering process finishes
#    for future in futures: # baffles, FC, PS
#      future.result()

#    print('Line 2 ' +  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )

    #  Calculate Barrier functions Once the buffering process of baffles ends
    ########################################################
    ## Calculate control parameters
    ########################################################
    # Add baffle CBF results to Bx
    bafflesList.Calc_CBFs(cbf_mu, cbf_t)
    # Store Bx value to the ESC algorithm
    # es_class.Set_Barrier(BF_result[0], tstep) 
    es_class.BufferBx( bafflesList.Fetch_Blist() )
    if (tstep >= Bx_buffsize):
      # activate adaptive weighting for baffles
      if (tstep % Bx_buffsize == 0):
        es_class.UpdateWeights(tstep, eval_bfname)
      # regulate weight value to have small enough value s.t. ESC stay stable
      es_class.SanityCheck(eval_bfname)
      es_class.Set_weightedBarrier(bafflesList.Fetch_Blist(), tstep) 
    else:
      # simple sum of all the barrier function
      es_class.SanityCheck(eval_bfname)
      es_class.Set_Barrier(bafflesList.Fetch_Bx(), tstep) 

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
    #    #psesList.apply_currents_pool(esc_input_dict, executor) 
    # psesList.apply_currents(esc_input_dict) 
    # see if step value is changed and DAC value reflected
    psesList.check_reflection()

#    psesList.apply_currents_sequential(esc_input_dict0, executor, tstep) 
#    for future in futures: # baffles, FC, PS
#      future.result()

    ######### when testing apply function, consider taking ES_step to be 1 #########

    if tstep == (ES_steps-1): 
      break
    # Calculate the elapsed time
    current_time_list[tstep+1] = time.time()
    elapsed_sec = current_time_list[tstep+1] - begin_time # in second
    sleep_sec = sampleT - (elapsed_sec % sampleT)
    #    if sleep_sec > 0.5*sampleT: 
    #      sleep_sec = 0
    time.sleep(max(sleep_sec, 0))


  print('Full time steps successfully finished')
