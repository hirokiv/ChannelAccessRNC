from UserChannelAccess import  UserChannelAccess  as CHA
from PSClass import PSClass as PSClass
from FCClass import FCClass as FCClass
from BaffleClass import BaffleClass as BaffleClass
from concurrent.futures import ThreadPoolExecutor
from user_pickle import pickle_dump
import matplotlib.pyplot as plt
import numpy as np
import time

def printCurrentTime(current_time,begin_time):
    print('Time = ' + str(current_time-begin_time) + ' [sec]')
    return str(current_time-begin_time)

if __name__ == '__main__':
#  mgname = 'A_Q20'
  fcname = 'FCs_A11b'
#  bfname = 'BF_R_MDC1e'
  bfname = 'SL_U10'

  # buffer size
  T = 1000
#  ps_on = CHA('ps:A_Q2P:OnButton')
#  ps_off = CHA('ps:A_Q2P:OffButton')
#  ps_on.put_ca('enabled')
  
  diffmax = 0.0
#  mg1 = PSClass(mgname, diffmax, 'dim', T) # right
  fc0 = FCClass(fcname, T, buff_mode='Average', ave_times=8) # Averaging by 8 times
  fc1 = FCClass(fcname, T, buff_mode='Average', ave_times=4) # Averaging by 8 times
  fc2 = FCClass(fcname, T, buff_mode='Single') # No averaging
  bf1 = BaffleClass(bfname, T)

  # Kalman filter configuration 
  sigv2 = 20 # nA faraday cup sensitivity corresponds to R
  sigw2 = 40 # nA observation noise corresponds to Q
  fc2.initialize_nACur_KF( sigv2, sigw2 )
  fc0.set_SavePath( '../Image_sampleT_test/fc0.pickle' )
  fc1.set_SavePath( '../Image_sampleT_test/fc1.pickle' )
  fc2.set_SavePath( '../Image_sampleT_test/fc2.pickle' )

#     mg1.apply_current(4. + 0.5 * cur)

  ## initialize loop function with certain sampling period
  sampleT = 1.5 # time interval between each loop
  step = 0
  step_end = 2100
  current_time_list = np.array([np.nan]*step_end)


  ##############################################################
  # inline function to be looped. defined by global variables
  ##############################################################
  def loop_func():
    global fc1, bf1, current_time_list

    # initialize executor
    executor = ThreadPoolExecutor(max_workers=8)
    futures = []
    # buffer measured value
#    mg1.buffering()
    for kind in [fc0, fc1,fc2, bf1]:
      future1 = executor.submit( kind.buffering )
      futures.append(future1)
#    mg1.show_current()
    for kind in [fc0, fc1, fc2]:
      future2 = executor.submit( kind.show_current )
      futures.append(future2)
    
    # following plot line can still process while executing the functions above
    # just submit
    if (step % 20 == 0):
      async_future1 = executor.submit(plot_functions, fc1,fc2,bf1)
      async_future2 = executor.submit(dump_functions, fc0,fc1,fc2)

    # Wait only for the buffering functions to finish. 
    # Plot & dumping functions can be still operated
#    for future in futures:
#        future.result()
#        print('%sお待たせしました。' % future.result())
#    executor.shutdown(wait=True)


# sub inline functions

  def plot_functions(fc1,fc2,bf1):
    global sampleT, sigv2, sigw2
    fig, ax = plt.subplots(3,1,figsize=(15,10))
    plt.title("Sampling time = {0}".format(sampleT))
#    mg1.aiCnv.plot_buffer(ax[0])
#    mg1.dacCnv.plot_buffer(ax[0])
    fc1.nACur.plot_buffer(ax[0])
    fc2.nACur.plot_buffer(ax[1])
    bf1.Plot_buffer(ax[2])
    plt.savefig('../Image_sampleT_test/TestFc_SampleT_{0}_sigv2_{1}_sigw2_{2}.png'.format(sampleT,sigv2,sigw2))
    plt.close()

  def dump_functions(fc0,fc1,fc2):
  # dump several observables
    fc0.dump_buffer()
    fc1.dump_buffer()
    fc2.dump_buffer()

  ##############################################################
  # inline function end
  ##############################################################


  # store time at the begining of loop 
  begin_time = time.time() 
  # initialize the curretn time_list
  current_time_list[0] = begin_time 
  # initialize sleep_sec variable
  sleep_sec = 0
  time_elapsed_buff = np.array([np.nan]*T) # this should be compared with the Output buffers of observables

  # start loop process
  while (step<step_end):
    # Show current_time
    printCurrentTime(current_time_list[step], begin_time )
    # add iteration number for the information of the next step
    step = step + 1
    print('Step = ' +  str(step))

    # Launch thread pool executor     

    # main buffering loop to be parallerized
    loop_func()

    # store time at the end of loop, but the time at the beginning
    time_elapsed_buff = time_elapsed_buff[1:]
    time_elapsed_buff = np.append(time_elapsed_buff, ( current_time_list[step-1] + sleep_sec - begin_time ) )
    pickle_dump(time_elapsed_buff, '../Image_sampleT_test/time_buff.pickle')
    # Calculate the elapsed time
    current_time_list[step] = time.time()
    elapsed_sec = current_time_list[step] - begin_time # in second
    sleep_sec = sampleT - (elapsed_sec % sampleT)

    time.sleep(max(sleep_sec, 0))

    
      
