from UserChannelAccess import UserChannelAccess as CHA
from user_pickle import pickle_dump
import datetime

###################################################################
class FCClass():  # Store Channel Access class for each faraday cup
###################################################################
# Given a faradaycup name, this class automatically prepare functions
  def __init__(self, fcname, T=100, delay=0.1, ave_times=8, buff_mode='Average', KFSET = 'OFF'):
    print ('FCClass ' + fcname + ' initialized')

    self.fcname = fcname
    self.buff_mode = buff_mode
    self.KFSET=KFSET
    self.nACur = CHA('fc:' + fcname + ':nACur', T, KFSET=self.KFSET, buff_mode = self.buff_mode) # 
    self.status = CHA('fc:' + fcname + ':status.B04', T) # 
    self.SetOut = CHA('fc:' + fcname + ':StartFcSeq', T) # 
    self.delay = delay
    self.ave_times = ave_times
    self.path = ''
    self.buff_iter = 0
    self.verbose = 0

    if buff_mode == 'Average':
      print('\nBuffering mode set to be averaging by ' + str(ave_times) + ' times\n')


  def fetch_nACur(self):
    if self.KFSET == 'ON':
      return self.nACur.fetch_est()
    else:
      return self.nACur.fetch()

  def show_current(self):
    print( self.fcname + ' nACur : ' + str(self.fetch_nACur()))

  def buffering(self):
      self.buff_iter = self.buff_iter + 1
      if self.buff_mode == 'Average':
        self.nACur.buffering_ave(self.ave_times, self.delay, verbose=self.verbose)

      elif self.buff_mode == 'Single':
        self.nACur.buffering()
      
      if self.verbose == 1:
        print(self.fcname + ' buffering finished')
        print('Iteration number = ' + str(self.buff_iter))

      return 0


  def buffering_pool(self, executor):
      self.buff_iter = self.buff_iter + 1
      futures = []
      if self.buff_mode == 'Average':
        future = executor.submit(  self.nACur.buffering_ave,  self.ave_times, self.delay, verbose=self.verbose )

      elif self.buff_mode == 'Single':
#        self.nACur.buffering()
        future = executor.submit(  self.nACur.buffering() )
      
      if self.verbose == 1:
        print(self.fcname + ' buffering finished')
        print('Iteration number = ' + str(self.buff_iter))

      futures.append(future)
      return futures

  def monitor_ca(self):
    self.nACur.monitor_ca()

  def initialize_nACur_KF(self, sigv2, sigw2):
     # sigv2 variance noise
     # sigw2 observation noise
     self.nACur.initialize_KF(sigv2, sigw2)

  def set_SavePath(self, path):
    self.path = path

  def dump_buffer(self):
    if self.path=='':
      pass
    else:
      pickle_dump(self.nACur.buff, self.path)
    return self.nACur.buff

     
