from UserChannelAccess import UserChannelAccess as CHA
import CBF
import time
import random
from concurrent.futures import ThreadPoolExecutor

###################################################################
class BaffleClass():  # Store Channel access class for each baffle components
###################################################################

  def __init__(self, bfname, bf_curr_lim, CBFTYPE='Linear'):
    print ('BafleClass ' + bfname + ' initialized')

    self.bfname = bfname
    self.bf = {}
#    self.CBFTYPE = 'Linear'
    self.CBFTYPE = CBFTYPE

    bfr = CHA('sl:' + bfname + 'R:nACur', T=100, timeout=1.0, KFSET='ON') # right
    if bfr.error_flag == 0:
      self.bf['R'] = bfr

    bfl = CHA('sl:' + bfname + 'L:nACur', T=100, timeout=1.0, KFSET='ON') # left
    if bfl.error_flag == 0:
      self.bf['L'] = bfl

    bfu = CHA('sl:' + bfname + 'U:nACur', T=100, timeout=0.5, KFSET='ON') # up
    if bfu.error_flag == 0:
      self.bf['U'] = bfu

    bfd = CHA('sl:' + bfname + 'D:nACur', T=100, timeout=1.0, KFSET='ON') # down
    if bfd.error_flag == 0:
      self.bf['D'] = bfd

    self.num = len(self.bf.keys())
    self.bf_curr_lim  = bf_curr_lim

    #    Prepare buffers for each component of baffles

  def show_bfname(self):
    print(self.bfname)
    return self.bfname

  def buffering_pool(self, executor):
    try:
      futures = []
      for i in self.bf.keys():
        # Avoid multiple components access to IOC at the same time
        future = executor.submit ( self.bf[i].buffering )
        futures.append(future)
#        time.sleep(0.023)
#      self.bf['R'].chan.poll()
      return futures 
    except CaChannelException as e:
      print(e)
      return e

  def buffering(self):
    executor = ThreadPoolExecutor()
    try:
      for i in self.bf.keys():
        # Avoid multiple components access to IOC at the same time
        self.bf[i].buffering()
#        time.sleep(0.023)
#      self.bf['R'].chan.poll()
      return  0
    except CaChannelException as e:
      print(e)
      return e

  def monitor_ca(self):
    for i in self.bf.keys():
      self.bf[i].monitor_ca()
    return 0
    

  def Calc_CBF(self,cbf_mu,cbf_t):
  # calculate barrier function and sum for all(RLUD) components
    Bx_est = 0
    comp_CBF = {}
    for key in self.bf.keys():
#      print('BaffleClassCalc_CBF called' )
      # if sigmoid function preferrred 
      if self.CBFTYPE == 'Sigmoid':
        comp_CBF[key] = CBF.CBF_Sigmoid(self.bf[key].fetch_raw(), self.bf_curr_lim, cbf_mu, cbf_t)
      elif self.CBFTYPE == 'Linear':
        # cbf_t describes the slope of CBF
        # curr_lim describes where CBF initiates
        comp_CBF[key] = CBF.CBF_LinearWithDeadZone(self.bf[key].fetch_raw(), self.bf_curr_lim, cbf_t)
      elif self.CBFTYPE == 'AbsLinear':
#        print('AbsLinear CBF called')
        comp_CBF[key] = CBF.CBF_AbsLinear(self.bf[key].fetch_raw(), self.bf_curr_lim, cbf_t)

      Bx_est = Bx_est +  comp_CBF[key]    
      
#      print('comp_CBF = {}' .format(comp_CBF))
    
    return Bx_est, comp_CBF


  def Plot_buffer(self,ax):
    # axis where to plot
#    print('Plot Buffer called in baffle name ' + self.bfname)
    for key in self.bf.keys():
      ax1 = self.bf[key].plot_buffer(ax)  # )),bf_cha_list[i].bf[key].buff, bf_cha_list[i].bfname  + key, fig)


  # recursive least square being implemented
  def buffer_rls(self):
    pass

