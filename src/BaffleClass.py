from UserChannelAccess import UserChannelAccess as CHA
from CBF import CBF_Sigmoid

###################################################################
class BaffleClass():  # Store Channel access class for each baffle components
###################################################################

  def __init__(self, bfname, bf_curr_lim):
    print ('BafleClass ' + bfname + ' initialized')

    self.bfname = bfname
    self.bf = {}

    bfr = CHA('sl:' + bfname + 'R:nACur', timeout=0.2, KFSET='ON') # right
    if bfr.error_flag == 0:
      self.bf['R'] = bfr

    bfl = CHA('sl:' + bfname + 'L:nACur', timeout=0.2, KFSET='ON') # left
    if bfl.error_flag == 0:
      self.bf['L'] = bfl

    bfu = CHA('sl:' + bfname + 'U:nACur', timeout=0.2, KFSET='ON') # up
    if bfu.error_flag == 0:
      self.bf['U'] = bfu

    bfd = CHA('sl:' + bfname + 'D:nACur', timeout=0.2, KFSET='ON') # down
    if bfd.error_flag == 0:
      self.bf['D'] = bfd

    self.num = len(self.bf.keys())
    self.bf_curr_lim  = bf_curr_lim

    #    Prepare buffers for each component of baffles

  def show_bfname(self):
    print(self.bfname)
    return self.bfname

  def buffering(self):
    for i in self.bf.keys():
      self.bf[i].buffering()

  def Calc_CBF(self,cbf_mu,cbf_t):
  # calculate barrier function and sum for all(RLUD) components
    Bx_est = 0
    for key in self.bf.keys():
#      print('BaffleClassCalc_CBF')
#      print(self.bf[key].fetch_est())
#      print(CBF_Sigmoid(self.bf[key].fetch_est(), self.bf_curr_lim, cbf_mu, cbf_t))
      Bx_est = Bx_est + CBF_Sigmoid(self.bf[key].fetch_est(), self.bf_curr_lim, cbf_mu, cbf_t)
     # Bx = Bx + CBF_Sigmoid(self.bf[key].fetch(), self.bf_curr_lim, cbf_mu, cbf_t)
    return Bx_est


  def Plot_buffer(self,ax):
    # axis where to plot
    print('Plot Buffer called in baffle name ' + self.bfname)
    for key in self.bf.keys():
      ax1 = self.bf[key].plot_buffer(ax)  # )),bf_cha_list[i].bf[key].buff, bf_cha_list[i].bfname  + key, fig)


  # recursive least square being implemented
  def buffer_rls(self):
    pass

