from BaffleClass import BaffleClass as BaffleClass
from PSClass import PSClass as PSClass
from FCClass import FCClass as FCClass
import user_pickle
import decimal 
import datetime
import time

decimal.getcontext().prec = 6
decimal.getcontext().traps[decimal.FloatOperation] = True


###########################################################
# Store component classes and define methods in this class
###########################################################
class ComponentsList:
  def __init__(self, comp_list):
    # store component names to initialize channels
    self.comp_list = comp_list
    # channel class ensamble
    self.comp_cha_list = {}
    # list of index
    self.idx_list = range(0,len(comp_list))
    self.num = len(comp_list) 
    self.buff_iter = 0
    self.verbose = 0

  def return_cha_list(self):
    return self.comp_cha_list

#   def buffering(self):
#   # utilize thread from ThreadPoolExecutor
#     self.buff_iter = self.buff_iter + 1
# #    futures = []
#     for comp in self.comp_list:
#       # Baffle readouts and calculation of Barrierfunctions
#       self.comp_cha_list[comp].buffering()
# 
#     if self.verbose == 1:
#       print(self.comp_list[0] + ' buffering finished')
#       print('Iteration number = ' + str(self.buff_iter))
#     
#     return 0

  def buffering(self, executor):
  # utilize thread from ThreadPoolExecutor
    self.buff_iter = self.buff_iter + 1
    futures = []
    for comp in self.comp_list:
      # Baffle readouts and calculation of Barrierfunctions
      future =  self.comp_cha_list[comp].buffering_pool( executor )
      futures.extend(future)

    if self.verbose == 1:
      print(self.comp_list[0] + ' buffering finished')
      print('Iteration number = ' + str(self.buff_iter))
    
    return futures
 

###########################################################
# Baffles list
###########################################################
class BafflesList(ComponentsList):
# Generate baffle classes for control output
  def __init__(self, bf_list, bf_curr_lim_list):

    # inherit ComponentList
    super().__init__(bf_list,)
    # Store limit values
    self.comp_lim_list = bf_curr_lim_list

    # check number of components v.s. number of limits
    if len(bf_list) != len(bf_curr_lim_list):
      print('Error in BafflesList : number of curreent limit and baffles different' )
      sys.exit()

    # initialize baffle classes
    for idx in self.idx_list:
      self.comp_cha_list[bf_list[idx]] =  BaffleClass(bf_list[idx],bf_curr_lim_list[idx])

  # visualize baffle values
  def Plot_buffers(self,ax):
      for comp in self.comp_list:
        self.comp_cha_list[comp].Plot_buffer(ax)

  # calculate CBFs and sum for all buffles
  def Calc_CBFs(self,cbf_mu,cbf_t):
    Bx = 0  # [observed, estimated]
    for comp in self.comp_list:
      Bx = Bx + self.comp_cha_list[comp].Calc_CBF(cbf_mu, cbf_t)
    return Bx
   
  # create time list data
  def write2df(self,df,row):
    for comp in self.comp_list:
      for i in self.comp_cha_list[comp].bf.keys():
        df.loc[row, self.comp_cha_list[comp].bf[i].caname] =   self.comp_cha_list[comp].bf[i].fetch()
    return df
 
#####################################################################
## PS
#####################################################################
class PSesList(ComponentsList):
  def __init__(self, ps_list, ps_allowable_diff_list, pstype_list):
    # inherit ComponentList
    super().__init__(ps_list,)

   # Generate power supply classes for control input
    self.init_esc_input_list = {} # store power supply classes
    self.pstype_list = pstype_list

    for idx in self.idx_list:
      self.comp_cha_list[ps_list[idx]] =  PSClass(ps_list[idx], ps_allowable_diff_list[idx], pstype_list[idx])
      self.init_esc_input_list[ps_list[idx]] = self.comp_cha_list[ps_list[idx]].fetch_dac_input_current()


  def save_init_ESC_input(self, dt_now):
    init_save_path = '../data/init_ps_' + dt_now.strftime('%Y-%m-%d:%H:%M:%S') + '.pickle'
    user_pickle.pickle_dump(self.init_esc_input_list, init_save_path)
    print( user_pickle.pickle_load(init_save_path) )
    init_save_path = '../data/ps_type_' + dt_now.strftime('%Y-%m-%d:%H:%M:%S') + '.pickle'

    user_pickle.pickle_dump(self.pstype_list, init_save_path)
    print( user_pickle.pickle_load(init_save_path) )
    print(' Saved initial ESC input in ../data/ folder')

  def return_esc_input_list(self):
    return self.init_esc_input_list

  def Plot_PS_buffers(self, ax):
    for comp in self.comp_list:
      self.comp_cha_list[comp].aiCnv.plot_buffer(ax)
      self.comp_cha_list[comp].dacCnv.plot_buffer(ax)

  def apply_currents(self,esc_input_dict):
    for comp in self.comp_list:
      self.comp_cha_list[comp].apply_current(esc_input_dict[comp])
   
  # create time list data
  def write2df(self,df,row):
    for comp in self.comp_list:
      df.loc[row, self.comp_cha_list[comp].aiCnv.caname]  =   self.comp_cha_list[comp].aiCnv.fetch() 
      df.loc[row, self.comp_cha_list[comp].dacCnv.caname] =   self.comp_cha_list[comp].dacCnv.fetch()
    return df
 

#####################################################################
## FC 
#####################################################################
class FCsList(ComponentsList):
  def __init__(self, fc_list, T, delay = 0.1,  ave_times = 8, buff_mode = 'Average'):
    # inherit ComponentList
    super().__init__(fc_list)


    for idx in self.idx_list:
      self.comp_cha_list[fc_list[idx]] =   FCClass(fc_list[idx], T, delay, ave_times, buff_mode) # averaging it 8 times 

  def Plot_FC_buffers(self, ax):
    for comp in self.comp_list:
      return self.comp_cha_list[comp].nACur.plot_buffer(ax)

  def fetch(self):
    for comp in self.comp_list:
      return self.comp_cha_list[comp].nACur.fetch()

  def show_caname(self):
    for comp in self.comp_list:
      return self.comp_cha_list[comp].nACur.show_caname()

  # create time list data
  def write2df(self,df,row):
    for comp in self.comp_list:
      df.loc[row, self.comp_cha_list[comp].nACur.caname] = self.comp_cha_list[comp].nACur.fetch()
    return df
 
