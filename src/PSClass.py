from UserChannelAccess import UserChannelAccess as CHA
import time


###################################################
## field PS controller inherit parent channel class
###################################################
class CHA_PS_input(CHA):

  def __init__(self, caname, diffmax, T=1000):
    super().__init__(caname,T)
    self.val_ulimit = self.val + abs(diffmax) # upper limit
    self.val_llimit = self.val - abs(diffmax) # lower limit


  def apply_current(self,val):
    if val >= self.val_ulimit: 
      val = self.val_ulimit
    if val <= self.val_llimit: 
      val = self.val_llimit
    self.put_ca(val)
    


###################################################################
class PSClass():  # Store Channel Access class for each power supplier
###################################################################
# Given a magnet name, this class automatically prepare functions for tuning magnets
  def __init__(self, mgname, diffmax, mgtype='none',T=200):
    print ('PSClass ' + mgname + ' initialized')

    self.mgname = mgname

    if mgtype == 'dim':
      self.aiCnv = CHA('psld:' + mgname + ':AiCnv', T, KFSET='ON') # right
      self.dacCnv = CHA('psld:' + mgname + ':DacCnv', T) # right # reading out DAC value
  #    self.error = CHA('psld:' + mgname + ':error.VAL') # right
  #    self.zero = CHA('psld:' + mgname + ':zero.VAL') # right
#      self.speedSelect = CHA('psld:' + mgname + ':SpeedSelect.VAL', T) # right
#      self.updown = CHA('psld:' + mgname + ':UpDownButton', T) # right
      self.step = CHA_PS_input('psld:' + mgname + ':Step', diffmax, T) # right

    elif mgtype == 'ndim':
      self.aiCnv = CHA('psld_ndim:' + mgname + ':adc_cur', T, KFSET='ON') # right
      self.dacCnv = CHA('psld_ndim:' + mgname + ':dac_set', T) # right # reading out DAC value
  #    self.error = CHA('psld:' + mgname + ':error.VAL') # right
  #    self.zero = CHA('psld:' + mgname + ':zero.VAL') # right
#      self.speedSelect = CHA('psld_ndim:' + mgname + ':SpeedSel.VAL', T) # right
#      self.updown = CHA('psld_ndim:' + mgname + ':UpDownButton', T) # right
      self.step = CHA_PS_input('psld_ndim:' + mgname + ':dac_set_0', diffmax, T) # right


    # register step value as pre_step value
    self.renew_step_val()



    self.mgtype = mgtype # either {dim, nio, f3rp, ndim }

  def fetch_ai_input_current(self):
#    self.aiCnv.read_ca()
    return self.aiCnv.fetch()

  def fetch_dac_input_current(self):
#    self.dacCnv.read_ca()
    return self.dacCnv.fetch()

  def show_current(self):
    print( self.mgname + ' PS DacCnv current : ' + str(self.fetch_dac_input_current()))
    print( self.mgname + ' PS AiCnv current  : ' + str(self.fetch_ai_input_current()))


  def apply_current(self,current):
    if self.mgtype == 'dim' or 'ndim': # cim or dim
#      self.dacCnv.apply_current(current)
      self.step.apply_current(current)
    else :
      pass
      print('Apply current in PS Class called, not implemented ' + self.mgname)

  def buffering(self):
      self.aiCnv.buffering()
      self.dacCnv.buffering()
      
      return 0

  def monitor_ca(self):
    self.aiCnv.monitor_ca()
    self.dacCnv.monitor_ca()
    self.step.monitor_ca()
    return 0

  def buffering_pool(self, executor):
      futures = []
      future = executor.submit( self.aiCnv.buffering )
      futures.append( future )
      future = executor.submit( self.dacCnv.buffering )
      futures.append( future )
      return futures

  def renew_step_val(self):
    self.pre_step_val = self.step.val

  def check_Step_reflection(self):
    start = time.time()
    # check if step value is reflected on DAC value
    while True:
      if (self.step.val != self.pre_step_val):
        self.renew_step_val()
        break
      if ((time.time() - start) > 5.0):
        print('Warning : Step reflection is severely slow in ' + self.mgname)
        break
      # add sleep function so that computation cost doesn't go up infinitely
      time.sleep(0.01)


  def check_DAC_reflection(self):
    start = time.time()
    # ensure DAC value reflected
    while True:
      if (self.dacCnv.val == self.step.val):
        break
      if ((time.time() - start) > 5.0):
        print('Warning : DAC reflection is severely slow in ' + self.mgname)
        break
      # add sleep function so that computation cost doesn't go up infinitely
      time.sleep(0.01)





