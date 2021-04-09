from UserChannelAccess import UserChannelAccess as CHA


###################################################
## field PS controller inherit parent channel class
###################################################
class CHA_PS_input(CHA):

  def __init__(self, caname, diffmax, T=100):
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
  def __init__(self, mgname, diffmax, mgtype='none',T=100):
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
      self.aiCnv.buffering_KF()
      self.dacCnv.buffering()

  def buffering(self):
      self.aiCnv.buffering()
      self.dacCnv.buffering()


