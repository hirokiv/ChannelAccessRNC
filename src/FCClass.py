from UserChannelAccess import UserChannelAccess as CHA

###################################################################
class FCClass():  # Store Channel Access class for each faraday cup
###################################################################
# Given a faradaycup name, this class automatically prepare functions
  def __init__(self, fcname, T=100):
    print ('FCClass ' + fcname + ' initialized')

    self.fcname = fcname
    self.KFSET='ON'
    self.nACur = CHA('fc:' + fcname + ':nACur', T, KFSET=self.KFSET) # 
    self.status = CHA('fc:' + fcname + ':status.B04', T) # 
    self.SetOut = CHA('fc:' + fcname + ':StartFcSeq', T) # 

    print('\nBuffering mode set to be averaging by 10 times\n')

  def fetch_nACur(self):
    if self.KFSET == 'ON':
      return self.nACur.fetch_est()
    else:
      return self.nACur.fetch()

  def show_current(self):
    print( self.fcname + ' nACur : ' + str(self.fetch_nACur()))

  def buffering(self):
#      self.nACur.buffering()
      self.nACur.buffering_ave(10,0.1,verbose=1)

  def initialize_nACur_KF(self, sigv2, sigw2):
     # sigv2 variance noise
     # sigw2 observation noise
     self.nACur.initialize_KF(sigv2, sigw2)
