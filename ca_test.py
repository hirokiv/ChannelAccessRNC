# Delete default warnings when importing CaChannel as they are tadius
import warnings
warnings.simplefilter("ignore",UserWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
import gauss_fit as gf # Original function. should be in the same folder
from sklearn import svm
 
# Import EPICS channel access interface
from CaChannel import CaChannel, CaChannelException

# Set division warning to error
np.seterr(divide='raise')

#####################
# begin CaAccess class
#####################
class CHA: # CHannel Access
  caname = ''
  chan = []
  val = []
  rawval = []
  val_length = []

  def __init__(self, caname):
    self.caname = caname
    self.chan = CaChannel(caname)
    #  chan = CaChannel('akito12Host:xxxExample')
    print("CA init:  " + str(self.caname))
    self.val = self.read_ca()
    self.rawval = self.val

  def __del__(self):
	# body of destructor
    return

  def read_ca(self):
  
    try:
      if isinstance(self.caname,str):
        self.chan.searchw()
      #  chan.putw(3.14)
        #chan.pend_io()
        a=self.chan.getw()
        return a 
      else:
        print('Channel definition false : specify string') 
    except CaChannelException as e:
       print(e)

  def fetch(self):
    return self.val

  def fetch4dpos(self):
    return self.val - 10 # subtract 10mm as diagonal wire ahead of x y wires

  def show(self):
    print(self.val)

  def process_data(self):
    middle = int("8000",16)
    self.val_length = len(self.val)
    self.val = np.array(self.val) - middle

  def process_data_dpf(self):
    self.process_data()
    
  # process data of position
  def process_data_pos(self,stroke,vout,vin,vcenter):
    self.process_data()
    dV = 10 / (int("FFFF",16) - int("8000",16)) # 10 V range within stroke
    val_V = self.val * dV - vcenter # in volt
    self.val = val_V / (vout-vin) * stroke # in mm

#####################
# End CaAccess class
#####################

def profile_set(string):
  # profiling operation duration is 2 second
  pfl_instate  = CHA('pfl:' + string + ':byte_r1.B00')  # check insertion
  pfl_outstate = CHA('pfl:' + string + ':byte_r1.B04')  # check insertion
  # pfl_outstate = CHA('pfl' + string + 'byte_w1')  # check insertion


def SVM_regression(x,y):

  try:
   
    amp = max(y) - min(y)
   
    y = y / amp
  #  # データを用意する------------------------------------------------------------------
  #  x = np.arange(0.0, 10.0, 0.1)                           # 横軸を作成
  #  noise = np.random.normal(loc=0, scale=0.1, size=len(x)) # ガウシアンノイズを生成
  #  y = np.sin(2 * np.pi * 0.2 * x) + noise                 # 学習用サンプル波形
  #  # ----------------------------------------------------------------------------------
    
    #  learn by svm
    model = svm.SVR(C=0.1, kernel='rbf', epsilon=0.01, gamma='auto')    # RBF kernel
    model.fit(x.reshape(-1, 1), y)                          # fitting
    
    # predict with supervised parameter set
    x_reg = x                         # generate x axis for regression
    y_reg = model.predict(x_reg.reshape(-1, 1))             # predicted
    r2 = model.score(x.reshape(-1, 1), y)                   # determine coefficient
    
    # Plot configuration---------------------------------------------------------------
    plt.rcParams['font.size'] = 14
    
    # show scales internally
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    # add measures in each side of the graph
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # データプロットの準備。
    ax1.scatter(x, y, label='Dataset', lw=1, marker="o")
    ax1.plot(x_reg, y_reg, label='Regression curve', color='red')
    plt.legend(loc='upper right')
    
    # グラフ内に決定係数を記入
    plt.text(0.0, -0.5, '$\ R^{2}=$' + str(round(r2, 2)), fontsize=20)
    fig.tight_layout()
    plt.show()
    fig.savefig('Image/' + 'regression_result.png')
  
    plt.clf()
    plt.close(fig)
  
    error_flag = 0 # no error
    return y_reg * amp, amp, error_flag
  
#  except FloatingPointError | ZeroDivisionError as e:
  except FloatingPointError  as error_flag:
    print('SVM_regression false' ) 
    return np.zeros_like(y), 0, error_flag
    print(e)

 


def profile_readout(string,idx):

  # read out functions
  pf_pos = CHA('pf:' + string + ':DATA') # position
  pf_xpf = CHA('pf:' + string + ':DATA.XPF') # x 
  pf_ypf = CHA('pf:' + string + ':DATA.YPF') # y 
  pf_dpf = CHA('pf:' + string + ':DATA.ZPF') # diagonal
  pf_dir = CHA('pf:' + string + ':ORIENTATION') # insertion direction 1n 3w 5s 7e

  # profile position definition
  pf_vin  = CHA('pf:' + string + ':VLTIN') # insertion voltage initial
  pf_vct  = CHA('pf:' + string + ':VLTCEN') # insertion voltage center
  pf_vout = CHA('pf:' + string + ':VLTOUT') # insertion voltage output
  pf_str  = CHA('pf:' + string + ':STROKE') # STROKE distance in mm
#  pf_str.show()
 
  # Subtract 0x8000
  pf_xpf.process_data() # x 
  pf_ypf.process_data() # y 
  pf_dpf.process_data_dpf() # z 
  pf_pos.process_data_pos(pf_str.fetch(),pf_vout.fetch(),pf_vin.fetch(),pf_vct.fetch()) # pos

#  pfA2_xpf = CaChannel('epics_iocmydaq2:aiExample1')
#  pfA2_xpf.search()
#  pfA2_xpf.pend_io()
#  b=pfA2_xpf.getw()
#  print(b)

#  read_ca('pf:PFb_U10a:DATA')
  pos = pf_pos.fetch()
  dpos = pf_pos.fetch4dpos()
  x = pf_xpf.fetch()
  y = pf_ypf.fetch()
  z = pf_dpf.fetch()
#  data_y = pf_pos.read_ca('pf:PFb_A1:DATA')
  xsvm, amp, svmerror_x = SVM_regression(pos,x)
  ysvm, amp, svmerror_y = SVM_regression(pos,y)

  sigma_x,mu_x = gf.main(pos, xsvm, x, string + '_x', amp, idx)
  sigma_y,mu_y = gf.main(pos, ysvm, y, string + '_y', amp, idx)

  l1,l2,l3,l4 = "DATA.XPF","DATA.YPF","DATA.ZPF","SVM DATA.XPF"
  c1,c2,c3,c4 = "blue","green","red","black"

  # plot configuration
  plt.rcParams['font.size'] = 14
  
  # show scales internally plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
 
  fig = plt.figure()
  ax = fig.gca()
  ax.grid()
#  ax.plot(data1, color=c1, label=l1)
#  ax.plot(data2, color=c2, label=l2)
  ax.plot(pos,x, color=c1, label=l1)
  ax.plot(pos,y, color=c2, label=l2)
  ax.plot(dpos,z, color=c3, label=l3)
  ax.plot(pos,xsvm, color=c4, label=l4)
#  ax.plot(a,color = c2)
  ax.set_xlabel('.DATA')
  ax.set_ylabel('y')
  ax.legend(loc=0)
  fig.tight_layout()

  fig.savefig('Image/' + 'IDX' + "{:0>4}".format(idx) + '_' + string + '.png')

  plt.clf()
  plt.close(fig)

  return sigma_x,sigma_y,mu_x,mu_y,svmerror_x,svmerror_y

def main_profile():

  course_list = [ 'PFb_U10a', 'PFb_U10b', 'PFb_B12a', 'PFb_B12b', 'PFb_B22', 'PFb_B31', 'PFb_B41', 'PFb_B50', 'PFb_B61', 'PFb_B71', 'PFb_C21a', 'PFb_S23', 'PFb_S31a', 'PFb_S31b', 'PFb_S41', 'PFb_S61', 'PFb_S64', 'PFb_S71', 'PFb_RRC_BM2', 'PFb_RRC_EBM', 'PFb_A01a', 'PFb_A02a', 'PFb_A02b', 'PFb_A1', 'PFb_A11b', 'PFb_D13', 'PFb_D14', 'PFb_D15', 'PFb_D16a', 'PFb_D16b', 'PFb_D17', 'PFb_D18', 'PFb_D50', 'PFb_5A1', 'PFb_5B1']
#  course_list = [ 'PFb_S64', 'PFb_S71', 'PFb_RRC_BM2', 'PFb_RRC_EBM', 'PFb_A01a']
  idx_list = range(0,len(course_list))
 
  sigx_list = np.zeros_like(course_list,dtype=float)
  sigy_list = np.zeros_like(course_list,dtype=float)
  mux_list = np.zeros_like(course_list,dtype=float)
  muy_list = np.zeros_like(course_list,dtype=float)
  svmerror_x = np.zeros_like(course_list,dtype=bool)
  svmerror_y = np.zeros_like(course_list,dtype=bool)

  for idx in idx_list:
    sigx_list[idx],sigy_list[idx],mux_list[idx],muy_list[idx],svmerror_x[idx],svmerror_y[idx] = profile_readout(course_list[idx],idx)

  df = pd.DataFrame({ 'IDX' : idx_list,
		     'COMP' : course_list,
		     'SIGX' : sigx_list,
		     'MUX' : mux_list,
		     'XERR' : svmerror_x,
		     'SIGY' : sigy_list,
		     'MUY' : muy_list,
		     'YERR' : svmerror_y,
                      })

  print(df.head())

  XShow = df[df["XERR"].isin([False])]
  YShow = df[df["YERR"].isin([False])]
  print(XShow.head())

  # Plot course info
  c1,c2,c3,c4 = "blue","green","red","black"
  fig = plt.figure()
  ax1 = fig.add_subplot(2,1,1)
#  ax.plot(data1, color=c1, label=l1)
#  ax.plot(data2, color=c2, label=l2)
  l1,l2,l3,l4 = "SIGX","SIGY","",""
  ax1.plot(XShow['IDX'],XShow['SIGX'], color=c1, label=l1, marker="o")
  ax1.plot(YShow['IDX'],YShow['SIGY'], color=c2, label=l2, marker="o")
#  ax.plot(a,color = c2)
  ax1.set_xlabel('INDEX')
  ax1.set_ylabel('y')
  ax1.legend(loc=0)

  l1,l2,l3,l4 = "MUX","MUY","",""
  ax2 = fig.add_subplot(2,1,2)
  ax2.plot(XShow['IDX'],XShow['MUX'], color=c1, label=l1, marker="o")
  ax2.plot(YShow['IDX'],YShow['MUY'], color=c2, label=l2, marker="o")
  ax2.set_xlabel('INDEX')
  ax2.set_ylabel('y')
  ax2.legend(loc=0)

  fig.tight_layout()

  fig.savefig('Image/' + 'BeamParams' + '.png')

  return df


def main_buffles():


if __name__ == '__main__':
  main_profile()
  main_Buffles()
  
