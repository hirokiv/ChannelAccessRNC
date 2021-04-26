

# Profile monitors
  pf_list = [ 'PFb_U10a', 'PFb_U10b'] # , 'PFb_B12a', 'PFb_B12b']  #, 'PFb_B22', 'PFb_B31', 'PFb_B41', 'PFb_B50', 'PFb_B61', 'PFb_B71', 'PFb_C21a', 'PFb_S23', 'PFb_S31a', 'PFb_S31b', 'PFb_S41', 'PFb_S61', 'PFb_S64', 'PFb_S71', 'PFb_RRC_BM2', 'PFb_RRC_EBM', 'PFb_A01a', 'PFb_A02a', 'PFb_A02b', 'PFb_A1', 'PFb_A11b', 'PFb_D13', 'PFb_D14', 'PFb_D15', 'PFb_D16a', 'PFb_D16b', 'PFb_D17', 'PFb_D18', 'PFb_D50', 'PFb_5A1', 'PFb_5B1']
#  pf_list = [ 'PFb_S64', 'PFb_S71', 'PFb_RRC_BM2', 'PFb_RRC_EBM', 'PFb_A01a']
#  main_profile(pf_list)


################################################### 
## profile monitor inherit parent channel class 
################################################### 
class CHA_Profile_Current(CHA): 
 
  def process_data(self): 
    middle = int("8000",16) 
    self.val_length = len(self.val) 
    self.val = np.array(self.val) - middle 
     
  # process data of position for profile monitor case 
  def process_data_pos(self,stroke,vout,vin,vcenter): 
    self.process_data() 
    dV = 10 / (int("FFFF",16) - int("8000",16)) # 10 V range within stroke 
    val_V = self.val * dV - vcenter # in volt 
    self.val = val_V / (vout-vin) * stroke # in mm 
 
  # process position for diagonal analysis. subtract 10mm from the actual position 
  def fetch4dpos(self): 
    return self.val - 10 # subtract 10mm as diagonal wire ahead of x y wires 
 
 
def profile_set(string): 
  # profiling operation duration is 2 second 
  pfl_instate  = CHA('pfl:' + string + ':byte_r1.B00')  # check insertion 
  pfl_outstate = CHA('pfl:' + string + ':byte_r1.B04')  # check insertion 
  # pfl_outstate = CHA('pfl' + string + 'byte_w1')  # check insertion 
 
 
def ProfileClass(string,idx): 
  # read out functions 
  pf_pos = CHA_Profile_Current('pf:' + string + ':DATA') # position 
  pf_xpf = CHA_Profile_Current('pf:' + string + ':DATA.XPF') # x  
  pf_ypf = CHA_Profile_Current('pf:' + string + ':DATA.YPF') # y  
  pf_dpf = CHA_Profile_Current('pf:' + string + ':DATA.ZPF') # diagonal 
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
  pf_dpf.process_data() # z  
  pf_pos.process_data_pos(pf_str.fetch(),pf_vout.fetch(),pf_vin.fetch(),pf_vct.fetch()) # pos 
 
#  pfA2_xpf = CaChannel('epics_iocmydaq2:aiExample1') 
#  pfA2_xpf.search() 
#  pfA2_xpf.pend_io() 
#  b=pfA2_xpf.getw() 
#  print(b) 
 
#  read_ca('pf:PFb_U10a:DATA') 
  pos = pf_pos.fetch() 
  dpos = pf_pos.fetch4dpos() # position for diagonal wire 
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
 
  fig.savefig('../Image/esc_test_' + 'IDX' + "{:0>4}".format(idx) + '_' + string + '.png') 
 
  plt.clf() 
  plt.close(fig) 
 
  return sigma_x,sigma_y,mu_x,mu_y,svmerror_x,svmerror_y 
 
def main_profile(pf_list): 
 
  idx_list = range(0,len(pf_list)) 
  
  sigx_list = np.zeros_like(pf_list,dtype=float) 
  sigy_list = np.zeros_like(pf_list,dtype=float) 
  mux_list = np.zeros_like(pf_list,dtype=float) 
  muy_list = np.zeros_like(pf_list,dtype=float) 
  svmerror_x = np.zeros_like(pf_list,dtype=bool) 
  svmerror_y = np.zeros_like(pf_list,dtype=bool) 
 
 
  idx = 0 
  for pf in pf_list: 
#  for idx in idx_list: 
    sigx_list[idx],sigy_list[idx],mux_list[idx],muy_list[idx],svmerror_x[idx],svmerror_y[idx] = ProfileClass(pf_list[idx],idx) 
    # increment idx by 1 
    idx = idx + 1 
 
  df = pd.DataFrame({ 'IDX' : idx, 
                     'COMP' : pf_list, 
                     'SIGX' : sigx_list, 
                     'MUX' : mux_list, 
                     'XERR' : svmerror_x, 
                     'SIGY' : sigy_list, 
                     'MUY' : muy_list, 
                     'YERR' : svmerror_y, 
                      }) 
 
  print(df.head()) 
 
 
  # print centroid and radial evolution 
  fig = plt.figure() 
  ax1 = fig.add_subplot(2,1,1) 
  Plot_RadiusEvolution(ax1,df) 
  ax2 = fig.add_subplot(2,1,2) 
  Plot_CentroidEvolution(ax2,df) 
  fig.tight_layout() 
  fig.savefig('../Image/esc_test_' + 'BeamParams' + '.png') 
 
  return df 
 
def Plot_RadiusEvolution(ax,df): 
  # Plot pf radius info along course list 
  XShow = df[df["XERR"].isin([False])] 
  YShow = df[df["YERR"].isin([False])] 
     
  l1,l2,l3,l4 = "SIGX","SIGY","","" 
  c1,c2,c3,c4 = "blue","green","red","black" 
  ax.plot(XShow['IDX'],XShow['SIGX'], color=c1, label=l1, marker="o") 
  ax.plot(YShow['IDX'],YShow['SIGY'], color=c2, label=l2, marker="o") 
  ax.set_xlabel('INDEX') 
  ax.set_ylabel('y') 
  ax.legend(loc=0, frameon=False) 
 
def Plot_CentroidEvolution(ax,df): 
  # Plot pf centroid info along course list 
  XShow = df[df["XERR"].isin([False])] 
  YShow = df[df["YERR"].isin([False])] 
  
  c1,c2,c3,c4 = "blue","green","red","black" 
  l1,l2,l3,l4 = "MUX","MUY","","" 
  ax.plot(XShow['IDX'],XShow['MUX'], color=c1, label=l1, marker="o") 
  ax.plot(YShow['IDX'],YShow['MUY'], color=c2, label=l2, marker="o") 
  ax.set_xlabel('INDEX') 
  ax.set_ylabel('y') 
  ax.legend(loc=0, frameon=False) 


