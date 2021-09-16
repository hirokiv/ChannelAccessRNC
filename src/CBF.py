# Implement Control barrier function in sigmoid
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
matplotlib.use('ps')

def CBF_Sigmoid(x,b=20,mu=100,t=1):
  B = mu / (1 + np.exp((b-x)/t))
  return B

def CBF_LinearWithDeadZone(x,lim,slope):
  if x<lim : 
    B = 0
  else:
    B = (x-lim) / slope 
  return B

def CBF_AbsLinear(x,lim,slope):
  if x< 0.1 :
    B = (lim-x) / slope  + 0.1
  elif x<lim : 
    B = (lim-x) / slope 
  else:
    B = (x-lim) / slope 
  return B


if __name__=='__main__':
  x = np.linspace(-10.,200.,100) 
  fig = plt.figure()
  for b in [20,40,80]:
    for t in [5,10,20]:
      B = CBF_Sigmoid( x, b, 100, t)
      plt.plot(x,B,label="t-b"+str(t)+'-'+str(b))
#  plt.legend({'t-b:5-20','t-b:10-20','t-b:20-20','t-b:5-40','t-b:10-40','t-b:20-40','t-b:5-80','t-b:10-80','t-b:20-80'})
  plt.legend()
  fig.savefig('../Image/CBF.png')


