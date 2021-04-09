from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


def fit_plot(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        sigma = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/sigma)**2) + params[-1]
        y_list.append(y)
    return y_list

def func(x, *params):

    # judge number of functions to be fitted
    num_func = int(len(params)/3)

    # add each gaussian functions and add to y_list
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        sigma = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/sigma)**2)
        y_list.append(y)

    # superpose all the gaussians within y_list
    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum = y_sum + i

    # Add background value (offset)
    y_sum = y_sum + params[-1]

    return y_sum

def fit_plot(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        sigma = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/sigma)**2) + params[-1]
        y_list.append(y)
    return y_list


def main(x,y,yraw,filename,amp,idx):
  # [amp, ctr, sigma]
  guess = []
  guess.append([amp, 0, 20])
 # guess.append([7500, 775, 10])
  
  # initial background value
  background = 1
  
  # append initial values
  guess_total = []
  for i in guess:
      guess_total.extend(i)
  guess_total.append(background)

  popt, pcov = curve_fit(func, x, y, p0=guess_total)

  fig = plt.figure()
  ax1 = fig.add_subplot(111)

  fit = func(x, *popt)
  l1, l2 = "SVM data", "Raw data"
  ax1.scatter(x, y, s=20, label=l1)
  ax1.scatter(x, yraw, s=10, label=l2)
  ax1.plot(x, fit , ls='-', c='black', lw=1)
  

  y_list = fit_plot(x, *popt)
  baseline = np.zeros_like(x) + popt[-1]
  for n,i in enumerate(y_list):
     ax1.fill_between(x, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)
  ax1.text(0.0, 0.5, r'$\sigma =$' + str(round(popt[2], 2)) + r'   $\mu =$' + str(round(popt[1], 2)), fontsize=20)
  ax1.legend(loc='upper left')

  fig.savefig('../Image_fit/IDX' + "{:0>4}".format(idx) + '_gauss_fit_' + filename + '.png')

  return abs(popt[2]), popt[1]  # sigma,mu

if __name__ == '__main__':
  main()

