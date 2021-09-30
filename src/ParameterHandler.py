# change parameter while script is runnning by accessing external file
import numpy as np
import os
import time

line_num = 1


class ParameterHandler:
  def __init__(self, filename, parameter, prange):
    self.filename = filename + '.txt'
    self.orig_parameter = parameter
    self.lines = ''
    self.write_parameter(parameter)
    self.prange = prange # allowable range in ratio
    self.prefile = self.read_parameter()


  def read_parameter(self):
    with open(self.filename, "r") as txt_file:
      return txt_file.readlines()
  
  def write_parameter(self, param):
    with open("amplitude.txt", "w") as txt_file:
      lines = [self.filename[:-4]+'\n', str(param)]
      txt_file.writelines(lines)
      self.prefile = lines

  def renew_parameter(self, parameter):
    # detect if any changes made to the original file?
    postfile = self.read_parameter()
    if self.prefile != postfile: 
      # change detected
      temp = float(postfile[line_num])
      if ( not self.check_parameter_range(temp) ):
        # range check passed
        parameter = temp
        
    self.write_parameter(parameter)
    return parameter

  def check_parameter_range(self, param):
    # check if parameter is within allowable range
    flag_lower = (self.orig_parameter * self.prange[0]) > param
    flag_upper = (self.orig_parameter * self.prange[1]) < param
    flag = flag_upper | flag_lower
    if flag:
      # value out of range, stop writing
      print('Parameter out of range')
      print('Retrieve previous value')

    return flag


if __name__ == '__main__':
  amp_ratio_range = [0.02, 1] # specify param in ratio
  amplitude = 1
  ph = ParameterHandler('amplitude', amplitude, amp_ratio_range)
  decay_rate = 0.999
  while True: 
    time.sleep(0.01)
    amplitude = amplitude * decay_rate
    amplitude = ph.renew_parameter(amplitude)

    # print('Current amplitude')
    print(amplitude)

