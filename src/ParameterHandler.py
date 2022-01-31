# change parameter while script is runnning by accessing external file
import numpy as np
import os
import time

line_num = 1
base_path = './Writable/'


class ParameterHandler:
  def __init__(self, filename, parameter, prange):
    self.filename = filename + '.txt'
    self.orig_parameter = parameter
    self.lines = ''
    self.write_parameter(parameter)
    self.prange = prange # allowable range in ratio
    self.prefile = self.read_parameter()


  def read_parameter(self):
    try:
      with open(base_path+self.filename, "r") as txt_file:
        temp = txt_file.readlines()
        print(temp)
        return temp 
    except:
      return self.prefile
  
  def write_parameter(self, param):
    with open(base_path+self.filename, "w") as txt_file:
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
    flag_lower = (self.prange[0]) > param
    flag_upper = (self.prange[1]) < param
    flag = flag_upper | flag_lower
    if flag:
      # value out of range, stop writing
      print('Parameter out of range')
      print('Retrieve previous value')

    return flag


if __name__ == '__main__':
  amp_range = [0.02, 1.0] # specify param in abs
  amplitude = 1
  ph_amp = ParameterHandler('amplitude', amplitude, amp_range)
  drate_range = [0.9,1.0]
  decay_rate = 0.999
  ph_decay = ParameterHandler('decay_rate', decay_rate, drate_range)

  while True: 
    time.sleep(0.1)
    amplitude = amplitude * decay_rate
    amplitude = ph_amp.renew_parameter(amplitude)
    decay_rate = ph_decay.renew_parameter(decay_rate)

    # print('Current amplitude')
    print(amplitude)
    print(decay_rate)

