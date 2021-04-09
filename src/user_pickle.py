import pickle

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data
# 
# def save_input_state(obj,path):
#   
# 
# 
# def save_input_state(obj,path):
#   user_pickle.save_esc_input_state(esc_input_dict, init_save_path)
#   user_pickle.show_esc_input_state(esc_input_dict, init_save_path)
# 




if __name__=='__main__':
  import glob
  import os
  from ComponentsList import PSesList, BafflesList
  
  list_of_ps_files = glob.glob('/home/fujii/python/data/init_ps*.pickle') # * means all if need specific format then *.csv
  latest_ps_file = max(list_of_ps_files, key=os.path.getctime)
  data_ps = pickle_load(latest_ps_file)

  ps_allowable_diff_list = [10]*len(data_ps.keys())

  list_of_pstype_files = glob.glob('/home/fujii/python/data/ps_type*.pickle') # * means all if need specific format then *.csv
  latest_pstype_file = max(list_of_pstype_files, key=os.path.getctime)
  data_pstype = pickle_load(latest_pstype_file)

  psesList = PSesList(list(data_ps.keys()), ps_allowable_diff_list, data_pstype)

  print('')
  print('Last saved pickle dictionary is ')
  print (latest_ps_file)
  print(data_ps)
  print (latest_pstype_file)
  print(data_pstype)
  apply_flag = input('Do you wanna apply the last value stored? : answer ''yes'' if so \n')

  if apply_flag == 'yes':
    print('initial value applied....')
    psesList.apply_currents(data_ps)
