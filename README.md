# ChannelAccessRNC

compatible with python environment. 

For AVF-RRC application, the main python file to be executed is
```
fujii@filesrv-centos7.rarfadv.riken.go.jp:$PATHTOGIT/src/ $ python esc_test.py
``` 

You are required to specify the quadrupoles in question for the variable `ps_list`, and specify the allowable input range in `ps_allowable_diff_list`.
Baffles to be included as observables should be specified in `bf_list`, and Faraday cup current for the variable `fc_list`. 

For tyring out Extremum Seeking Control, start with small initial gain $k$ of $p_i = a_i \cos(\omega_i t + k_{ES} C)$
by increasing `kES = 1e-9`. The input's oscillation size $a_i$ is determined by `oscillation_size` variable relative to teh minimum and maximum allowable input values.

Experimental images will be generated under `../Image/`, and numerical observations under `RW_Data/`. These directory should be manually generated before running the experiments.  

Before running the main routine, you are encouranged to save the current input status by following the prompt. 
```
  print('\n############################################')
  print('Would you like to store the initial input values? ')
  print('############################################\n')
  SaveInputStatus = ''
  while SaveInputStatus not in ['yes', 'no']:
    SaveInputStatus = input("Save input status.\n ''yes'' for saving, ''no'' for skipping this process \n\n " )
```
The initial input states wil be stored under `../data/` and can be restored by running `user_pickele.py`. 


# Editor information

This documentation and extensions are prepared by H. Fujii, Beam Dynamics and diagnostics team led by N. Fukunisi at Riken Nishina Center. 
Also affiliated with Dynamics & Control Laboratory led by M. Yamakita at Tokyo Tech.

Email: fujii $at$ ac.ctrl.titech.ac.jp / hirokifujii9 $at$ gmail.com

The algorithm employed through the close consultation with A. Scheinker. 

Experimental results reported in 
H Fujii, A. Scheinker, A. Uchiyama, S.J. Gessner, O. Kamigaito and N. Fukunishi, EXTREMUM SEEKING CONTROL FOR THE OPTIMIZATION OF HEAVY ION BEAM TRANSPORTATION, Proceedings of the 18th Annual Meeting of Particle Accelerator Society of Japan, August 9 - 12, 2021, QST-Takasaki Online, Japan.
