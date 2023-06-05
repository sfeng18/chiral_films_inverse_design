#!/usr/bin/env python

tag_wave = 565.0
oth_wave=510.0
target_locat=73
oth_locat=62
# ideal_Gvalue = 1.8
db_abs = 'ABS'
# db_abs = 'PROP'

if db_abs == 'ABS':
    col_max, col_min = 1.806, 0.04
elif db_abs == 'PROP':
    col_max, col_min = 0.671413806451613, 0.07797111363636361
else:
    raise RuntimeError('ERROR: Unkown db_abs value: %s' % db_abs)
thick_max,thick_min=80.0,17.0
str_max,str_min=267.0,20.0
gra_max,gra_min=8.0,1.0

bit_num = 3
bit_str = '111'
# bit_center = [550,650,750]
bit_center = [435,545,700]
bit_width = [0,0,0]

# idealg=2.0