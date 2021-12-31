
import os

this_dir = os.path.dirname(__file__)

# list all the corresponding setup files
setups = ['cysetup.py', ]

for setup in setups:
    setup = os.path.join(this_dir, setup)
    cmd = f'python {setup} build_ext --inplace'
    os.system(cmd)

print('\ndone')
