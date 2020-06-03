import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.system('rm global.txt')
os.system('for i in {1..20}; do /usr/bin/time -f "\t%E" -ao timing.txt python test_fvgp.py 2> junk | grep "New hyper-parameters" | tail -c 16 >> global.txt; done;')

timing = pd.read_csv('global.txt')
timing = np.array([x.strip().strip(']').strip('[') for x in [y[0] for y in timing.values]]).astype(float)


plt.hist(timing, label='global')
plt.show()

