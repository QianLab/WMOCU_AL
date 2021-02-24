#%%
import numpy as np
import multiprocessing

from joblib import Parallel, delayed
import sys
import ProblemSetting as PS

num_cores = 1
runnum = 100
T = 300
methodlist = [0, 1, 2, 3, 5]
xnum = 100
thetanum = 100


inputs = list(range(runnum))
sq = np.random.SeedSequence(555)
rglist = sq.generate_state(runnum)
if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(delayed(PS.SingleIteration)(k, T, rglist, methodlist, xnum, thetanum) for k in inputs)
    
    
#if i == 0:
#    str_label = 'random'
#elif i == 1:
#    str_label = 'MES'
#elif i == 2:
#    str_label = 'BALD'
#elif i == 3:
#    str_label = 'ELR'
#elif i == 5:
#    str_label = 'Weighted_MOCU'