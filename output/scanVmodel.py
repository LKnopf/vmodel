import vmodel
import os
import numpy as np

out_str = "/home/lars/vmodel_output/vmodel_output/npPred/"
args_def = {
'nprey': 100,
'npred': 1,
'frange': 10,
'fstr': 0.0,
'visPred': 300.0,
'visPrey': 330,
'astr': 3,
'dphi': 0.2,
'repPrey': 1,
'repRadPrey': 1.5,
'repPred': 1,
'repRadPred': 20,
'attPrey': 3,
'attRadPrey': 1.5,
'repCol': 10000000,
'hstr': 1,
'steps': 12000,
    }



paraChange1_name = "--alignment-strength"
paraChange2_name = "--dphi"
steps = 20
reps = 30
saveLoc = "/home/lars/vmodel_output/test2"
saveLoc = "/extra2/knopf/vmodel_output/noPred_noMultCol"

paraChange1_val = np.linspace(0,1.5,steps)
paraChange2_val = np.linspace(0,1,steps)



if (not saveLoc.endswith('/')):
    saveLoc += "/"
    
try:
    os.makedirs(saveLoc)
except FileExistsError:
    # directory already exists
    pass



for i in paraChange1_val:
    for j in paraChange2_val:
        
        cmd = "vmodel --num-timesteps 1200 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 2 --vision-pred 300 --flee-strength 0 --num-runs "+str(reps)+" --outfolder "+str(saveLoc)+" "+str(paraChange1_name)+" "+str(i)+" "+str(paraChange2_name)+" "+str(j)
        print(cmd)
        os.system(cmd)
