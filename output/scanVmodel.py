import vmodel
import os
import numpy as np
import shutil

#out_str = "/home/lars/vmodel_output/vmodel_output/noPred_NoCol/"
args_def = {
'nprey': 100,
'npred': 1,
'frange': 10,
'fstr': 0.0,
'visPred': 300.0,
'visPrey': 359.0,
'astr': 3,
'dphi': 0.2,
'repPrey': 3,
'repRadPrey': 1.5,
'repPred': 1,
'repRadPred': 20,
'attPrey': 3,
'attRadPrey': 1.5,
'repCol': 10000000,
'hstr': 1,
'steps': 4000,
    }



paraChange1_name = "--alignment-strength"
paraChange2_name = "--vision-prey"
steps = 10
reps = 40
#saveLoc = "/home/lars/vmodel_output/test2"
saveLoc = "/extra2/knopf/vmodel_output/hm_millstart_dphi=0.2_occ"

paraChange1_val = np.linspace(0,3,steps)
#paraChange2_val = np.linspace(0,1,steps)
paraChange2_val = np.round(np.linspace(30,359,steps),3)

#paraChange1_val = [0]
#paraChange2_val = [0.05263157894736842]


if (not saveLoc.endswith('/')):
    saveLoc += "/"
    
try:
    os.makedirs(saveLoc)
except FileExistsError:
    # directory already exists
    pass

src_path = "/extra2/knopf/scanVmodel.py"
dst_path = saveLoc + "/scanVmodel.py"
shutil.copy(src_path, dst_path)


for i in paraChange1_val:
    for j in paraChange2_val:
        
        cmd = "vmodel --num-timesteps 20000 --verbose --pred-angle 0 pred--hunt 1 --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 col-style 1 --pred-time 1 --vision-prey 240 --vision-pred 300 --flee-strength 0 --num-runs "+str(reps)+" --outfolder "+str(saveLoc)+" "+str(paraChange1_name)+" "+str(i)+" "+str(paraChange2_name)+" "+str(j)
        print(cmd)
        os.system(cmd)
