import vmodel
import os
import numpy as np
import h5py
import datetime
import scipy.spatial

def calc_order(vel: np.ndarray) -> float:
    """Compute order parameter from velocity matrix
    Args:
        vel: velocity matrix (N x D)
    Returns:
        order: velocity correlation
    """
    N, _ = vel.shape
    speed = np.linalg.norm(vel, axis=1, keepdims=True)  # N x 1
    speed_prod = speed.dot(speed.T)  # N x N
    mask = (speed_prod != 0)  # avoid division by zero!
    dot_prod = vel.dot(vel.T)  # N x N
    np.fill_diagonal(dot_prod, 0)  # i != j
    return (dot_prod[mask] / speed_prod[mask]).sum() / (N * (N - 1))

def calc_IID(pos: np.ndarray):
    """Compute IID"""
    N, _ = pos.shape
    dist = 0
    for i in range(N):
        for j in range(N):
            dist += np.linalg.norm(pos[i]-pos[j])
    
    return dist / (N * (N - 1))

out_str = "/extra2/knopf/vmodel_output/noPred/"
saveLoc = "/extra2/knopf/vmodel_output/"
saveName = "noPred_"


args = {
'nprey': 100,
'npred': 1,
'frange': 10,
'fstr': 0.0,
'visPred': 300.0,
'visPrey': 330,
'astr': 3.0,
'dphi': 0.2,
'repPrey': 3,
'repRadPrey': 1.5,
'repPred': 1,
'repRadPred': 20,
'attPrey': 3,
'attRadPrey': 1.5,
'repCol': 10000000,
'hstr': 1,
'steps': 1500,
    }



paraChange1_name = "astr"
paraChange2_name = "dphi"
steps = 20
reps = 30

total = steps*steps*reps

paraChange1_val = np.linspace(0,1.5,steps)
paraChange2_val = np.linspace(0,1,steps)

pol_scan = np.zeros((steps, steps))
IID_scan = np.zeros((steps, steps))

time_now = datetime.datetime.now()
time_elapsed = 0


for i in range(len(paraChange1_val)):
    
    for j in range(len(paraChange2_val)):

        #np.savetxt(str(saveLoc)+""+str(saveName)+"pol_"+str(paraChange1_name)+"_"+str(paraChange2_name)+".csv", pol_scan, delimiter=",")
  
        pol_reps = []
        IID_reps = []
        
        args[paraChange1_name] = paraChange1_val[i]
        args[paraChange2_name] = paraChange2_val[j]

        npred = args["npred"]
        nprey = args["nprey"]

        args_str = '_'.join(f'{k}_{v}' for k, v in args.items())

        file_h5 = f'{out_str}_{args_str}.states.nc'
        #file_h5 = f'/home/lars/vmodel_output/_nprey_100_npred_1_frange_10_fstr_0.0_visPred_90.0_visPrey_330_astr_3_dphi_0.2_repPrey_3_repRadPrey_1_repPred_1_repRadPred_20_attPrey_3_attRadPrey_1_repCol_10000000_hstr_1_steps_50000.states.nc'

        #print(file_h5)
        



        with h5py.File(file_h5) as fh5:
            vel = np.moveaxis(np.array(fh5['/velocity']), [3,2], [1,3])[:,:,:,:]
            pos = np.moveaxis(np.array(fh5['/position']), [3,2], [1,3])[:,:,:,:]
        

        for rep in range(reps):
            

            
            
            
            
            vel_rep = vel[rep,:,:,:]
            pos_rep = pos[rep,:,:,:]


            #print(np.shape(vel_rep))


            pol = []
            IID = []
            predTime = 2
            for ii in range(1300,args["steps"]-predTime):
                pol.append(calc_order(vel_rep[ii,:nprey,:]))
                pos_calc = pos_rep[ii,:nprey,:]
                scipy.spatial.distance.cdist(pos_calc,pos_calc).sum() / (nprey * (nprey - 1))
                
                
                
                
            pol_reps.append(np.mean(pol))
            IID_reps.append(np.mean(IID))
            
            
        time_last = time_now
            
        time_now = datetime.datetime.now()
            
        time_diff = np.round((time_now-time_last).total_seconds(),2)
            
        time_elapsed += time_diff

        progress = rep+reps*(j+i*steps)
            
        time_finish = (time_elapsed/progress) * (total - progress)
            
        print("progress: "+str(np.round(100*progress/total,2))+" %, time running: "+str(np.round(time_elapsed,2))+" s, est. finish: "+str(np.round(time_finish/60,2))+" min.")
            
            
        pol_scan[i,j] = np.mean(pol_reps)
        IID_scan[i,j] = np.mean(IID_reps)
            
        
np.savetxt(str(saveLoc)+""+str(saveName)+"pol_"+str(paraChange1_name)+"_"+str(paraChange2_name)+".csv", pol_scan, delimiter=",")
np.savetxt(str(saveLoc)+""+str(saveName)+"IID_"+str(paraChange1_name)+"_"+str(paraChange2_name)+".csv", IID_scan, delimiter=",")
 
