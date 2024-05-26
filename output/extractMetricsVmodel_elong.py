import vmodel
import os
import numpy as np
import h5py
import datetime
import scipy.spatial
import math


class Vect:

   def __init__(self, a, b):
        self.a = a
        self.b = b

   def findClockwiseAngle(self, other):
       # using cross-product formula
       return -math.degrees(math.asin((self.a * other.b - self.b * other.a)/(self.length()*other.length())))
       # the dot-product formula, left here just for comparison (does not return angles in the desired range)
       # return math.degrees(math.acos((self.a * other.a + self.b * other.b)/(self.length()*other.length())))

   def length(self):
       return math.sqrt(self.a**2 + self.b**2)
    
def get_elongation(pos, vel):
    meanDir = np.mean(vel, axis=0)
    meanDir = meanDir / np.linalg.norm(meanDir)
    
    p_meanDir= np.zeros(2)
    p_meanDir[0] = meanDir[1]
    p_meanDir[1] = -meanDir[0]
    
    dist = []
    p_dist = []
    
    for i in range(len(pos[:,0])):
        
        dist.append(pos[i,0] * meanDir[0] + pos[i,1] * meanDir[1])
        p_dist.append(pos[i,0] * p_meanDir[0] + pos[i,1] * p_meanDir[1])
    

    elongation = (np.array(dist).max()-np.array(dist).min()) / (np.array(p_dist).max()-np.array(p_dist).min())
    return elongation

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
out_str = "/extra2/knopf/vmodel_output/longsim/"
saveLoc = "/extra2/knopf/vmodel_output/data/"
saveName = "single_mill_longsim"


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
'repPred': 21,  
'repRadPred': 20,
'attPrey': 3,
'attRadPrey': 1.5,
'repCol': 10000000,
'hstr': 1,
'steps': 20000,
    }



paraChange1_name = "astr"
paraChange2_name = "repRadPrey"
steps = 10 #10
reps = 20 #20
pred_time = 1200

total = steps*steps*reps

paraChange1_val = np.linspace(0,3,steps)
#paraChange2_val = np.linspace(0,1,steps)
paraChange2_val = np.linspace(0.7,3.7,steps)
elong_scan = np.zeros((steps, steps, reps, args["steps"]-pred_time))



time_now = datetime.datetime.now()
time_elapsed = 0


for i in range(len(paraChange1_val)):
    
    for j in range(len(paraChange2_val)):

        #np.savetxt(str(saveLoc)+""+str(saveName)+"pol_"+str(paraChange1_name)+"_"+str(paraChange2_name)+".csv", pol_scan, delimiter=",")
  
        #pol_reps = np.zeros((reps, args["steps"]))
        elon_reps = []

        
        args[paraChange1_name] = paraChange1_val[i]
        args[paraChange2_name] = paraChange2_val[j]

        npred = args["npred"]
        nprey = args["nprey"]

        args_str = '_'.join(f'{k}_{v}' for k, v in args.items())

        file_h5 = f'{out_str}_{args_str}.states.nc'
        #file_h5 = f'/extra2/knopf/vmodel_output/longsim/_nprey_100_npred_1_frange_10_fstr_0.0_visPred_300.0_visPrey_330_astr_1.6666666666666665_dphi_0.1111111111111111_repPrey_3_repRadPrey_1.5_repPred_21_repRadPred_20_attPrey_3_attRadPrey_1.5_repCol_10000000_hstr_1_steps_20000.states.nc'

        print(file_h5)
        


        try:
            with h5py.File(file_h5) as fh5:
                vel = np.moveaxis(np.array(fh5['/velocity']), [3,2], [1,3])[:,:,:,:]
                pos = np.moveaxis(np.array(fh5['/position']), [3,2], [1,3])[:,:,:,:]

        except:
            print("File not Found, going on")
        
        

        for rep in range(reps):
            
            print(rep)
            
            vel_rep = vel[rep,:,:nprey,:]
            pos_rep = pos[rep,:,:nprey,:]


            time, N, dim = np.shape(pos_rep)
            #print(time)
            com_pos = np.zeros((time,2))
            angle_time = []
            #angle_time_full = np.zeros((time, N, 2))

            for ii in range(time):
                elong_scan[i,j,rep,ii] = get_elongation(pos_rep[ii,:nprey,:],vel_rep[ii,:nprey,:])

                
                
            

            
            
        time_last = time_now
            
        time_now = datetime.datetime.now()
            
        time_diff = np.round((time_now-time_last).total_seconds(),2)
            
        time_elapsed += time_diff

        progress = rep+reps*(j+i*steps)
            
        time_finish = (time_elapsed/progress) * (total - progress)
            
        print("progress: "+str(np.round(100*progress/total,2))+" %, time running: "+str(np.round(time_elapsed,2))+" s, est. finish: "+str(np.round(time_finish/60,2))+" min.")
            
            
        

            
        
np.save(str(saveLoc)+""+str(saveName)+"_elong_"+str(paraChange1_name)+"_"+str(paraChange2_name), elong_scan)

 
