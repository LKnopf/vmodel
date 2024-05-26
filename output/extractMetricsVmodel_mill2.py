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
       return -math.radians(math.asin((self.a * other.b - self.b * other.a)/(self.length()*other.length())))
       # the dot-product formula, left here just for comparison (does not return angles in the desired range)
       # return math.degrees(math.acos((self.a * other.a + self.b * other.b)/(self.length()*other.length())))

   def length(self):
       return math.sqrt(self.a**2 + self.b**2)

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
saveName = "longsim_dphi_occ"



args = {
'nprey': 100,
'npred': 1,
'frange': 10,
'fstr': 0.0,
'visPred': 300.0,
'visPrey': 330,
'astr': 3,
'dphi': 0.2,
'repPrey': 3,
'repRadPrey': 1.5,
'repPred': 21,  
'repRadPred': 20,
'attPrey': 3,
'attRadPrey': 1.5,
'repCol': 10000000,
'hstr': 1,
'steps': 20000} #4000 for relax
#'fangle': 30, #0.524, #30,
#'pangle':0
#    }



paraChange1_name = "astr"
paraChange2_name = "dphi"
#paraChange2_name = "visPrey"
steps = 10 #10
reps = 20 #20
pred_time = 1200

total = steps*steps*reps

paraChange1_val = np.linspace(0,3,steps)
paraChange2_val = np.linspace(0,1,steps)
#paraChange2_val = np.round(np.linspace(30,359,steps),3)

#paraChange1_val = [30.0]
#paraChange2_val = [0]


mil_scan = np.zeros((steps, steps, reps, args["steps"]-pred_time))
pol_scan = np.zeros((steps, steps, reps, args["steps"]-pred_time))

#mil_scan = np.zeros((1, 3, reps, args["steps"]-pred_time))
#pol_scan = np.zeros((1, 3, reps, args["steps"]-pred_time))

time_now = datetime.datetime.now()
time_elapsed = 0


for i in range(len(paraChange1_val)):
    
    for j in range(len(paraChange2_val)):

        #np.savetxt(str(saveLoc)+""+str(saveName)+"pol_"+str(paraChange1_name)+"_"+str(paraChange2_name)+".csv", pol_scan, delimiter=",")
  
        #pol_reps = np.zeros((reps, args["steps"]))
        IID_reps = []
        CND_reps = []
        
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
                
                for rep in range(reps):
            
                    print(rep)

                    vel_rep = vel[rep,:,:nprey,:]
                    pos_rep = pos[rep,:,:nprey,:]


                    time, N, dim = np.shape(pos_rep)
                    #time -= 10000
                    com_pos = np.zeros((time,2))
                    angle_time = []
                    angle_time_full = np.zeros((time, N, 2))

                    for ii in range(time):
                        com_pos[ii] = np.mean(pos_rep[ii,:,0]),np.mean(pos_rep[ii,:,1])
                        dir_com = com_pos[ii] - pos_rep[ii,:,:]
                        #pol_scan[i,j,rep,ii] = calc_order(vel_rep[ii,:,:])
                        #print(np.shape(dir_com))
                        #print(np.shape(pos_rep[ii,:,:]))
                        #print(np.shape(pos_rep))

                        angle = 0
                        for jj in range(nprey):
                            vector2 = Vect(dir_com[jj,0],dir_com[jj,1])
                            vector1 = Vect(vel_rep[ii,jj,0],vel_rep[ii,jj,1])
                            #print("V1", dir_com[jj,0],dir_com[jj,1], "V2", vel_rep[ii,jj,0],vel_rep[ii,jj,1])


                            try:
                                angle_i = vector1.findClockwiseAngle(vector2)
                            except:
                                print("ERR")

                            angle += np.sin(angle_i)
                            #angle_time_full[ii, jj, :] = angle_i

                        mil_scan[i,j,rep,ii] = angle/nprey

        except:
            print("File not Found, going on")
        
        



                
                
            
            
        

            
        
np.save(str(saveLoc)+""+str(saveName)+"_mill2_"+str(paraChange1_name)+"_"+str(paraChange2_name), mil_scan)
#np.save(str(saveLoc)+""+str(saveName)+"_pol_"+str(paraChange1_name)+"_"+str(paraChange2_name), pol_scan)
#np.savetxt(str(saveLoc)+""+str(saveName)+"IID_"+str(paraChange1_name)+"_"+str(paraChange2_name)+".csv", IID_scan, delimiter=",")
#np.savetxt(str(saveLoc)+""+str(saveName)+"CND_"+str(paraChange1_name)+"_"+str(paraChange2_name)+".csv", CND_scan, delimiter=",")
 
