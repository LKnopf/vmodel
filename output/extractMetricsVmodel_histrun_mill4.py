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
saveName = "histrun"



steps = 4
reps = 40
pred_time = 1
nprey = 100


mil_scan = np.zeros((4, 40, 20000-pred_time))
pol_scan = np.zeros((4, 40, 20000-pred_time))


#files = os.listdir('/extra2/knopf/vmodel_output/single_histruns/')
#files = sorted(files)


files = ("/extra2/knopf/vmodel_output/hm_millstart_fov360_vor/_nprey_100_npred_1_frange_10_fstr_0.0_visPred_300.0_visPrey_359.0_astr_1.6666666666666665_dphi_0.2222222222222222_repPrey_3_repRadPrey_1.5_repPred_21_repRadPred_20_attPrey_3_attRadPrey_1.5_repCol_10000000_hstr_1_steps_20000_fangle_30_pangle_0.states.nc",
"/extra2/knopf/vmodel_output/hm_millstart_fov360_vor/_nprey_100_npred_1_frange_10_fstr_0.0_visPred_300.0_visPrey_359.0_astr_3.0_dphi_0.0_repPrey_3_repRadPrey_1.5_repPred_21_repRadPred_20_attPrey_3_attRadPrey_1.5_repCol_10000000_hstr_1_steps_20000_fangle_30_pangle_0.states.nc",
"/extra2/knopf/vmodel_output/hm_millstart_fov360_occ/_nprey_100_npred_1_frange_10_fstr_0.0_visPred_300.0_visPrey_359.0_astr_1.6666666666666665_dphi_0.2222222222222222_repPrey_3_repRadPrey_1.5_repPred_21_repRadPred_20_attPrey_3_attRadPrey_1.5_repCol_10000000_hstr_1_steps_20000_fangle_30_pangle_0.states.nc",
"/extra2/knopf/vmodel_output/hm_millstart_fov360_occ/_nprey_100_npred_1_frange_10_fstr_0.0_visPred_300.0_visPrey_359.0_astr_3.0_dphi_0.0_repPrey_3_repRadPrey_1.5_repPred_21_repRadPred_20_attPrey_3_attRadPrey_1.5_repCol_10000000_hstr_1_steps_20000_fangle_30_pangle_0.states.nc")

for i in files: print(i)

for i in range(1):
    
    for j in range(steps):


        file_h5 = "/extra2/knopf/vmodel_output/single_histruns/"+files[j]
        file_h5 = files[j]
        
        print(file_h5)
        

        with h5py.File(file_h5) as fh5:
            vel = np.moveaxis(np.array(fh5['/velocity']), [3,2], [1,3])[:,:,:,:]
            pos = np.moveaxis(np.array(fh5['/position']), [3,2], [1,3])[:,:,:,:]
            
            
            
                
            for rep in range(reps):
            
                print(rep)

                vel_rep = vel[rep,:,:nprey,:]
                pos_rep = pos[rep,:,:nprey,:]
                print(np.shape(pos_rep))
    
                time, N, dim = np.shape(pos_rep)
                    #time -= 10000
                com_pos = np.zeros((time,2))
                angle_time = []
                angle_time_full = np.zeros((time, N, 2))

                for ii in range(time):
                    com_pos[ii] = np.mean(pos_rep[ii,:,0]),np.mean(pos_rep[ii,:,1])
                    dir_com = com_pos[ii] - pos_rep[ii,:,:]


                    angle = 0
                    for jj in range(nprey):
                        a = ((dir_com[jj,0],dir_com[jj,1]))
                        b = ((vel_rep[ii,jj,0],vel_rep[ii,jj,1]))
                            
                        pol = (np.cross((a/np.linalg.norm(a)),b))
                            
                        angle += pol

                    mil_scan[j,rep,ii] = angle/nprey
                np.save(str(saveLoc)+""+str(saveName)+"_mill_histrun_", mil_scan)
                        


        
        



                
                
            
            
        

            
        
#np.save(str(saveLoc)+""+str(saveName)+"_mill4_"+str(paraChange1_name)+"_"+str(paraChange2_name), mil_scan)
#np.save(str(saveLoc)+""+str(saveName)+"_pol_"+str(paraChange1_name)+"_"+str(paraChange2_name), pol_scan)
#np.savetxt(str(saveLoc)+""+str(saveName)+"IID_"+str(paraChange1_name)+"_"+str(paraChange2_name)+".csv", IID_scan, delimiter=",")
#np.savetxt(str(saveLoc)+""+str(saveName)+"CND_"+str(paraChange1_name)+"_"+str(paraChange2_name)+".csv", CND_scan, delimiter=",")
 
