import matplotlib.pyplot as plt
import numpy as np


mill = np.load("/home/lars/vmodel_output/test_mill_longsim_mill_astr_dphi.npy")
pol = np.load("/home/lars/vmodel_output/test_mill_longsim_pol_astr_dphi.npy")

paraChange1_name = "astr"
paraChange2_name = "dphi"
steps = 10
reps = 20


paraChange1_val = np.linspace(0,3,steps)
paraChange2_val = np.linspace(0,1,steps)

for i in range(len(paraChange1_val)):
    
    for j in range(len(paraChange2_val)):




        
        for rep in range(reps):
            #print(rep)
            fig, ax1 = plt.subplots(1,1, figsize=(20,10))

            steps, steps, reps, time = np.shape(mill)
            x = range(time)

            ax2 = ax1.twinx()
            
            ax1.plot(x, mill[i,j,rep,:], 'b-', alpha = 0.3)
            ax2.plot(x, pol[i,j,rep,:], 'c-', alpha=0.3)

            ax1.set_xlabel('Timestep')
            ax1.set_ylim(-50,50)
            ax2.set_ylim(0,1)
            ax1.set_ylabel('Sum of orientation angle to COM', color='b')
            ax2.set_ylabel('Polarization', color='c')

            plt.rcParams.update({'font.size': 22})

            #plt.show()
            plt.savefig("mill-pol_"+str(paraChange1_name)+"="+str(paraChange1_val[i])+"_"+str(paraChange2_name)+"="+str(paraChange2_val[j])+"_rep="+str(rep)+".png")
            plt.close()