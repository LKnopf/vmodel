import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import h5py
import glob
sys.path.insert(1, '/extra2/knopf/predprey-staticnw/')
from animateSwarm import AnimateTools as at
from TsTools import general as gen
from pathlib import Path
import pickle
from functools import partial
from vmodel import geometry as vgeom
from vmodel import plot
from vmodel.util import color as vcolor
import math

def filter_front_target(pos, vel, target):
    vel_self = vel[target]
    pos_self = pos[target]
    pos = np.delete(pos, target, axis=0)
    pos = pos - pos_self
    
    
    
    
    out_idx = []
    for i in range(len(pos)):
        
        ort_pv = [0,0]
        ort_pv[0] = -vel_self[1]
        ort_pv[1] = vel_self[0]
        
        r_pi = np.array(pos[i])
        if (-r_pi[0] * ort_pv[1] + r_pi[1] * ort_pv[0] < 0):
            out_idx.append(i)
            
    return out_idx
        
def pavas2colors(pavas):
    if np.std(pavas) > 1e-5:
        colors = np.squeeze(pavas)
        colors -= colors.min()
        colors /= colors.max()
    else:
        colors = 'k'
    return colors

paraChange1_name = "--alignment-strength"
paraChange2_name = "--dphi"
steps = 10
paraChange1_val = np.linspace(0,3,steps)
paraChange2_val = np.linspace(0,1,steps)



mode = 'gif' # 'normal', 'pictures', 'movie', 'gif'
fps = 60
dpi = 200
NsamShow = 4
sizePrey = 1/4
sizePred = sizePrey * 2
cmap = plt.get_cmap('coolwarm') # alternatives 'bwr', 'Reds'


variant = "longsim"

# Construct output file name
out_str = "/extra2/knopf/vmodel_output/"+str(variant)+"/"
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

args_def = {
'nprey': 100,
'npred': 1,
'frange': 10,
'fstr': 0.0,
'visPred': 300.0,
'visPrey': 330,
'astr': paraChange1_val[5],
#'dphi': paraChange2_val[3],
'dphi': paraChange2_val[1],
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




#list_args[1]["fstr"] = 20
#list_args[2]["fstr"] = 50

#list_args[3]["frange"] = 5
#list_args[4]["frange"] = 20

#list_args[5]["astr"] = 1
#list_args[6]["astr"] = 10

#list_args[0]["hstr"] = 5.0
#list_args[1]["hstr"] = 20.0

#list_args[2]["visPred"] = 60
#list_args[3]["visPred"] = 30

#list_args[4]["visPrey"] = 200
#list_args[5]["visPrey"] = 100

#var_par = ["fstr", "fstr", "fstr", "frange", "frange", "astr", "astr", "visPred", "visPred", "visPrey", "visPrey"]
#var_val = [5, 20, 50, 5, 20, 1, 10, 60, 30, 200, 100]

#regions: astr 0.4, dphi 0.684
#1.737 0.158
#2.684  0.684
#2.684 0.158


#var_par = ["dphi", "dphi", "repRadPred", "repRadPred", "repRadPred", 'repPred', 'repPred', 'repPred','visPred', 'visPred', 'visPred' ]
#var_val = [0.01, 1.0, 5.0, 10.0, 25.0, 0.5, 2.0, 5.0, 90.0, 180.0, 300.0]



var_par = ["astr", "astr"]
#var_val = [paraChange1_val[11], paraChange1_val[17]]
var_val = [paraChange1_val[5], paraChange1_val[6]]

runs = len(var_par)
#runs = 1

list_args = []
for i in range(runs):
    list_args.append(args_def.copy())

if len(var_par) != len(var_val):
    sys.exit( "Lists do not match!")


for i in range(runs):
    
    args = list_args[i]
    args[var_par[i]] = var_val[i]
    
    npred = args["npred"]
    nprey = args["nprey"]
    pred_visangle = 2*math.pi*args["visPred"]/360
    prey_visangle = 2*math.pi*args["visPrey"]/360



    args_str = '_'.join(f'{k}_{v}' for k, v in args.items())
    file_h5 = f'{out_str}_{args_str}.states.nc'




    #file_h5 = "/home/lars/vmodel/output/state.nc"
    
    reps = range(20)
    
    
    with h5py.File(file_h5) as fh5:
        pos = np.moveaxis(np.array(fh5['/position']), [3,2], [1,3])[:,:,:,:]
        vel = np.moveaxis(np.array(fh5['/velocity']), [3,2], [1,3])[:,:,:,:]
        vis = np.array(fh5['/visibility'])
    print(np.shape(pos))
    for rep in reps:
        name = "/extra2/knopf/vmodel_output/"+str(variant)+"_rep="+str(rep)+"_change_"+str(var_par[i])+"="+str(var_val[i])+"_dphi="+str(args_def["dphi"])+".mp4"
        print(name)
        


        pos_rep=pos[rep,:,:,:]
        vel_rep=vel[rep,:,:,:]

        vis_pred = vis[rep,nprey:nprey+npred,:nprey,:]
        
        posprey = pos[:,:,:]

        lines = pos_rep[:,:,:]
        lines = pos_rep
        linesDat = at.datCollector( lines )
        posDat_prey = at.datCollector( pos_rep[:,:nprey,:] )
        posDat = at.datCollector( pos_rep )
        velDat = at.datCollector( vel_rep )
        positions = [posDat_prey]
        velocities = [velDat]

        posDat.tail_length = 20
        posDat.radius = .25

        # comment line below for colors representing alignment strength
        colors = 'k'
        # get info from files
        time, N, _ = posDat.dat.shape 


    #
        f, ax = plt.subplots(1)
        ax.axis('off')
        ax.set_aspect('equal')
        # Collect update-tasks
        #preds.colors = "r"
        #posDat.colors = "b"
        tasks = at.taskCollector()

        #tasks.append( at.plot_nselect_visual(ax, pos_rep, vel_rep, vis, nprey, npred, pred_visangle))
        #tasks.append( at.vision_cones(ax, pos_rep, vel_rep, npred, nprey, pred_visangle))
        #tasks.append( at.color_front(pos_rep, vel_rep, posDat, npred, nprey))
        #tasks.append( at.color_vis(pos_rep, vel_rep, posDat, vis_pred, nprey))
        tasks.append( at.Limits4Pos(positions, ax) )
        tasks.append( at.headAndTail(posDat_prey, ax))
        tasks.append( at.movingArrows(posDat_prey, ax, color = "cyan", noShaft = True))
        #tasks.append( at.lineFadingCOM(posDat, ax))



        # animation
        interval = 1000*(1/fps) # interval in ms
        anim = animation.FuncAnimation(f, tasks.update, interval=interval,
                                       frames=range(0-1, time), repeat=True)


        #plt.show()

        anim.save(name, writer='ffmpeg', dpi=dpi, bitrate=-1, codec='libx264')
        
        


    
