import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import h5py
import glob
sys.path.insert(1, '/home/lars/predatorprey-1/predprey-staticnw/')
from animateSwarm import AnimateTools as at
#from TsTools import general as gen
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




#file_h5 = str(file)+".h5"
#file_dat = str(file)+".dat"



mode = 'gif' # 'normal', 'pictures', 'movie', 'gif'
fps = 60
dpi = 200
NsamShow = 4
sizePrey = 1/4
sizePred = sizePrey * 2
cmap = plt.get_cmap('coolwarm') # alternatives 'bwr', 'Reds'




# Construct output file name
out_str = "/home/lars/vmodel_output/vmodel_output/npPred/"
out_str = "/home/lars/vmodel_output/"
args_def = {
'nprey': 100,
'npred': 20,
'frange': 10,
'fstr': 5,
'visPred': 120,
'visPrey': 330,
'astr': 3,
'dphi': 0.2,
'repPrey': 1,
'repRadPrey': 1,
'repPred': 1,
'repRadPred': 20,
'attPrey': 2,
'attRadPrey': 3,
'repCol': 1000,
'hstr': 1,
'steps': 3100,
    }

args_def = {
'nprey': 100,
'npred': 1,
'frange': 10,
'fstr': 5,
'visPred': 300.0,
'visPrey': 330,
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
'steps':6000,
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




var_par = ["dphi", "dphi", "repRadPred", "repRadPred", "repRadPred", 'repPred', 'repPred', 'repPred','visPred', 'visPred', 'visPred' ]
var_val = [0.01, 1.0, 5.0, 10.0, 25.0, 0.5, 2.0, 5.0, 90.0, 180.0, 300.0]



var_par = ["npred"]
var_val = [1]

runs = len(var_par)
runs = 4

list_args = []
for i in range(runs):
    list_args.append(args_def.copy())

if len(var_par) != len(var_val):
    sys.exit( "Lists do not match!")


    
    
for i in range(runs):
    
    #args = list_args[i]
    #args[var_par[i]] = var_val[i]
    args = args_def
    npred = args["npred"]
    nprey = args["nprey"]
    pred_visangle = 2*math.pi*args["visPred"]/360
    prey_visangle = 2*math.pi*args["visPrey"]/360
    


    #args_str = '_'.join(f'{k}_{v}' for k, v in args.items())
    #file_h5 = f'{out_str}_{args_str}.states.nc'
    file_h5 = "/home/lars/vmodel_output/front_v6_vis330_relax_nprey_100_npred_1_frange_10_fstr_50.0_visPred_330.0_visPrey_330_astr_3_dphi_0.2_repPrey_3_repRadPrey_1.5_repPred_21_repRadPred_20_attPrey_3_attRadPrey_1.5_repCol_10000000_hstr_1_steps_4000_fangle_30.0_pangle_0.states.nc"

    run = i
    #file_h5 = "/home/lars/vmodel/output/state.nc"
    #name = "/home/lars/vmodel_output/testingPoly2_change_"+str(var_par[i])+"="+str(var_val[i])+"_"+args_str
    name = "/home/lars/vmodel_output/relax_v6_front_vis330_"+str(run)
    print(name)


    with h5py.File(file_h5) as fh5:


        pos = np.moveaxis(np.array(fh5['/position']), [3,2], [1,3])[run,:,:101,:]  #101 to include pred
        vel = np.moveaxis(np.array(fh5['/velocity']), [3,2], [1,3])[run,:,:101,:]
        #vis = np.array(fh5['/visibility'])

    #print(np.shape(vis[0,nprey:nprey+npred,:nprey,:]))

    #vis_pred = vis[run,nprey:nprey+npred,:nprey,:]

    lines = pos[:,:,:]
    lines = pos
    linesDat = at.datCollector( lines )
    posDat = at.datCollector( pos )
    velDat = at.datCollector( vel )
    positions = [posDat]
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

    #tasks.append( at.plot_nselect_visual(ax, pos, vel, vis, nprey, npred, pred_visangle))
    #tasks.append( at.vision_cones(ax, pos, vel, npred, nprey, pred_visangle))
    #tasks.append( at.color_front(pos, vel, posDat, npred, nprey))
    #tasks.append( at.color_vis(pos, vel, posDat, vis_pred, nprey))
    tasks.append( at.Limits4Pos(positions, ax) )
    tasks.append( at.headAndTail(posDat, ax))
    tasks.append( at.movingArrows(posDat, ax, color = "cyan", noShaft = True))
    #tasks.append( at.lineFadingCOM(posDat, ax))



    # animation
    interval = 1000*(1/fps) # interval in ms
    anim = animation.FuncAnimation(f, tasks.update, interval=interval,
                                   frames=range(0-1, time), repeat=True)


    #plt.show()
    anim.save(name + '.mp4', writer='ffmpeg', dpi=dpi, bitrate=-1, codec='libx264')
    

for i in range(runs):
    
    #args = list_args[i]
    #args[var_par[i]] = var_val[i]
    args = args_def
    npred = args["npred"]
    nprey = args["nprey"]
    pred_visangle = 2*math.pi*args["visPred"]/360
    prey_visangle = 2*math.pi*args["visPrey"]/360
    


    #args_str = '_'.join(f'{k}_{v}' for k, v in args.items())
    #file_h5 = f'{out_str}_{args_str}.states.nc'
    file_h5 = "/home/lars/vmodel_output/side_v6_vis330_relax_nprey_100_npred_1_frange_10_fstr_50.0_visPred_330.0_visPrey_330_astr_3_dphi_0.2_repPrey_3_repRadPrey_1.5_repPred_21_repRadPred_20_attPrey_3_attRadPrey_1.5_repCol_10000000_hstr_1_steps_4000_fangle_30.0_pangle_90.states.nc"

    run = i
    #file_h5 = "/home/lars/vmodel/output/state.nc"
    #name = "/home/lars/vmodel_output/testingPoly2_change_"+str(var_par[i])+"="+str(var_val[i])+"_"+args_str
    name = "/home/lars/vmodel_output/relax_v6_side_vis330_"+str(run)
    print(name)


    with h5py.File(file_h5) as fh5:


        pos = np.moveaxis(np.array(fh5['/position']), [3,2], [1,3])[run,:,:101,:]  #101 to include pred
        vel = np.moveaxis(np.array(fh5['/velocity']), [3,2], [1,3])[run,:,:101,:]
        #vis = np.array(fh5['/visibility'])

    #print(np.shape(vis[0,nprey:nprey+npred,:nprey,:]))

    #vis_pred = vis[run,nprey:nprey+npred,:nprey,:]

    lines = pos[:,:,:]
    lines = pos
    linesDat = at.datCollector( lines )
    posDat = at.datCollector( pos )
    velDat = at.datCollector( vel )
    positions = [posDat]
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

    #tasks.append( at.plot_nselect_visual(ax, pos, vel, vis, nprey, npred, pred_visangle))
    #tasks.append( at.vision_cones(ax, pos, vel, npred, nprey, pred_visangle))
    #tasks.append( at.color_front(pos, vel, posDat, npred, nprey))
    #tasks.append( at.color_vis(pos, vel, posDat, vis_pred, nprey))
    tasks.append( at.Limits4Pos(positions, ax) )
    tasks.append( at.headAndTail(posDat, ax))
    tasks.append( at.movingArrows(posDat, ax, color = "cyan", noShaft = True))
    #tasks.append( at.lineFadingCOM(posDat, ax))



    # animation
    interval = 1000*(1/fps) # interval in ms
    anim = animation.FuncAnimation(f, tasks.update, interval=interval,
                                   frames=range(0-1, time), repeat=True)


    #plt.show()
    anim.save(name + '.mp4', writer='ffmpeg', dpi=dpi, bitrate=-1, codec='libx264')
    
    
for i in range(runs):
    
    #args = list_args[i]
    #args[var_par[i]] = var_val[i]
    args = args_def
    npred = args["npred"]
    nprey = args["nprey"]
    pred_visangle = 2*math.pi*args["visPred"]/360
    prey_visangle = 2*math.pi*args["visPrey"]/360
    


    #args_str = '_'.join(f'{k}_{v}' for k, v in args.items())
    #file_h5 = f'{out_str}_{args_str}.states.nc'
    file_h5 = "/home/lars/vmodel_output/back_v6_vis330_relax_nprey_100_npred_1_frange_10_fstr_50.0_visPred_330.0_visPrey_330_astr_3_dphi_0.2_repPrey_3_repRadPrey_1.5_repPred_21_repRadPred_20_attPrey_3_attRadPrey_1.5_repCol_10000000_hstr_1_steps_4000_fangle_30.0_pangle_180.states.nc"

    run = i
    #file_h5 = "/home/lars/vmodel/output/state.nc"
    #name = "/home/lars/vmodel_output/testingPoly2_change_"+str(var_par[i])+"="+str(var_val[i])+"_"+args_str
    name = "/home/lars/vmodel_output/relax_v6_back_vis330"+str(run)
    print(name)


    with h5py.File(file_h5) as fh5:


        pos = np.moveaxis(np.array(fh5['/position']), [3,2], [1,3])[run,:,:101,:]  #101 to include pred
        vel = np.moveaxis(np.array(fh5['/velocity']), [3,2], [1,3])[run,:,:101,:]
        #vis = np.array(fh5['/visibility'])

    #print(np.shape(vis[0,nprey:nprey+npred,:nprey,:]))

    #vis_pred = vis[run,nprey:nprey+npred,:nprey,:]

    lines = pos[:,:,:]
    lines = pos
    linesDat = at.datCollector( lines )
    posDat = at.datCollector( pos )
    velDat = at.datCollector( vel )
    positions = [posDat]
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

    #tasks.append( at.plot_nselect_visual(ax, pos, vel, vis, nprey, npred, pred_visangle))
    #tasks.append( at.vision_cones(ax, pos, vel, npred, nprey, pred_visangle))
    #tasks.append( at.color_front(pos, vel, posDat, npred, nprey))
    #tasks.append( at.color_vis(pos, vel, posDat, vis_pred, nprey))
    tasks.append( at.Limits4Pos(positions, ax) )
    tasks.append( at.headAndTail(posDat, ax))
    tasks.append( at.movingArrows(posDat, ax, color = "cyan", noShaft = True))
    
    

    #tasks.append( at.lineFadingCOM(posDat, ax))



    # animation
    interval = 1000*(1/fps) # interval in ms
    anim = animation.FuncAnimation(f, tasks.update, interval=interval,
                                   frames=range(0-1, time), repeat=True)


    plt.show()
    anim.save(name + '.mp4', writer='ffmpeg', dpi=dpi, bitrate=-1, codec='libx264')
    
