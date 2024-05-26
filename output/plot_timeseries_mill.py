import matplotlib.pyplot as plt
import numpy as np
import h5py


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


def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.150, 0.3), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)



file_h5 = "timeseries_mill_long__nprey_100_npred_1_frange_10_fstr_0.0_visPred_120_visPrey_280.0_astr_3.0_dphi_0.11_repPrey_3_repRadPrey_1.5_repPred_21_repRadPred_20_attPrey_3_attRadPrey_1.5_repCol_10000000_hstr_1_steps_200000_fangle_30.0_pangle_0.states.nc"
        
nprey = 100
reps = 20
time = 2000000

mil_scan = np.zeros((reps, time))
mil_ring_scan = np.zeros((reps, time))
pol_scan = np.zeros((reps, time))


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
            pol_scan[rep,ii] = calc_order(vel_rep[ii,:,:])

            angle = 0
            angle_ring=[]
            for jj in range(nprey):
                a = ((dir_com[jj,0],dir_com[jj,1]))
                a_len = np.linalg.norm(a)
                b = ((vel_rep[ii,jj,0],vel_rep[ii,jj,1]))
                            
                mill = (np.cross((a/a_len),b))
                angle += mill

                            
                

            mil_scan[rep,ii] = angle/nprey

            
            
np.save("test1_mill5_8", mil_scan)
np.save("test1_pol_8", pol_scan)




#pol_scan = np.load("test1_pol_8.npy")
#mil_scan = np.load("test1_mill5_8.npy")
#mil_ring_scan = np.load("test1_mill5_ring_8.npy")



astr = 0
dphi = 0
#time = time-1

thresh = 0.4
reps = 20

for rep in range(reps):
    #mill = mill[astr,dphi,rep,:]
    pol = pol_scan[rep, 1:time]
    mill = abs(mil_scan[rep, 1:time])

    mill_top = np.copy(mill)
    mill_top[mill_top < thresh] = "NaN"

    fig, ax1 = plt.subplots(1,1, figsize=(20,10), sharey = True)
    #steps, steps, reps, time = np.shape(mill)
    x = (np.array(range(time-1))*0.02)

    ax2 = ax1.twinx()
    ax1.plot(x, (mill), 'b-', alpha = 0.5)
    ax1.plot(x, mill_top, 'c-', alpha=0.5)
    ax1.hlines(y=thresh, xmin=0, xmax=max(x), linewidth=2, color='black', linestyles = "--")
    ax2.plot(x, pol, color = "orange", alpha = 0.5)
    ax1.set_xlabel('timesteps')
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,1)
    #ax1.set_ylabel('milling ${M}$', color = "blue")
    #ax2.set_ylabel('polarization $Φ$', color = "orange")
    multicolor_ylabel(ax1,('${M}$',' ,','$Φ$'),('orange','k','blue'),axis='y',size=15,weight='bold')
    #ax2.set_ylabel('Polarization', color='c')

    #plt.rcParams.update({'font.size': 22})
    plt.gcf().set_size_inches(5,3)
    #plt.tight_layout()
    plt.savefig("/extra2/knopf/vmodel_output/data/milling_series_"+str(rep)+".pdf",bbox_inches="tight")