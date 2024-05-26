import matplotlib.pyplot as plt
import numpy as np
import h5py



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
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.05, 0.2), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)


            
pol_scan = np.load("test1_pol_8.npy")
mil_scan = np.load("test1_mill5_8.npy")        



astr = 0
dphi = 0
#time = time-1

thresh = 0.4
reps = 20
time = 2000000

for rep in range(reps):
    print(rep)
    SMALL_SIZE = 15
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)

    pol = pol_scan[rep, 1:time]
    mill = abs(mil_scan[rep, 1:time])

    mill_top = np.copy(mill)
    mill_top[mill_top < thresh] = "NaN"

    fig, ax1 = plt.subplots(2,1, figsize=(20,10), sharey = True,gridspec_kw={'height_ratios': [1, 3]})
    x = (np.array(range(time-1))*0.02)

    ax1[1].plot(x, (mill), '#6962f5', alpha = 0.9)
    ax1[1].hlines(y=thresh, xmin=0, xmax=max(x), linewidth=2, color='black', linestyles = "--")
    ax1[1].plot(x, pol, color = "#f56262", alpha = 0.9)
    ax1[1].set_xlabel('timesteps')
    ax1[1].set_ylim(0,1)

    multicolor_ylabel(ax1[1],('${M}$',' ,','$Φ$'),('#f56262','k','#6962f5'),axis='y',size=18,weight='bold')

    mill[mill < thresh] = -1
    mill[mill >= thresh] = 1

    new_array = [1]*len(mill)
    colors = [('#6962f5' if i == 1 else '#f56262') for i in mill]
    for i,a in enumerate(new_array):
        ax1[0].barh(y=0, width=a, height=0.5, left=i, color=colors[i])

    ax1[0].set_xlim(0, len(mill))

    ax1[0].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax1[0].axes.get_yaxis().set_visible(False)
    ax1[0].axes.get_xaxis().set_visible(False)

    ax1[1].spines[['right', 'top']].set_visible(False)


    #plt.rcParams.update({'font.size': 22})
    plt.gcf().set_size_inches(12,3)
    plt.margins(x=0)
    plt.tight_layout()
    plt.savefig("milling_series_"+str(rep)+".pdf",bbox_inches="tight")