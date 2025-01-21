import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from vmodel.util.util import clean_attrs

def generate_filename(args):

    # Construct output file name
    
    if args.outfolder == "":
    	out_str = "/home/lars/vmodel_output/"
    else:
    	out_str = args.outfolder
    fnamedict = {
        'nprey': args.num_agents,
        'npred': args.num_preds,
        'frange': args.flee_range,
        'fstr': args.flee_strength,
        'visPred': round(args.vision_pred,3),
        'visPrey': round(args.vision_prey,3),
        'astr': args.alignment_strength,
        'dphi': args.dphi,
        'repPrey': args.repulsion_prey,
        'repRadPrey': args.repradius_prey,
        'repPred': args.repulsion_pred,
        'repRadPred': args.repradius_pred,
        'attPrey': args.attraction_prey,
        'attRadPrey': args.attradius_prey,
        'repCol': args.repulsion_col,
        'hstr': args.hunt_str,
        'steps': args.num_timesteps,
        'fangle': round(args.flee_ang,3),
        'pangle': args.pred_angle,
        #'runs': args.num_runs,
        #'times': args.num_timesteps,
        #'dist': args.ref_distance,
        #'perc': args.perception_radius,
        #'topo': args.max_agents,
        #'rngstd': args.range_std,
    }
    formatexts = {'netcdf': 'nc', 'pickle': 'pkl'}
    args_str = '_'.join(f'{k}_{v}' for k, v in fnamedict.items())
    return f'{out_str}_{args_str}.states.{formatexts[args.format]}'


def create_dataset(datas, args):

    ds = xr.Dataset()

    # Clean up attrs dict to be compatible with YAML and NETCDF
    ds.attrs = clean_attrs(vars(args))

    time = np.array(datas[0].time)
    pos = np.array([d.pos for d in datas])
    vel = np.array([d.vel for d in datas])
    flee = np.array([d.flee for d in datas])

    coord_run = np.arange(args.num_runs, dtype=int) + 1
    coord_time = pd.to_timedelta(time, unit='s')
    coord_agent = np.arange(args.num_agents+args.num_preds, dtype=int) + 1
    coord_space = np.array(['x', 'y'])

    coords_rtas = {
        'run': coord_run,
        'time': coord_time,
        'agent': coord_agent,
        'space': coord_space
    }
    
    coords_flee = {
        'run': coord_run,
        'time': coord_time,
        'agent': np.arange(args.num_agents, dtype=int)+1,
        'space': coord_space
    }
    
    dapos = xr.DataArray(pos, dims=coords_rtas.keys(), coords=coords_rtas)
    dapos.attrs['units'] = 'meters'
    dapos.attrs['long_name'] = 'position'
    ds['position'] = dapos

    davel = xr.DataArray(vel, dims=coords_rtas.keys(), coords=coords_rtas)
    davel.attrs['units'] = 'meters/second'
    davel.attrs['long_name'] = 'velocity'
    ds['velocity'] = davel
    
    #daflee = xr.DataArray(flee, dims=coords_rtas.keys(), coords=coords_rtas)
    daflee = xr.DataArray(flee, dims=coords_flee.keys(), coords=coords_flee)
    dapos.attrs['units'] = 'meters'
    dapos.attrs['long_name'] = 'magnitude'
    ds['flee'] = daflee


    ds = ds.transpose('run', 'agent', 'space', 'time')

    # Return only state (position and velocity)
    #if args.no_save_precomputed:
    #    return ds

    coords_rtaa = {
        'run': coord_run,
        'time': coord_time,
        'agent': coord_agent,
        'agent2': coord_agent
    }

    vis = np.array([d.vis for d in datas])
    davis = xr.DataArray(vis, dims=coords_rtaa.keys(), coords=coords_rtaa)
    davis.attrs['units'] = 'boolean'
    davis.attrs['long_name'] = 'visibility'
    ds['visibility'] = davis

    # Tranpose to match data generated from Gazebo
    ds = ds.transpose('run', 'agent', 'agent2', 'space', 'time')

    return ds


def save_dataset(ds, fname, args):

    if args.format == 'pickle':
        with open(fname, 'wb') as f:
            pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.format == 'netcdf':
        comp = dict(zlib=True, complevel=5)
        encoding = None if args.no_compress else {v: comp for v in ds.data_vars}
        ds.to_netcdf(fname, encoding=encoding)

    with open(f'{fname}.yaml', 'w') as f:
        yaml.dump(ds.attrs, f)
