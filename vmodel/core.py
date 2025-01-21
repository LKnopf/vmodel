import argparse

import numpy as np
import math
from math import cos, sin

from vmodel.geometry import subtended_angle_circle, voronoi_neighbors
from vmodel.math import filter_by_angle, limit_norm, wrap_angle
from vmodel.random import random_uniform_within_circle
from vmodel.visibility import visibility_set
from vmodel.romanczuk import flock, flock_pred, flee, addForces, flock_col

# Uncomment the @profile decorator and run to perform line-by-line profiling:
# kernprof -l -v vmodel ...
# @profile

def angle_between(vhat1, vhat2, tol = 5e-6):
    cosang = np.dot(vhat1, vhat2)
    if math.fabs(cosang) > 1.0:
        if math.fabs(cosang) - 1.0 < tol:
            cosang = math.modf(cosang)[1]
        else:
            raise ValueError('Invalid arguments (vectors not normalized?)')
    return math.acos(cosang)


def getCOM(positions):
    return np.mean(positions, axis = 0)
    

def filter_front(pos, vel_self, idx, args):
    out_idx = []
    for i in range(len(pos)):
        
        ort_pv = [0,0]
        ort_pv[0] = -vel_self[1]
        ort_pv[1] = vel_self[0]
        
        r_pi = np.array(pos[i])
        if (-r_pi[0] * ort_pv[1] + r_pi[1] * ort_pv[0] < 0):
            out_idx.append(i)
            
    return out_idx
    
    
    
def filter_angle(dist_full, pos, vel_self, idx, args, blind_angle):
    out_idx = []
    if sum(vel_self) != 0:
        vel_self = vel_self/np.linalg.norm(vel_self)
    for i in range(len(idx)):
        dist = dist_full[i]
        if dist != 0:
            ang_pos = angle_between(vel_self, pos[i]/dist)
            if ang_pos < blind_angle:
                out_idx.append(i)
        else:
        	print("TEST2")
            
    return out_idx


def update_single(i, positions, velocities, prey_flee, func, args, pred_hunt, t, pastCOM, pred_time, dist_full):
    """Update a single agent
    Args:
        i: index of focal agent (zero-indexed)
        positions: absolute positions of agents
        func: flocking algorithm function to apply
        args: parsed command-line arguments
    Returns:
        command: velocity command for focal agent
        cache: holds precomputed values
    """
    idx_self = i

        
    cache = argparse.Namespace()

    # Shorthands for arguments
    ndim = args.num_dims

    # Get relative positions of others
    pos_self = positions[i]

    pos_others = np.delete(positions, i, axis=0)
    
    # Create true COM for hunt
    
    if i >= args.num_agents:
    	com_true = getCOM(pos_others)
    
    pos = pos_others - pos_self
    
    vel_self = velocities[i]
     
    #use to check if vel of pred is calculated correctly
    #if sum(vel_self) == 0:
        #print("NO VEL",i)
    
    vel_others = np.delete(velocities, i, axis=0)
    
    vel = vel_others
    
    
    
    # pos_waypoint = args.pos_waypoint - pos_self

    # Keep track of original agent indices
    idx = np.arange(len(pos), dtype=int)

    # Radii of agents
    rad = np.full(len(pos), args.radius)

    # Compute relative distances
    dist = dist_full[idx_self, :]
    dist = np.delete(dist, i, axis=0)

    cache.dist = dist.copy()
    # cache.dist = np.insert(dist, i, 0.0)

    # Optionally add false positive detections
    if args.num_clutter > 0.0:
        # Add more clutter if not enough available
        num_clutter = np.random.poisson(args.num_clutter)

        if num_clutter > 0:
            low, high = args.radius_safety, args.perception_radius
            pos_clutter = random_uniform_within_circle(low, high,
                                                       size=(num_clutter, ndim))
            pos_clutter = np.array(pos_clutter).reshape(-1, ndim)

            # while len(clutter_list) < num_clutter:
            #     low, high = args.radius_arena, args.radius_arena + args.perception_radius
            #     clutter = sample_fn(low=low, high=high)
            #     clutter_list.append(clutter)

            # # Ranomly choose clutters from list
            # choice = np.random.choice(np.arange(len(clutter_list)), num_clutter)
            # pos_clutter = np.array(clutter_list)[choice].reshape(-1, ndim)

            # Add clutter to rest of detections
            dist_clutter = np.linalg.norm(pos_clutter, axis=1)
            pos = np.concatenate([pos, pos_clutter])
            dist = np.concatenate([dist, dist_clutter])
            idx = np.arange(len(pos), dtype=int)
            rad = np.concatenate([rad, np.full(len(pos_clutter), args.radius)])

            

    
    # If using visual migration, check if the waypoint is visible (as with agents)
    # waypoint_visible = True
    # if args.visual_migration:
    #     idx_waypoint = idx[-1] + 1
    #     idx = np.append(idx, idx_waypoint)
    #     rad = np.append(rad, args.radius_waypoint)
    #     dist = np.append(dist, np.linalg.norm(pos_waypoint))
    #     pos = np.append(pos, [pos_waypoint], axis=0)

    # Filter out agents by metric distance
    if args.perception_radius > 0:
        mask = dist < args.perception_radius
        pos, dist, idx, rad = pos[mask], dist[mask], idx[mask], rad[mask]

    # Filter out agents by angle proportion of field of view
    if args.perception_angle > 0:
        angles_rad = subtended_angle_circle(dist, rad)
        mask = angles_rad > np.deg2rad(args.perception_angle)
        pos, dist, idx, vel, rad = pos[mask], dist[mask], idx[mask], vel_others[mask], rad[mask]

    # Filter out occluded agents
    if args.filter_occluded and pred_time:

        mask = visibility_set(pos, rad, dist)

        pos, dist, idx, vel = pos[mask], dist[mask], idx[mask], vel_others[mask]
        
    if args.filter_occluded and pred_time!=True:

        mask = visibility_set(pos[:args.num_agents-1, :], rad[:args.num_agents-1], dist[:args.num_agents-1])
        mask = np.append(mask, False)
        pos, dist, idx, vel = pos[mask], dist[mask], idx[mask], vel_others[mask]




        

    # Filter out waypoint (if visible)
    # if args.visual_migration:
    #     waypoint_visible = idx_waypoint in idx
    #     if waypoint_visible:
    #         mask = idx != idx_waypoint
    #         pos, dist, idx = pos[mask], dist[mask], idx[mask]

    # Filter out agents by topological distance
    if args.max_agents > 0:
        indices = dist.argsort()[:(args.max_agents)]
        pos, dist, idx = pos[indices], dist[indices], idx[indices]

    if args.topo_angle > 0:
        angle = np.deg2rad(args.topo_angle)
        mask = filter_by_angle(pos, angle)
        pos, dist, idx = pos[mask], dist[mask], idx[mask]

    if args.filter_voronoi and pred_time:
        posme = np.insert(pos, 0, np.zeros(ndim), axis=0)
        #print(posme)
        try:
            indices = np.array(voronoi_neighbors(posme)[0]) - 1
        except Exception:
            #print("ERRER")
            pass  # in case there are not enough points, do nothing
        else:
            #print("OKAY")
            pos, dist, idx, vel = pos[indices], dist[indices], idx[indices], vel_others[indices]

    if args.filter_voronoi and pred_time!=True:
        posme = np.insert(pos[:args.num_agents-1, :], 0, np.zeros(ndim), axis=0)
        #print(posme)
        try:
            indices = np.array(voronoi_neighbors(posme)[0]) - 1
        except Exception:
            #print("ERRER")
            pass  # in case there are not enough points, do nothing
        else:
            #print("OKAY")
            pos, dist, idx, vel = pos[indices], dist[indices], idx[indices], vel_others[indices]
    
    #reduce visibility to agents in front
    visPrey = 0.5*2*math.pi*args.vision_prey/360
    visPred = 0.5*2*math.pi*args.vision_pred/360
    
    posF, distF, idxF, velF = pos, dist, idx, vel
    
    
    #filter out agents in blind angle
    if i < args.num_agents:
        idx_front = filter_angle(dist, pos, vel_self, idx, args, blind_angle = visPrey)
    else:
        idx_front = filter_angle(dist, pos, vel_self, idx, args, blind_angle = visPred)
        
    pos, dist, idx, vel = pos[idx_front], dist[idx_front], idx[idx_front], vel[idx_front]


    # Save visibility data (after applying perception filtering)
    visibility = np.zeros(len(pos_others), dtype=bool)
    visibility[idx[idx < len(pos_others)]] = True

    # cache.vis = np.insert(visibility, i, False)
    cache.vis = visibility.copy()

    # Optionally add noise to range and bearing
    # We add range/bearing noise *after* the visual filtering since it prevents false
    # occlusions where (visually speaking) there weren't any
    if args.range_std > 0 or args.bearing_std > 0:
        print("TEST")
        xs, ys = pos.T.copy()
        distance = dist.copy()
        bearing = np.arctan2(ys, xs)  # [-pi, +pi]
        noise_distance, noise_bearing = np.random.normal(size=(ndim, len(xs)))
        noise_distance *= args.range_std
        noise_bearing *= np.deg2rad(args.bearing_std)
        distance += (noise_distance * distance)  # distance noise is distance-dependent!
        bearing += noise_bearing
        bearing = wrap_angle(bearing)  # wrap to [-pi, +pi]
        xs, ys = distance * np.cos(bearing), distance * np.sin(bearing)
        pos = np.array([xs, ys]).T
        dist = np.linalg.norm(pos, axis=1)

    # Optionally discard agents with false negative prob
    if args.false_negative_prob:
        keep_prob, size = 1 - args.false_negative_prob, len(pos)
        mask = np.random.binomial(n=1, p=keep_prob, size=size).astype(bool)
        pos, dist = pos[mask], dist[mask]

    # Compute command from interactions (if any relative positions)
    command_interaction = np.zeros(ndim)
    
    


    # Compute migration command
    command_migration = np.zeros(ndim)

    # Migrate to migration point
    if len(args.migration_point) > 0:
        pos_waypoint = (args.migration_point - pos_self)
        command_migration = limit_norm(pos_waypoint, args.migration_gain)

    # Follow general migration direction
    if len(args.migration_dir) > 0:
        command_migration = limit_norm(args.migration_dir, args.migration_gain)
        


    
    
    command_flee_pred = [0,0]
    command_interaction = [0,0]
    command_migration = [0,0]

    pos_prey = pos[idx < args.num_agents-1]
    pos_preyF = posF[idxF < args.num_agents-1]
    pos_predF = posF[idxF >= args.num_agents-1]
    

    #print("pos",pos, idx)
    #print("prey", pos[idx < args.num_agents])
    
    vel_prey = vel[idx < args.num_agents-1]
    vel_preyF = velF[idxF < args.num_agents-1]
    vel_predF = velF[idxF >= args.num_agents-1]
    
    dist_prey = dist[idx < args.num_agents-1]
    dist_preyF = distF[idxF < args.num_agents-1]
    dist_predF = distF[idxF >= args.num_agents-1]
    
    #create array which contains only preds that i can see
    ################
    #idx_sort = idx.sort()
    #idx_pred_vis = idx
    #idx_pred = np.array(range(args.num_preds))+args.num_agents
    
    idx_pred_vis = idx[idx >= args.num_agents-1]

    #create arrays without preds for prey-prey interaction




    
    if len(idx) > -1:
        if i < args.num_agents:
            #pos_preyF = pos180[idx180 < args.num_agents-1]
            #vel_preyF = vel180[idx180 < args.num_agents-1]
            
            
            
            
            fflee = [0,0]
            fflock = flock(pos_self, pos_prey , vel_self, vel_prey , dist_prey, args)
            
            
            
            
            #idx[idx >= args.num_agents-1]
            
            if args.col_style == 0:
                col = False
            else:
                fcol, col, vel_col = flock_col(pos_self, pos_preyF , vel_self, vel_preyF , t, idx_self, idxF, dist_preyF, args)
                

            if len(idx_pred_vis) > 0 and pred_time:
                fflee = flee(idx_pred_vis, idx, pos, vel, args, pos_self)
                
		 
                prey_flee[i,:] = np.linalg.norm(fflee), np.linalg.norm(fflock)
                

                #if np.linalg.norm(fflee) > 0:
                #	prey_flee[i] = 1
                #elif prey_flee[i] > 0:
                #	prey_flee[i] = 2
                	

                if args.pred_segments == 1:
                    for pred_idx in idx_pred_vis:
                        
                        pos_long = np.copy(pos)
        
                        idx_pred = np.where(idx == pred_idx)[0][0]

                        pos_long[idx_pred] -= args.pred_length*(vel[idx_pred]/np.linalg.norm(vel[idx_pred]))
                    
                    fflee = flee(idx_pred_vis, idx, pos_long, vel, args, pos_self)
                    
                    	
                

            command_interaction = addForces(fflock, fflee, vel_self, args)
            if col == "front":
                command_interaction = vel_col



                

        else:
            command_flee_pred = [0,0]
            seekPrey = False
            col = False
            fhunt = np.array([0,0])

            if args.pred_hunt == 1:
                if t >= (args.pred_time+args.trans_pred)*args.delta_time:
                    if len(pos_prey) == 0:
                        fhunt = vel_self
                        seekPrey = True
                    else:
                        com_vis = getCOM(pos_prey)
                        fhunt = limit_norm(com_vis, 1)


                pos_pred = pos[idx >= args.num_agents]
                vel_pred = vel[idx >= args.num_agents]
                dist_pred = dist[idx >= args.num_agents]
                fflock = flock_pred(pos_self, pos_pred , vel_self, vel_pred , dist_pred, args)
                
                if args.col_style == 0:
                    col = False
                else:
                    fcol, col, vel_col = flock_col(pos_self, pos_predF , vel_self, vel_predF , t, idx_self, idxF, dist_predF, args)
                
                
                command_interaction = addForces(np.array(fflock)/2, np.array(fhunt)/2, vel_self, args, seekPrey, pred=True)
                if col == "front":
                    command_interaction = vel_col
                
                
            elif args.pred_hunt == 2:
            #one attack, vision limited
                if pastCOM == False and pred_time:
                    com_dist = getCOM(pos)
                    if np.linalg.norm(com_dist) < 3 and np.linalg.norm(com_dist) != 0.0:
                        pastCOM = True
                        print("PASSED COM")
                    else:
                        com_vis = getCOM(pos)
                        fhunt = limit_norm(com_vis, 1)


                        pos_pred = pos[idx >= args.num_agents]
                        vel_pred = vel[idx >= args.num_agents]
                        dist_pred = dist[idx >= args.num_agents]
                        
            elif args.pred_hunt == 3:
            #one attack, sees all
                if pastCOM == False and pred_time:
                
                    
                    com_dist = com_true-pos_self
                    
                    if np.linalg.norm(com_dist) < 3 and np.linalg.norm(com_dist) != 0.0:
                        pastCOM = True
                        print("PASSED COM")
                    else:
                        
                        fhunt = limit_norm(com_dist, 1)
                        #print(type(fhunt))
                
            
                #command_interaction = fhunt*args.hunt_str
            
            if pastCOM and args.pred_hunt >= 2:
                #print("STAY COURSE")
                command_interaction = vel_self
            else:
                command_interaction = addForces([0,0], np.array(fhunt)*args.hunt_str, vel_self, args)
            #print(command_interaction)
            


    command = command_interaction + command_migration + command_flee_pred #+ 0.2 * vel_self

    

    # Limit speed
    command = limit_norm(command, args.max_speed)

    
    if i >= args.num_agents:
        #print(command, np.linalg.norm(command))
        command = command * 2

        

    return np.array(command), cache, pred_hunt, pastCOM, pos_prey, prey_flee
