
import numpy as np
import math


#use with filter-voronoi



def flock(pos_self: np.ndarray, pos: np.ndarray, vel_self: np.ndarray, vel: np.ndarray, dist_full, args):
    
    coll = False
    dt = args.delta_time
    rep_strength = 2
    alg_strength = args.alignment_strength
    
    vel_dif = vel

    vel_dif[:,0] -= vel_self[0]
    vel_dif[:,1] -= vel_self[1]

    f_alg = np.sum(vel_dif, axis = 0)
    
    pos_dif = pos
    
    
    
    #pos_dif[:,0] -= pos_self[0]
    #pos_dif[:,1] -= pos_self[1]

    f_rep = np.zeros(2)
    n_rep = 0

    for i in range(len(pos)):
        dist = dist_full[i]
        pos_dif[i] = pos[i] / dist
        f_repChange, coll = att(dist, np.array(pos_dif[i]), args)
        #f_repChange, coll = attDRA(dist, np.array(pos_dif[i]), args)
        f_rep += f_repChange

    if len(pos) == 0:
        f_rep = rep_strength * f_rep * 0
        f_alg = alg_strength * f_alg * 0
    else:
    
        f_rep = rep_strength * (f_rep / len(pos))
        #print("frep", f_rep)

        f_alg = alg_strength * (f_alg / len(pos))
        #print("falg", f_alg)
    
    force = f_alg + f_rep
    
    
    return force


def flock_col(pos_self: np.ndarray, pos: np.ndarray, vel_self: np.ndarray, vel: np.ndarray, t, idx_self, idx, dist_full, args):
    
    col_set = False
    dt = args.delta_time


    
    pos_dif = pos
    

    preySize = 2*args.prey_size+4*args.delta_time

    
    f_rep = np.zeros(2)
    vel_col = np.zeros(2)
    idx_front = []
    dist_store= []

    
    for i in range(len(pos)):
        dist = dist_full[i]
        
        if (abs(dist) < preySize):
            if t/dt <= 10:
                f_repChange, coll = col(dist, np.array(pos_dif[i]), args)
            else:
                f_repChange, coll= col2(pos[i], vel[i], vel_self, dist, args)
                
            
            f_rep += f_repChange

            if coll == "front":
                idx_front.append(i)
                dist_store.append(abs(dist))
                col_set = "front"

    if len(idx_front) > 0:
        if args.col_style == 2:
            #for checking all collisions
            vel_col = avoid_col_mult(pos, vel_self, vel, idx_front, args)
        elif args.col_style == 1:
            #for only closest collision
            vel_col = avoid_col_mult(pos, vel_self, vel, [idx_front[np.array(dist_store).argmin()]], args)



        
    if len(pos) == 0:
        f_rep = f_rep * 0
        
    
    #part for physical world limits
    #sizebox = 10 
    #if abs(pos_self[0])+args.prey_size+args.delta_time > sizebox or abs(pos_self[1]) +args.prey_size+args.delta_time > sizebox:
    #	col_set = "front"
    #	vel_col = -vel_self



    
    return f_rep, col_set, vel_col
    
    

def addForces(flock, flee, vel_self, args, seekPrey = False, pred=False):
    dphi = args.dphi
    
    
    dt = args.delta_time
    
    phi = np.arctan2(vel_self[1],vel_self[0])
    phi_before = phi
    #phi = math.fmod(phi, 2 * math.pi)
    #phi = math.fmod(2*math.pi+phi, 2 * math.pi)
    
    
    force = (flock+flee)


    phi_f_dir = np.arctan2(force[1],force[0])

    
    #forcep = -force[0] * sin(lphi) + force[1] * cos(lphi);
    phi_f = -force[0] * np.sin(phi) + force[1] * np.cos(phi)
    
    phi_diff = phi-phi_f_dir

    #print("forces", flock, flee, force, phi, phi_f)
    noisep = math.sqrt(dt * 2 * dphi)
    noise = noisep * np.random.normal()
    
    if seekPrey:
        noise = noise * 1.0
    
    
    
    
    if abs(vel_self[0]) + abs(vel_self[1]) != 0:

        vproj = math.sqrt(vel_self[0]*vel_self[0] + vel_self[1]*vel_self[1])

        phi += ((phi_f*dt + noise)) / vproj
        
        phi_diff_after = phi-phi_f_dir
        

        #check later if needed again
        #if np.sign(phi_diff) != np.sign(phi_diff_after) and phi_f != 0:
        #    phi = phi_f_dir


        phi = math.fmod(phi, 2 * math.pi)
        phi = math.fmod(2*math.pi+phi, 2 * math.pi)
    else:
        phi = math.atan2(force[1], force[0])

    



    return np.array((np.cos(phi), np.sin(phi)))


def flee(idx_pred_vis, idx, pos, vel, args, pos_self):
    
    command_flee_pred = [0,0]
    
                
    for pred_idx in idx_pred_vis:
        
        idx_pred = np.where(idx == pred_idx)[0][0]

        pos_pred = pos[idx_pred]
        vel_pred = vel[idx_pred]/2
        dist_pred = abs(np.linalg.norm(pos_pred))
        pred_hunt = True #disable for orthog. vector

        ort_pv = [0,0]
        ort_pv[0] = -vel_pred[1]
        ort_pv[1] = vel_pred[0]

        
        r_pi = np.array(pos_pred)
        command_flee = np.zeros(2)


        r_ip = [0,0]

        side = 1 # if 1-> prey is right, if -1-> prey is left (?)


        ### Calc relative distance vector and corresponding unit vector
        r_ip = pos_pred
        ### Calc if prey is left or right from pred
        ### (project r_ip on unit vector right from u of pred  -> [u[1], -u[0]])
        if ((r_ip[0] * vel_pred[1] - r_ip[1] * vel_pred[0]) > 0):
            side = -1
        ### change fleeing direction
        
        
        fleeang = (args.flee_ang/360)*2*math.pi


        ang = math.atan2(r_ip[1], r_ip[0]) - side * fleeang
        #print(ang, side)

        ang = math.fmod(ang, 2 * math.pi)
        ang = math.fmod(2*math.pi+ang, 2 * math.pi)
        
        x0 = args.flee_range

        if (dist_pred > 0):

            command_flee[0] = np.cos(ang)
            command_flee[1] = np.sin(ang)

        steepness = -1
        
        
        fleestr = 0
        
        if (dist_pred < x0):
            fleestr = 0.5*(np.tanh(steepness*(dist_pred-x0))+1.0)
        command_flee_pred = command_flee_pred - command_flee * fleestr

    
    if abs(command_flee_pred[0])+abs(command_flee_pred[1]) != 0:
        command_flee_pred = command_flee_pred/abs(np.linalg.norm(command_flee_pred))
    #print("flee", command_flee_pred)
    
    #print(command_flee_pred*args.flee_strength, command_flee_pred, args.flee_strength, np.linalg.norm(command_flee_pred*args.flee_strength), np.linalg.norm(command_flee_pred))
    
    return command_flee_pred*args.flee_strength



def att(dist_interaction, dif, args):
      
    l0 = args.repradius_prey
    l1 = args.repradius_prey
    k_att = args.attraction_prey
    k_rep = args.repulsion_prey
    fstrength = 0
    preySize = 2*args.prey_size+1*args.delta_time

    coll = False
    


        
    if (abs(dist_interaction) < l0):
        #print("REPULSE")
        fstrength = k_rep * (dist_interaction - l0)
        coll = True
        
    elif (abs(dist_interaction) > l1):
        #print("ATTRACT")
        
        fstrength = k_att * (dist_interaction - l1)
        
    #print(dif * fstrength)
        
    return (dif * fstrength), coll


def attDRA(dist_interaction, dif, args):
      
    
    
    fstrength = 0
    rep_steepness=-1
    rep_range = 1
    

    coll = False
    
    fstrength = (2 * (0.5*(np.tanh(rep_steepness*(dist_interaction-rep_range))+1.0))) - 1

        
    return (dif * -fstrength), coll


def col(dist_interaction, dif, args):

    fstrength = 0
    preySize = 2*args.prey_size +2*args.delta_time

    col = False
    #distance regulation = repulsion + attraction
    
    
    
    
    if (abs(dist_interaction) < preySize):
        fstrength = args.repulsion_col * (dist_interaction - preySize)
        col = True

        
    return (dif * fstrength), col
    
    
    

def col2(pos_col, vel_col, vel_self, dist, args):

    fstrength = 0
    preySize = 2*args.prey_size+1*args.delta_time
    pos_pred = pos_col
    vel_pred = vel_col
    dist_pred = dist
    command_flee = np.zeros(2)
    command_flee_pred = command_flee
    col_pos = [0,0]

    col = False
    #distance regulation = repulsion + attraction
    if (abs(dist_pred) < preySize):
        
        col_pos_all = get_intersections(0, 0, preySize/2, pos_pred[0], pos_pred[1], preySize/2, dist)
        #print(col_pos_all[2])
        col_pos[0] = (col_pos_all[0]+col_pos_all[2])/2
        col_pos[1] = (col_pos_all[1]+col_pos_all[3])/2
        #print(col_pos)
        #print("dist", dist)
        
        front_self = np.array(vel_self)*args.prey_size
        front_other = np.array(pos_col)+np.array(vel_col)*args.prey_size
        

        dist_col_self = np.linalg.norm([front_self[0]-col_pos[0],front_self[1]-col_pos[1]])
        dist_col_other = np.linalg.norm([front_other[0]-col_pos[0],front_other[1]-col_pos[1]])
        #print(dist_col_self, dist_col_other)
        
        #phi_self = np.arctan2(vel_self[1],vel_self[0])
        #phi_other = np.arctan2(vel_col[1],vel_col[0])
        
        #phi_self_col = np.arctan2(col_pos[1],col_pos[0])
        #phi_other_col = np.arctan2(col_pos[1]-pos_col[1],col_pos[0]-pos_col[0])
        
        
        
        if abs(dist_col_self) < abs(dist_col_other):
            command_flee = 0
            col = "front"
            
        else:
            command_flee = vel_col
            col = "back"
            

        fstrength = args.repulsion_col * (dist_pred - preySize)
        

        
    return ( command_flee * fstrength), col



def flock_pred(pos_self: np.ndarray, pos: np.ndarray, vel_self: np.ndarray, vel: np.ndarray, dist_pred,  args):

    coll = False
    dt = args.delta_time
    rep_strength = args.repulsion_pred
    
    phi = math.atan2(vel_self[1],vel_self[0])
    
    #remove preds


    
    
    vel_dif = vel

    vel_dif[:,0] -= vel_self[0]
    vel_dif[:,1] -= vel_self[1]

    f_alg = np.sum(vel_dif, axis = 0)
    
    pos_dif = pos
    
    
    
    #pos_dif[:,0] -= pos_self[0]
    #pos_dif[:,1] -= pos_self[1]

    f_rep = np.zeros(2)
    
    for i in range(len(pos)):
        dist = dist_pred[i]
        pos_dif[i] = pos[i] / dist
        f_repChange, coll = att_pred(dist, np.array(pos_dif[i]), args)
        f_rep += f_repChange
        
    if len(pos) == 0:
        f_rep = rep_strength * f_rep * 0
    else:
    
        f_rep = rep_strength * f_rep / len(pos)

    
    force = f_rep
    
    
    
    

    return force



def att_pred(dist_interaction, dif, args):
      
    l0 = args.repradius_pred


    coll = False


    fstrength = np.exp(-0.5*pow(dist_interaction,2)/pow(l0,2))
        
        
    return (dif * -fstrength), coll
    
    
    
def get_intersections(x0, y0, r0, x1, y1, r1, dist):


    d=dist


    if d > (r0 + r1):
        return None

    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return (x3, y3, x4, y4)
    
def col_slide(pos_col, vel_col, dist,  args):
    #uses flee like behavior

    fstrength = 0
    preySize = 2*args.prey_size #+2*args.delta_time
    pos_pred = pos_col
    vel_pred = vel_col
    dist_pred = dist
    command_flee = np.zeros(2)
    command_flee_pred = command_flee
    
    
    


    col = False
    #distance regulation = repulsion + attraction
    if (abs(dist_pred) < preySize):
        
        
        
        
        col_pos = get_intersections(0, 0, preySize, pos_pred[0], pos_pred[1], preySize)
        col_pos = col_pos[:2]
        print(col_pos)


        pred_hunt = True #disable for orthog. vector

        ort_pv = [0,0]
        ort_pv[0] = -vel_pred[1]
        ort_pv[1] = vel_pred[0]

        
        r_pi = np.array(pos_pred)



        r_ip = [0,0]

        side = 1 # if 1-> prey is right, if -1-> prey is left (?)


        ### Calc relative distance vector and corresponding unit vector
        r_ip = pos_pred
        ### Calc if prey is left or right from pred
        ### (project r_ip on unit vector right from u of pred  -> [u[1], -u[0]])
        if ((r_ip[0] * vel_pred[1] - r_ip[1] * vel_pred[0]) > 0):
            side = -1
        ### change fleeing direction
        
        
        fleeang = math.pi / 2


        ang = math.atan2(r_ip[1], r_ip[0]) - side * fleeang
        #print(ang, side)

        ang = math.fmod(ang, 2 * math.pi)
        ang = math.fmod(2*math.pi+ang, 2 * math.pi)
        
        

        if (dist_pred > 0.0):

            command_flee[0] = np.cos(ang)
            command_flee[1] = np.sin(ang)

        steepness = -1
        x0 = 10
        
        fleestr = 0
        
        if (dist_pred < x0):
            fleestr = 0.5*(np.tanh(steepness*(dist_pred-x0))+1.0)
        command_flee_pred = command_flee_pred - command_flee * fleestr

    
        if abs(command_flee_pred[0])+abs(command_flee_pred[1]) != 0:
            command_flee_pred = command_flee_pred/abs(np.linalg.norm(command_flee_pred))
        #print("flee", command_flee_pred)

        fstrength = args.repulsion_col * (dist_pred - preySize)
        col = True
        #print("test")

        
    return ( command_flee_pred * fstrength), col



def avoid_col(pos_other, vel_self, vel_other, args):
    dt = args.delta_time
    ang = math.atan2(vel_self[1], vel_self[0])
    ang = math.fmod(ang, 2 * math.pi)
    ang = math.fmod(2*math.pi+ang, 2 * math.pi)
    
    pos_other = np.array(pos_other) + np.array(vel_other)*dt

    pos_self = np.array([0,0])
    #print(pos_other)

    col = True

    pos_new = [0,0]
    for step in np.linspace(0.02, 2, 100):
        #if step >= 2*dt:
            #print("STEP", step)
        for angle in np.linspace(0,2*math.pi, 100):
            #print("ANGLE",angle)
            ang_add = ang+angle
            ang_red = ang-angle

            pos_new[0] = np.cos(ang_add)*step
            pos_new[1] = np.sin(ang_add)*step

            dist = np.linalg.norm(np.array(pos_new)-np.array(pos_other))

            if dist >= 0.5:
                pos_new = np.array(pos_new)/step
                #print("pos_new",pos_new)
                return(pos_new)

            pos_new[0] = np.cos(ang_red)*step
            pos_new[1] = np.sin(ang_red)*step

            dist = np.linalg.norm(np.array(pos_new)-np.array(pos_other))

            #print("pos_new",pos_new)
            if dist >= 2*args.prey_size:
                pos_new = np.array(pos_new)/step
                #print("pos_new",pos_new)
                return(pos_new)
        
        
def avoid_col_mult(pos, vel_self, vel, front_idx, args):
    dt = args.delta_time
    ang = math.atan2(vel_self[1], vel_self[0])
    ang = math.fmod(ang, 2 * math.pi)
    ang = math.fmod(2*math.pi+ang, 2 * math.pi)
    
    nFront = len(front_idx)
    
    pos_other = np.zeros((2,nFront))
    
    for i in range(nFront):
        
        pos_other[:,i] = np.array(pos[front_idx[i]]) + np.array(vel[front_idx[i]])*dt

    pos_self = np.array([0,0])
    #print(pos_other)

    col = True

    pos_new = [0,0]
    for step in np.linspace(dt, 2, int(2/dt)):
        #use to check for long loops
        #if step > dt*2:
            #print(step)
        for angle in np.linspace(0,2*math.pi, 50):
            
            sucAvoid = 0
            ang_add = ang+angle
            ang_red = ang-angle

            pos_new[0] = np.cos(ang_add)*step
            pos_new[1] = np.sin(ang_add)*step
            
            for i in range(nFront):
                dist = np.linalg.norm(np.array(pos_new)-np.array(pos_other[:,i]))
                #print(i,dist)

                if dist >= 2*args.prey_size:
                    sucAvoid += 1
                    #print(sucAvoid)
            if sucAvoid >= nFront:
                
                pos_new = np.array(pos_new)/step
                #print(vel_self,ang, angle)
                #print(pos_new, pos_other[:,0], vel[front_idx[0]])
                
                return(pos_new)
            
            sucAvoid = 0
            pos_new[0] = np.cos(ang_red)*step
            pos_new[1] = np.sin(ang_red)*step

            for i in range(nFront):
                dist = np.linalg.norm(np.array(pos_new)-np.array(pos_other[:,i]))
                #print(i,dist)

                if dist >= 2*args.prey_size:
                    sucAvoid += 1
                    
                    
            if sucAvoid >= nFront:
                
                pos_new = np.array(pos_new)/step
                #print(vel_self,ang, angle)
                #print(pos_new, pos_other[:,0], vel[front_idx[0]])
                
                
                return(pos_new)
