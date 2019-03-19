from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
import pdb
from scipy.stats.distributions import chi2

def make_positive_definite(P):
    eigs = np.linalg.eigvals(P)
    if(np.all( eigs > 0)):
        return P
    else:
        offset = 1e-6 - min(0, eigs.min())

    return (P + offset * np.eye(P.shape[0]))

def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''
    vc = u[0] / ( 1- np.tan(u[1]) * vehicle_params['H']/vehicle_params['L'] )
    motion = np.zeros([3], np.float32)
    motion[0] =vc * np.cos(ekf_state['x'][2]) \
    -  (vc / vehicle_params['L'])*np.tan(u[1])*(vehicle_params['a']*np.sin(ekf_state['x'][2]) \
        + vehicle_params['b']*np.cos(ekf_state['x'][2]) ) 
    motion[1] =vc * np.sin(ekf_state['x'][2]) \
    +  (vc / vehicle_params['L'])*np.tan(u[1])*(vehicle_params['a']*np.cos(ekf_state['x'][2]) \
        - vehicle_params['b']*np.sin(ekf_state['x'][2]) ) 
    motion[2] =  (vc / vehicle_params['L'])*np.tan(u[1])
    motion = motion * dt 

    G = np.zeros((3,3), dtype = np.float32)
    G[0,2] =  - vc*np.sin(ekf_state['x'][2]) \
    -  (vc / vehicle_params['L'])*np.tan(u[1])*(vehicle_params['a']*np.cos(ekf_state['x'][2]) \
        - vehicle_params['b']*np.sin(ekf_state['x'][2]) ) 

    G[1,2] =  vc*np.cos(ekf_state['x'][2]) \
    +  (vc / vehicle_params['L'])*np.tan(u[1])*(- vehicle_params['a']*np.sin(ekf_state['x'][2]) \
        - vehicle_params['b']*np.cos(ekf_state['x'][2]) ) 

    G = G * dt
    G = G + np.eye(3)

    return motion, G

def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u 
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''
    # R - process noise
    R = np.zeros([3,3])
    R[0,0] = sigmas['xy']*sigmas['xy']
    R[1,1] = sigmas['xy']*sigmas['xy']
    R[2,2] = sigmas['phi']*sigmas['phi']
    f, G = motion_model(u, dt, ekf_state, vehicle_params)

    F = np.zeros((3, ekf_state['x'].size))
    F[0:3,0:3] = np.eye(3)

    G = G - np.eye(3)
    G =np.eye(ekf_state['x'].size) + np.matmul(np.matmul(F.T,G),(F))
    ekf_state['x']=  ekf_state['x'] + np.matmul(F.T,f)
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    ekf_state['P']= np.matmul(np.matmul(G,ekf_state['P']), G.T) + np.matmul(np.matmul(F.T,R), F) 
    ekf_state['P']= slam_utils.make_symmetric(ekf_state['P'])

    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    # Q = np.zeros([2,2])
    # Q[0,0] = sigmas['gps']*sigmas['gps']
    # Q[1,1] = sigmas['gps']*sigmas['gps']

    # H = np.array( [ [1,0,0],[0,1,0]], np.float32)
    # S = np.matmul(np.matmul(H,ekf_state['P'][0:3, 0:3]), H.T)+ Q

    # residual = gps - np.matmul(H , ekf_state['x'][0:3]) 
    # mahalanobis_dist = np.matmul(np.matmul(residual.T, np.linalg.inv(S)),residual)

    # #chi^2 test to throw out very unlikely measurements
    # if mahalanobis_dist < chi2.ppf(0.999, df=2):

    #     Kg = np.matmul(np.matmul(ekf_state['P'][0:3, 0:3],H.T),np.linalg.inv(S))
    #     ekf_state['x'][0:3]=  ekf_state['x'][0:3]+ np.matmul(Kg,residual)
    #     ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2]) 
    #     ekf_state['P'][0:3, 0:3]=  np.matmul((np.eye(3) -  np.matmul(Kg,H)),ekf_state['P'][0:3, 0:3]) 
    #     ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    
    
    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''
    zhat = np.zeros(2, dtype = np.float32)
    xv = float(ekf_state['x'][0])
    yv = float(ekf_state['x'][1])
    xl = float(ekf_state['x'][3+2*landmark_id])
    yl = float(ekf_state['x'][3+2*landmark_id+1])

    #range
    zhat[0] = np.sqrt(np.square(xl - xv) +np.square(yl - yv) )
    #bearing
    zhat[1] = slam_utils.clamp_angle(np.arctan2(yl - yv, xl - xv) - ekf_state['x'][2]) #+ np.pi/2 

    H= np.zeros([2, ekf_state['x'].shape[0]])

    H[0,0] =  (xv - xl) / zhat[0]
    H[0,1] = (yv - yl) / zhat[0]
    H[0,2] =  0
    H[0,3+2*landmark_id] =  - H[0,0]
    H[0,4+2*landmark_id] = - H[0,1]

    temp = 1.0/ ( 1 + np.square((yl-yv)/(xl-xv)) )

    H[1,0] =  (yl-yv)/np.square(zhat[0])
    H[1,1] = (xv-xl)/np.square(zhat[0])
    H[1,2] =  -1
    H[1,3+2*landmark_id] =  - H[1,0]
    H[1,4+2*landmark_id] = - H[1,1]

    return zhat, H

def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''


    xv = ekf_state['x'][0]
    yv = ekf_state['x'][1]
    phi = ekf_state['x'][2]
    xl = tree[0]*np.cos(phi+tree[1]) + xv
    yl = tree[0]*np.sin(phi+tree[1]) + yv

    ekf_state['x']= np.append(ekf_state['x'], [xl, yl])
    dim = ekf_state['x'].size

    temp2 = np.zeros([dim, dim])
    temp2[0:dim-2, 0:dim-2] = ekf_state['P']
    temp2[dim-2, dim-2] = 1
    temp2[dim-1, dim-1] = 1
    ekf_state['P'] = temp2
    ekf_state['num_landmarks'] += 1

    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''
    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    #0-----associate/match-----[alpha]-----ambiguous-----[beta]-----new_landmark-----
    #pdb.set_trace()
    alpha = chi2.ppf(0.95, df=2)
    beta = chi2.ppf(0.999, df=2)

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    measurements = np.array(measurements)[:,0:2] 
    zhat = np.zeros([ekf_state["num_landmarks"], 2])
    S = np.zeros([ ekf_state["num_landmarks"], 2, 2])
    Q = np.diag(np.array([sigmas['range']*sigmas['range'], sigmas['bearing']*sigmas['bearing']]))
    for j in range(ekf_state["num_landmarks"]):
        zhat[j], H = laser_measurement_model(ekf_state,  j)
        S[j] = np.matmul(np.matmul(H,ekf_state['P']),H.T) + Q.T

    M = alpha*np.ones((measurements.shape[0], ekf_state["num_landmarks"] + measurements.shape[0]))
    for i in range(measurements.shape[0]):
        residuals = measurements[i] - zhat
        for j in range(ekf_state["num_landmarks"]):
            mahalanobis_dist= np.matmul(residuals[j], np.matmul( np.linalg.inv(S[j]), residuals[j].T))
            M[i,j] = mahalanobis_dist

    matches = slam_utils.solve_cost_matrix_heuristic(np.copy(M))   
    matches.sort() 
    assoc=list(range(measurements.shape[0]))    
    for k in range(measurements.shape[0]):
        if(matches[k][1]>=ekf_state['num_landmarks']):
            if(np.amin(M[k,0:ekf_state['num_landmarks']])>beta): #new landmark
                assoc[matches[k][0]]=  -1
            else:  #ambiguous
                assoc[matches[k][0]]=  -2
        else:   #matched
            assoc[matches[k][0]] =  matches[k][1]
    return assoc

def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    # Perform a measurement update of the EKF state given a set of tree measurements.

    # trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    # assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    # ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    # in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    # it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''
    assoc = np.array(assoc)
    measurements = np.array(trees)[:,0:2]

    good_assoc = assoc[assoc>-1]
    good_measurements = measurements[assoc>-1]
    Q = np.diag(np.array([sigmas['range']*sigmas['range'], sigmas['bearing']*sigmas['bearing']]))
    Kg = np.zeros([ekf_state['x'].shape[0], 2*good_assoc.size])
    residuals= np.zeros((2*good_measurements.shape[0]))
    H= np.zeros([2*good_assoc.size, ekf_state['x'].shape[0]])
    for i in range(good_assoc.size):
        zhat, H[i:i+2,:] = laser_measurement_model(ekf_state, good_assoc[i])
        Kg[:, i:i+2] = np.matmul(ekf_state['P'] , np.matmul(H[i:i+2,:].T, \
            np.linalg.inv(np.matmul(H[i:i+2,:], np.matmul(ekf_state['P'], H[i:i+2,:].T))+ Q)))
        residuals[i:i+2] = good_measurements[i] - zhat


    ekf_state['x'] = ekf_state['x'] + np.matmul(Kg, residuals)
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2]) 
    ekf_state['P'] = np.matmul((np.eye(ekf_state['P'].shape[0]) - np.matmul(Kg,H)), ekf_state['P'])
    ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    ekf_state['P'] = make_positive_definite(ekf_state['P'])


    new_measurements = measurements[assoc==-1]
    for i in range(new_measurements.shape[0]):
        ekf_state = initialize_landmark(ekf_state, new_measurements[i])

    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))
            print('num_landmarks', ekf_state['num_landmarks'])

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": True

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()
