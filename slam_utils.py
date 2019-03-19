
import numpy as np

try:
    import PyQt5
    import pyqtgraph as pg
    can_plot = True
except ImportError:
    can_plot = False

def read_data_file(file_name):
    with open(file_name, "r") as f:
        raw_data = f.readlines()

    data = [ [float(x) for x in line.strip().split(',')] for line in raw_data ]

    return np.array(data)


def tree_to_global_xy(trees, ekf_state):
    if len(trees) == 0:
        return []

    trees = np.array(trees) # rows are [range, bearing, diameter]
    phi = ekf_state["x"][2]
    mu = ekf_state["x"][0:2]

    return np.reshape(mu, (2,1)) + np.vstack(( trees[:,0]*np.cos(phi+trees[:,1]), 
                                                trees[:,0]*np.sin(phi+trees[:,1])))

def plot_tree_measurements(trees, assoc, ekf_state, plot):
    if len(trees) == 0:
        return

    G_trees = tree_to_global_xy(trees, ekf_state)
    mu = ekf_state["x"][0:2]

    t_list = [t.tolist() for t in G_trees.T]

    if "lasers" not in plot:
        plot["lasers"] = []
        plot["laser_in_axis"] = []

    for i in range(len(assoc)):
        data = np.vstack((mu, t_list[i]))

        if assoc[i] >= 0:
            color = 'g'
        elif assoc[i] == -2:
            color = 'r'
        else:
            color = 'b'

        if i >= len(plot["lasers"]):
            new_item = plot["axis"].plot(data, pen=pg.mkPen(color, width=2))
            plot["lasers"].append(new_item)
            plot["laser_in_axis"].append(True)
        else:
            plot["lasers"][i].setData(data, pen=pg.mkPen(color, width=2))

        if not plot["laser_in_axis"][i]:
            plot["axis"].addItem(plot["lasers"][i])
            plot["laser_in_axis"][i] = True

    for i in range(len(assoc), len(plot["lasers"])):
        if plot["laser_in_axis"][i]:
            plot["axis"].removeItem(plot["lasers"][i])
            plot["laser_in_axis"][i] = False

def plot_trajectory(traj, plot):
    if np.prod(traj.shape) > 3:
        if "trajectory" not in plot:
            plot["trajectory"] = plot["axis"].plot()

        plot["trajectory"].setData(traj[:,:2], pen='k')

def plot_map(ekf_state, plot, params):
    if "map" not in plot:
        plot["map"] = plot["axis"].plot()

    lms = np.reshape(ekf_state["x"][3:], (-1, 2))
    plot["map"].setData(lms, pen=None, symbol="+", symbolPen='g', symbolSize=13)

    if params["plot_map_covariances"]:
        if "map_covariances" not in plot:
            plot["map_covariances"] = []

        for i in range(ekf_state["num_landmarks"]):
            idx = 3 + 2*i
            P = ekf_state["P"][idx:idx+2, idx:idx+2]

            circ = get_covariance_ellipse_points(ekf_state["x"][idx:idx+2], P)

            if i >= len(plot["map_covariances"]):
                plot["map_covariances"].append(plot["axis"].plot())

            plot["map_covariances"][i].setData(circ, pen='b')
        

def get_covariance_ellipse_points(mu, P, base_circ=[]):

    if len(base_circ) == 0:
        N = 20
        phi = np.linspace(0, 2*np.pi, N)
        x = np.reshape(np.cos(phi), (-1,1))
        y = np.reshape(np.sin(phi), (-1,1))
        base_circ.extend(np.hstack((x,y)).tolist())

    vals, _ = np.linalg.eigh(P)

    offset = 1e-6 - min(0, vals.min())

    G = np.linalg.cholesky(P + offset * np.eye(mu.shape[0]))

    # 3 sigma bound
    circ = 3*np.matmul(np.array(base_circ), G.T) + mu

    return circ

def convert_to_global_xy(ekf_state, scan, params):
    angles = np.array(range(361))*np.pi/360 - np.pi/2

    rb = np.vstack((scan, angles)).T
    phi = ekf_state["x"][2]
    mu = ekf_state["x"][0:2]

    rb = rb[ rb[:,0] < params["max_laser_range"], : ]

    xy = np.reshape(mu, (2,1)) + np.vstack(( rb[:,0]*np.cos(phi+rb[:,1]), 
                                                rb[:,0]*np.sin(phi+rb[:,1])))

    return xy

def plot_scan(ekf_state, scan, plot, params):
    if "scan" not in plot:
        plot["scan"] = plot["axis"].plot()

    scan = convert_to_global_xy(ekf_state, scan, params)

    plot["scan"].setData(scan.T, pen=None, symbol="d", symbolPen='k', symbolSize=3)

def plot_robot(ekf_state, plot):
    # base triangle shape
    triangle = 1.5 * np.array([[0, 0], [-3, 1], [-3, -1], [0, 0]])

    # rotate to correct orientation
    phi = ekf_state["x"][2]
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    triangle = np.matmul(triangle, R.T)

    # translate to correct position
    triangle += ekf_state["x"][:2]

    if "robot" not in plot:
        plot["robot"] = plot["axis"].plot()

    plot["robot"].setData(triangle, pen=pg.mkPen('k', width=2))

def plot_covariance(ekf_state, plot):
    Pp = ekf_state["P"][:2,:2]

    circ = get_covariance_ellipse_points(ekf_state["x"][:2], Pp)

    if "cov" not in plot:
        plot["cov"] = plot["axis"].plot()

    plot["cov"].setData(circ, pen='b')

def plot_state(ekf_state, plot, params):
    plot_map(ekf_state, plot, params)
    plot_robot(ekf_state, plot)
    plot_covariance(ekf_state, plot)

def do_plot(xhist, ekf_state, trees, scan, assoc, plot, params):
    plot_trajectory(xhist, plot)
    plot_state(ekf_state, plot, params)
    plot_tree_measurements(trees, assoc, ekf_state, plot)
    if len(scan) > 0 and params["plot_raw_laser"]:
        plot_scan(ekf_state, scan, plot, params)

    pg.QtGui.QApplication.processEvents()

def init_plot():
    if not can_plot:
        raise Exception("Unable to display graphics window due to missing PyQt or pyqtgraph library")

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    win = pg.GraphicsWindow()
    axis = win.addPlot()
    axis.setAspectLocked(True)

    plot = {
        "win": win,
        "axis": axis
    }

    plot["win"].setWindowTitle("EKF SLAM")

    return plot

def clamp_angle(theta):
    while theta >= np.pi:
        theta -= 2*np.pi

    while theta < -np.pi:
        theta += 2*np.pi

    return theta

def make_symmetric(P):
    return 0.5 * (P + P.T)

def invert_2x2_matrix(M):
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    return np.array([ [M[1,1], -M[0,1]], [-M[1,0], M[0,0]] ]) / det

def solve_cost_matrix_heuristic(M):
    n_msmts = M.shape[0]
    result = []

    ordering = np.argsort(M.min(axis=1))

    for msmt in ordering:
        match = np.argmin(M[msmt,:])
        M[:, match] = 1e8
        result.append((msmt, match))

    return result