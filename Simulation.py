import numpy as np
import pandas as pd
import os
import joblib
import contextlib
from tqdm import tqdm
from joblib import Parallel, delayed
import numba as nb
import scipy.spatial as sp
from scipy.integrate import solve_ivp
import random
import pickle
from scipy.stats import truncnorm
from warnings import filterwarnings
filterwarnings("ignore")

@contextlib.contextmanager
def tqdm_joblib(
    tqdm_object
):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

@nb.njit(nb.float64[:](nb.float64[:, :], nb.int32, nb.int32, nb.int32, nb.float64[:]))
def partials(
    positions: np.array, 
    i: int, 
    j: int, 
    k: int, 
    area: np.array
) -> np.array:
    """
    Calculate the partial derivatives required when calculating the encapsulation force (combined BM and embedding matrix contribution).
    Args:
        positions (np.array): A 2D numpy array of shape (n_bodies, 3) representing the positions of body centers.
            positions[i,:] = x,y,z positions of center of body i.
        i (int): The index of the first vertex of the simplex.
        j (int): The index of the second vertex of the simplex.
        k (int): The index of the third vertex of the simplex.
        area (np.array): A numpy array containing the area of the simplex defined by vertices i, j, and k.
    Returns:
        np.array: A numpy array representing 1/(2*area) * partial derivative of the area with respect to xi, yi, and zi.
    """
    x_i = positions[i,0] 
    y_i = positions[i,1]
    z_i = positions[i,2]
    x_j = positions[j,0] 
    y_j = positions[j,1]
    z_j = positions[j,2]
    x_k = positions[k,0] 
    y_k = positions[k,1]
    z_k = positions[k,2] 

    partial_xi =(y_j - y_k)*((x_j-x_i)*(y_k-y_i) - (x_k-x_i)*(y_j-y_i) )+ (z_k-z_j)*(-(x_j-x_i )*(z_k-z_i) + (x_k-x_i)*(z_j-z_i))
    partial_yi =(-x_j + x_k)*((x_j-x_i)*(y_k-y_i) - (x_k-x_i)*(y_j-y_i))+ (z_j - z_k)*((y_j-y_i)*(z_k-z_i) - (y_k-y_i)*(z_j-z_i))
    partial_zi =(-x_j + x_k)*((x_j-x_i)*(z_k-z_i) - (x_k-x_i)*(z_j-z_i)) + (y_k-y_j)*((y_j-y_i)*(z_k-z_i) - (y_k-y_i)*(z_j-z_i))
    
    return 1/(2*area)* np.array([partial_xi, partial_yi, partial_zi])

@nb.njit(nb.float64[:, :](nb.int32[:, :], nb.int32))
def get_neighbours(
    simplices: np.array, 
    N_bodies: int
) -> np.array:
    """
    Determine the neighbors for each body using simplices calculated from a Delaunay triangulation.
    This function creates a matrix where each row corresponds to a body and each column contains the indices of its neighbors.
    The matrix is filled with NaNs initially, and valid neighbor indices are populated based on the simplices provided.

    Args:
        simplices (np.array): A 2D numpy array of simplices of shape (N_simplices, 4), calculated as simplicies = sp.Delaunay(positions).simplices.
            Indices of the points forming the simplices in the triangulation.
        N_bodies (int): The total number of bodies in the system.
            N_bodies = N_cells + 1 (lumen)

    Returns:
        neighbours (np.array): A 2D numpy array where each row represents a body and each entry in the row is an index of a neighbor.
            The array dimensions are (N_bodies, N_bodies).
            neighbours[i,:] = indices of bodies neighbouring body i
            Neighbours of each body are filled left to right, so neighbours[;,j] = np.NaN indicates no further neighbours
    """
    neighbours = np.empty((N_bodies, N_bodies))
    neighbours[:] = np.nan
    for i in range(np.shape(simplices)[0]):
        simplex = simplices[i]
        for j in range(4):
            for k in range(4):
                if j != k and simplex[j] not in neighbours[simplex[k]]:
                    for m in range(N_bodies):
                        if np.isnan(neighbours[simplex[k], m]):
                            neighbours[simplex[k], m] = simplex[j]
                            break
    return neighbours

@nb.njit(nb.types.UniTuple(nb.float64[:, :], 2)(nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64, nb.float64[:, :], nb.int32, nb.int32[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.int32, nb.float64, nb.float64))
def calculate_force(
    positions: np.array, 
    neighbours: np.array, 
    ages: np.array, 
    lifetimes: np.array, 
    alpha: float, 
    morse_force_matrix: np.array, 
    N_bodies: int, 
    simplices: np.array, 
    areas: np.array, 
    equations: np.array, 
    bm_pressure_force_matrix: np.array, 
    bm_area_force_matrix: np.array, 
    beta: float, 
    P_star: float, 
    A_eq_star: float
) -> np.array:
    """
    Args:
        positions (np.array): A 2D numpy array of shape (n_bodies, 3) representing the positions of body centers.
            positions[i,:] = x,y,z positions of center of body i.
        neighbours (np.array): A 2D numpy array where each row represents a body and each entry in the row is an index of a neighbor.
            The array dimensions are (N_bodies, N_bodies).
            neighbours[i,:] = indices of bodies neighbouring body i
            Neighbours of each body are filled left to right, so neighbours[;,j] = np.NaN indicates no further neighbours
        ages (np.array): Array with dimensions (N_bodies, 1) where the entry ages[i,0] represents the age of body i
        lifetimes (np.array): Array with dimensions (N_bodies, 1) where the entry lifetimes[i,0] represents the lifetime of body i
            (i.e. the age at which it will divide into two daughters)
        N_bodies (int): The total number of bodies in the system.
            N_bodies = N_cells + 1 (lumen)
        simplices (np.array): A 2D numpy array of simplices of shape (N_simplices, 4), calculated as simplicies = sp.Delaunay(positions).simplices.
        areas: np.zeros([n simplices,1]), empty array to be filled with the areas of the convex hull simplices
        equations (np.array): ndarray of double, shape (nfacet, ndim+1) i.e. (n simplices, 4); [normal, offset] forming the hyperplane equation of the facet, returned from sp.convex_hull.equations
        bm_pressure_force_matrix (np.array): Zeros array np.zeros((self.N_bodies, 3)) that will be filled with the forces experienced by simulated bodies due to the preferred simplex areas (BM contribution)
            bm_pressure_force_matrix[i,:] = x,y,z pressure forces experienced by body i.
        bm_area_force_matrix (np.array): Zeros array np.zeros((self.N_bodies, 3)) that will be filled with the forces experienced by simulated bodies due to the preferred simplex areas (encapsualation force due to BM and embedding matrix )
            bm_area_force_matrix[i,:] = x,y,z encapsulation forces experienced by body i.
        beta: (float) nondimensional force parameter
        P_star: (float) nondimensional pressure parameter
        A_eq_star: (float) nondimensional preferred area parameter
    Returns:
        force_matrix (np.array): Array of size (N_bodies, 3) containing the combined forces experienced by simulated bodies (F_(neighbour/Morse) + F_(pressure) + F_(encapsulation))
            force_matrix[i,:] = x,y,z forces experienced by body i.
        areas (np.array): Array of size (n simplices ,1) representing the areas of the convex hull simplices

    """
    for ii in range(N_bodies):
        for jj in neighbours[ii]:
            if np.isnan(jj) or ii <= jj:
                continue
            else:   
                jj = int(jj)
                r_ij_star = positions[jj,:] - positions[ii,:]
                r_mag = np.sqrt(np.dot(r_ij_star,r_ij_star))
                J = 2 + ((np.cbrt(2)-1)* ((ages[ii]/lifetimes[ii])+(ages[jj]/lifetimes[jj]))) ###################POSSIBLY SPEED THIS UP BY CALCULATING ALL JS AT ONCE IN MATRIX
                F_morse_mag = -2*(np.exp(alpha*(r_ij_star - J)) - np.exp(2*alpha*(r_ij_star-J)))
                F_morse = F_morse_mag *  (r_ij_star/r_mag)
                morse_force_matrix[ii,:] -= F_morse
                morse_force_matrix[jj,:] += F_morse

    # CALCULATE FORCE DUE TO PREFERRED SIMPLEX AREAS (BM CONTRIBUTION)
    for count, face in enumerate(simplices): # face has the form [i j k], where i j and k are the indices of the cells that define one simplex
        areas[count] = 0.5* np.linalg.norm(np.cross(positions[face[1]]-positions[face[0]],positions[face[2]]-positions[face[0]])) #area of each face
        for index in range(3):
            bm_pressure_force_matrix[face[(index%3)]] += (beta/3)* P_star * areas[count] * equations[count,0:3]
            bm_area_force_matrix[face[index]] -= beta * (areas[count]-A_eq_star) * partials(positions, face[index], face[(index+1)%3], face[(index+2)%3], areas[count])     

    force_matrix = bm_pressure_force_matrix + bm_area_force_matrix + morse_force_matrix

    return force_matrix, areas

class Simulation:

    def __init__(
        self,
        N_bodies=70,
        r_min=1,
        mean_lifetime=5,
        delta_t_max=0.001,
        timestep_reset=1000

    ):
        """
        Initializes an instance of the simulation class.
        Args:
            N_bodies (int): The total number of bodies in the system.
                N_bodies = N_cells + 1 (lumen)
            r_min (int, optional): Radius of a daughter cell that has age=0 (i.e. minimum radius). Defaults to 1.
            mean_lifetime (int, optional): Mean lifetime. Cell lifetimes are normally distributed around this with variance mean_lifetime/100. Defaults to 5.
            delta_t_max (float, optional): Maximum delta t for the ODE solver. Defaults to 0.001.
        """
        self.all_forces = []
        self.all_volumes = []
        self.all_positions = []
        self.all_ages = []
        self.all_lifetimes = []
        self.all_preferred_areas = []
        self.all_t_values = [0, 0]

        self.current_areas = np.inf
        self.last_event_time = 0

        self.N_bodies = N_bodies
        self.r_min = r_min
        self.mean_lifetime = mean_lifetime
        self.t_max = self.mean_lifetime * 2
        self.delta_t_max = delta_t_max

        self.lifetime_std = self.mean_lifetime/20
        self.lifetimes = truncnorm.rvs((0 - self.mean_lifetime) / self.lifetime_std, (np.inf - self.mean_lifetime) / self.lifetime_std, loc=self.mean_lifetime, scale=self.lifetime_std, size=(N_bodies, 1))
        self.ages = self.lifetimes * np.random.rand(self.N_bodies, 1)
        # initial positions calculated in separate notebook using a cost function to enable packing of spherical cells into a spherical (acinar) initial configuration
        self.positions = np.array([[ 7.85644822e-02,  2.03704701e+00, -1.44438203e-01],
       [-1.19378674e+00,  3.22591201e+00,  5.76453721e-01],
       [ 6.07914478e-02,  3.35680897e+00, -9.12230905e-01],
       [ 6.03164880e-01,  3.28445062e+00,  1.10986975e+00],
       [-1.60322975e+00,  2.69071956e+00, -1.08745096e+00],
       [ 1.64787572e+00,  2.87473262e+00, -1.61552999e-01],
       [-6.70269294e-01,  2.45537519e+00,  1.89254751e+00],
       [-2.72011301e-01,  2.44998835e+00, -2.54012604e+00],
       [ 2.22874634e+00,  2.24244031e+00,  1.31910310e+00],
       [-1.15906660e+00,  1.43449001e+00,  2.74366991e-01],
       [ 7.59850788e-01,  5.82603781e-01, -1.23170699e+00],
       [ 8.85505822e-01,  2.01402469e+00,  2.66701553e+00],
       [-2.08989122e+00,  1.70065410e+00, -2.35206526e+00],
       [ 2.97166583e+00,  1.83403082e+00, -5.62263581e-01],
       [-2.08737601e+00,  1.78023017e+00,  1.96467018e+00],
       [-1.87939149e-01,  1.40018277e+00, -1.67863203e+00],
       [ 6.84049565e-01,  1.47101553e+00,  1.16646047e+00],
       [-2.64561600e+00,  2.22992472e+00,  3.74648500e-01],
       [ 1.46436038e+00,  2.32726520e+00, -1.72735335e+00],
       [-4.29717969e-01,  6.10714993e-01,  1.77947830e+00],
       [-1.35893724e+00,  7.98574441e-01, -1.17746785e+00],
       [ 1.68272669e+00,  1.00929458e+00, -1.93722856e-01],
       [-1.98696836e+00,  3.97207802e-01,  7.71184101e-01],
       [ 1.07894215e+00,  1.13399313e+00, -3.04749244e+00],
       [ 2.29230372e+00,  7.38796460e-01,  2.63679873e+00],
       [-3.00801742e+00,  1.20441208e+00, -8.28983840e-01],
       [ 2.57911555e+00,  8.94544792e-01, -2.00294665e+00],
       [-7.88513147e-01,  1.19450396e+00,  3.23641456e+00],
       [-9.64935871e-01,  6.96213410e-01, -3.16187203e+00],
       [ 3.31544614e+00,  8.62114254e-01,  1.01632104e+00],
       [-3.33740516e+00,  5.64214959e-01,  1.05028914e+00],
       [ 1.50335959e+00, -3.81320936e-01, -1.61617624e+00],
       [ 7.20071155e-01,  2.50186071e-01,  2.93827879e+00],
       [-2.55361690e+00, -9.27942934e-02, -2.22515178e+00],
       [ 3.07105709e+00,  3.91797744e-04, -2.54305638e-01],
       [-2.15261178e+00,  9.31574067e-02,  2.61527138e+00],
       [ 2.27600331e-01, -2.51330706e-01, -3.01703023e+00],
       [ 1.75003279e+00,  3.80776395e-01,  1.34687845e+00],
       [-3.46449631e+00, -5.63937399e-01, -4.86917875e-01],
       [ 3.10403812e+00, -8.61465699e-01, -1.54889947e+00],
       [-4.32543139e-01, -6.95098928e-01,  3.27956949e+00],
       [-1.31049913e+00, -1.19497892e+00, -3.06259127e+00],
       [ 2.87384418e+00, -8.93590223e-01,  1.55172534e+00],
       [-2.83119781e+00, -1.20425175e+00,  1.31250374e+00],
       [ 1.82735225e+00, -7.37575890e-01, -2.97817125e+00],
       [ 1.56691344e+00, -1.13380235e+00,  2.82923637e+00],
       [-2.08753284e+00, -3.96051342e-01, -4.33969476e-01],
       [ 1.68966972e+00, -1.00919628e+00, -8.66784592e-02],
       [-1.14411370e+00, -7.98926639e-01,  1.38323015e+00],
       [-7.17759803e-01, -6.09653144e-01, -1.68441756e+00],
       [ 1.72841012e+00, -2.32721421e+00,  1.46312565e+00],
       [-2.67271045e+00, -2.23079942e+00,  6.57094533e-02],
       [ 4.80089768e-01, -1.46887089e+00, -1.26295990e+00],
       [ 9.05599990e-02, -1.40079411e+00,  1.68416618e+00],
       [-2.38314358e+00, -1.78146754e+00, -1.59476974e+00],
       [ 3.02420845e+00, -1.83412283e+00,  6.60391737e-02],
       [-1.67505407e+00, -1.70115935e+00,  2.66386951e+00],
       [ 4.35057728e-01, -2.01554587e+00, -2.77623452e+00],
       [ 9.51428673e-01, -5.86147871e-01,  1.08643658e+00],
       [-1.18735884e+00, -1.43278945e+00, -7.86240721e-02],
       [ 1.98217651e+00, -2.24304165e+00, -1.66833248e+00],
       [ 1.48905849e-01, -2.44934834e+00,  2.55049384e+00],
       [-9.74334420e-01, -2.45725323e+00, -1.75776480e+00],
       [ 1.65259255e+00, -2.87368878e+00, -1.11134113e-01],
       [-1.40209490e+00, -2.68943940e+00,  1.33646979e+00],
       [ 4.12803852e-01, -3.28633475e+00, -1.19454331e+00],
       [ 2.10620580e-01, -3.35612329e+00,  8.92019805e-01],
       [-1.27265202e+00, -3.22766796e+00, -3.71362205e-01],
       [ 1.00523611e-01, -2.03318446e+00,  1.32008783e-01],
       [-6.35964090e-02, -3.42730202e-03,  4.28524890e-03]])

        self.force_matrix = np.zeros((self.N_bodies, 3))
        self.hull = sp.ConvexHull(self.positions)
        self.radii = np.zeros_like(self.ages)
        self.volumes = np.zeros_like(self.ages)

        self.calculate_radius_from_age()
        self.calculate_volume_from_radius()
        self.lumen_volume = self.volumes[-1]
        self.lumen_radius = self.radii[-1]
        self.all_lumen_volumes = []

        self.events.terminal = True
        self.event_trigger_reason = None
        self.timestep_reset = timestep_reset
        self.reset_count = 0

    def calculate_radius_from_age(
        self
    ):
        """
        Calculates the radii of the N_bodies based on their age using the following formula:
            radius = r_min * (1 + ((np.cbrt(2)-1)  * (age / lifetime)))
        Radius increases linearly with age from a minimum, r_min, at age=0 and corresponding volume V_min
        A cell of age=mean_lifetime has a radius r_max, corresponding to a volume 2V_min
        """
        self.radii = self.r_min * (1 + ((np.cbrt(2)-1)  * (self.ages/self.lifetimes)))
        self.lumen_radius = self.radii[-1]

    
    def calculate_volume_from_radius(
        self
    ):
        """
        Calculates the volumes of the N_bodies given their current radii according to:
            volume = (4/3) * π * radius^3

        """
        self.volumes = 4/3 * np.pi * self.radii**3
        self.lumen_volume = self.volumes[-1]



    def calculate_total_forces(
        self
    ):
        """
        Calculates the total forces acting on each body.

        This function calculates the total forces acting on each body in the system. It performs the following steps:
        1. Computes the convex hull of the body positions.
        2. Constructs the Delaunay triangulation of the body positions.
        3. Determines the neighbors of each body based on the triangulation.
        4. Initializes force matrices for pressure, area (encapsulation), and Morse (neighbour) forces.
        5. Calculates the forces and areas using the provided parameters.
        6. Updates the current areas.
        """
        self.hull = sp.ConvexHull(self.positions)
        dela = sp.Delaunay(self.positions)
        self.neighbours = np.empty((self.N_bodies, self.N_bodies))
        self.neighbours[:] = np.nan
        self.neighbours = get_neighbours(dela.simplices, self.N_bodies)

        self.force_matrix = np.zeros((self.N_bodies, 3)) 
        self.bm_pressure_force_matrix = np.zeros((self.N_bodies, 3))
        self.bm_area_force_matrix = np.zeros((self.N_bodies,3))
        self.morse_force_matrix = np.zeros((self.N_bodies, 3))
        
        self.areas = np.zeros([len(self.hull.simplices),1])
        self.force_matrix, self.areas = calculate_force(
            self.positions, 
            self.neighbours, 
            self.ages, 
            self.lifetimes, 
            self.alpha, 
            self.morse_force_matrix, 
            self.N_bodies, 
            self.hull.simplices, 
            self.areas, 
            self.hull.equations, 
            self.bm_pressure_force_matrix, 
            self.bm_area_force_matrix, 
            self.beta, 
            self.P_star, 
            self.A_eq_star
        )
        self.current_areas = self.areas


    @staticmethod
    def r_dot(
        t,
        r,
        self
    ): 
        """
        Calculates the derivative of the position vector with respect to time.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
        Args:
            t (float): The current time.
            r (list): The position vector.
        Returns:
            list: The derivative of the position vector with respect to time.
        """

        self.positions = np.asarray(r).reshape(self.N_bodies,3)
        time_increment = self.all_t_values[-1]-self.all_t_values[-2]

        self.calculate_total_forces()
        current_ages = self.ages   
        self.all_t_values.append(t)
        self.A_eq_star = (self.hull.area / (self.hull.simplices.shape[0])) * self.A_eq_star_scaling

        self.ages[:-1] = current_ages[:-1] + time_increment
        self.ages[-1] = current_ages[-1]  
            
        self.all_ages.append(self.ages)
        self.all_lifetimes.append(self.lifetimes)
        round_percent = int(round((self.all_t_values[-1]/self.t_max)*50))
        drdt = self.force_matrix.flatten().tolist()

        self.all_positions.append(self.positions)
        self.all_forces.append(self.force_matrix)
        self.all_volumes.append(self.hull.volume)
        self.all_lumen_volumes.append(self.lumen_volume[0])
        self.all_preferred_areas.append(self.A_eq_star)

        return drdt

    @staticmethod
    def events( 
        t, 
        r,
        self
    ):
        """
        Determines if an event should be triggered based on certain conditions.
        Args:
            t (float): The current time.
            r (float): The current radius.
        Returns:
            int: 0 if no event should be triggered, 1 if an event should be triggered.
        """

        if self.N_bodies>= 400:
            self.event_trigger_reason = "too_many_cells"
            return 0

        if any(self.current_areas) <  self.A_eq_star:
            self.event_trigger_reason = "unphysical_area"
            return 0

        elif self.positions.shape[0] <= 3:
            self.event_trigger_reason = "unphysical_hull"
            return 0

        elif self.last_event_time != 0 and (self.all_t_values[-1] - self.last_event_time) <= self.mean_lifetime/1000:
            self.event_trigger_reason = "too_soon_event"
            return 1

        elif any(self.ages[0:-1] > self.lifetimes[0:-1]): 
            self.event_trigger_reason = "division"
            return 0
        else:
            distances = np.linalg.norm(self.positions - self.positions[-1], axis=1) 
            cells_inside = [i for i in range(self.N_bodies-1) if distances[i] <= self.radius_scaling * self.lumen_radius]
            if len(cells_inside) > 1:
                self.event_trigger_reason = "cell_joins_lumen"
                return 0
            else:
                return 1

    def cell_joins_lumen(
        self
    ):
        """
        Updates the state of the simulation when a cell joins the lumen.

        This function calculates the new state of the simulation when a cell joins the lumen. It calculates the new lumen volume,
        updates the positions, ages, and lifetimes of the cells, and recalculates the lumen radius and volume. If the number of
        cells in the lumen is less than 2, the event_trigger_reason is set to "unphysical_hull" as a convex hull can no longer be defined.
        """

        new_lumen_volume = 0
        numerator, denominator = 0, 0

        self.calculate_radius_from_age()
        self.calculate_volume_from_radius()

        distances = np.linalg.norm(self.positions - self.positions[-1], axis=1) 
        bodies_in_centre = [i for i in range(self.N_bodies) if distances[i] <= (self.radius_scaling * self.lumen_radius)] 

        surviving_exterior_cells = [i for i in range(self.N_bodies) if i not in bodies_in_centre]
        new_cell_number = len(surviving_exterior_cells)
        
        if new_cell_number >= 2:
            new_N_bodies = new_cell_number + 1
            new_positions = np.zeros((new_N_bodies, 3))
            new_ages = np.zeros((new_N_bodies, 1))
            new_lifetimes = np.zeros((new_N_bodies, 1))

            for i, body in enumerate(bodies_in_centre):
                if body != self.N_bodies-1:
                    new_lumen_volume += self.volumes[body] * self.volume_scaling
                else:
                    new_lumen_volume += self.lumen_volume
    
            for i, body in enumerate(surviving_exterior_cells):  
                numerator += self.volumes[body] * self.positions[body]
                denominator += self.volumes[body]

            for count, cell_index in enumerate(surviving_exterior_cells):
                new_positions[count] = self.positions[cell_index]
                new_ages[count] = self.ages[cell_index]
                new_lifetimes[count] = self.lifetimes[cell_index]

            self.lumen_volume = new_lumen_volume
            self.lumen_radius = (0.75 * self.lumen_volume * (1/np.pi))**(1/3)

            lumen_scale_age = self.mean_lifetime * ((self.lumen_radius/self.r_min)-1) * (1/(np.cbrt(2)-1))

            new_ages[-1] = lumen_scale_age
            new_lifetimes[-1] = self.mean_lifetime
            new_positions[-1,:] = numerator / denominator

            self.ages = new_ages
            self.lifetimes = new_lifetimes
            self.N_bodies = self.ages.shape[0]
            self.positions = new_positions
            
            self.calculate_radius_from_age()
            self.calculate_volume_from_radius()

            try:
                self.hull=sp.ConvexHull(self.positions)
                self.last_event_time = self.all_t_values[-1]
            except:
                self.event_trigger_reason = "unphysical_hull"

        else:
            self.event_trigger_reason = "unphysical_hull"

    def perform_divisions(
        self
    ): 
        """
        Perform divisions of bodies based on their ages and lifetimes.
        This method updates the positions, ages, lifetimes, and other attributes of the system
        based on the ages and lifetimes of the bodies. Cells that have exceeded their lifetimes
        each divide into two daughter cells that are displaced at random in such a way that the 
        centre of mass of the cell is conserved over divison. Both daughter cells are assigned
        independent, normally distributed lifetimes.
        """

        N_old = self.N_bodies
        old_positions = self.positions
        old_ages = self.ages

        number_dividing = (self.ages[:-1] > self.lifetimes[:-1]).sum()
        self.N_bodies += number_dividing

        new_positions = np.zeros((self.N_bodies,3))
        new_ages = np.zeros((self.N_bodies, 1))
        new_lifetimes = np.zeros((self.N_bodies, 1))
        
        j = 0 
        for i, age in enumerate(self.ages[:-1]):
            if age < self.lifetimes[i]:
                new_positions[i] = self.positions[i]
                new_ages[i] = self.ages[i]
                new_lifetimes[i] = self.lifetimes[i]
            else:
                new_ages[i] = 0
                new_lifetimes[i] =  truncnorm.rvs((0 - self.mean_lifetime) / self.lifetime_std, (np.inf - self.mean_lifetime) / self.lifetime_std, loc=self.mean_lifetime, scale=self.lifetime_std)
                new_ages[N_old + j-1] = 0
                new_lifetimes[N_old + j-1] = truncnorm.rvs((0 - self.mean_lifetime) / self.lifetime_std, (np.inf - self.mean_lifetime) / self.lifetime_std, loc=self.mean_lifetime, scale=self.lifetime_std)

                theta_rand = np.random.uniform(0,360)
                phi_rand = np.random.uniform(0,360)

                new_positions[i][0] = self.positions[i][0] + self.r_min * np.cos(phi_rand) * np.sin(theta_rand)
                new_positions[i][1] = self.positions[i][1] +  self.r_min * np.sin(phi_rand) * np.sin(theta_rand)
                new_positions[i][2] = self.positions[i][2] +  self.r_min * np.cos(theta_rand)
                new_positions[N_old + j-1][0] =  new_positions[i][0] - (2*self.r_min * np.cos(phi_rand) * np.sin(theta_rand))
                new_positions[N_old + j-1][1] = new_positions[i][1] - (2*self.r_min * np.sin(phi_rand) * np.sin(theta_rand))
                new_positions[N_old + j-1][2] = new_positions[i][2] - (2*self.r_min * np.cos(theta_rand))        
                j += 1

        new_ages[-1] = old_ages[-1]
        new_lifetimes[-1] = self.mean_lifetime
        new_positions[-1] = old_positions[-1]

        self.positions = new_positions
        self.ages = new_ages
        self.lifetimes = new_lifetimes
        self.hull = sp.ConvexHull(self.positions)
        self.last_event_time = self.all_t_values[-1]

    def execute(
        self,
        beta,
        alpha,
        P_star,
        radius_scaling,
        A_eq_star_scaling,
        volume_scaling=0.1,
        write_results=False,
        write_path="C:\\Simulation\\outputs",
        run_number=0,
        alter='all',
        max_reset_count=10
    ):
        """
        Executes the simulation with the given parameters.

        Args:
            beta (float): Nondimensional force parameter.
            alpha (float): Nondimensional parameter that governs the width of the Morse potential well
            P_star (float): Nondimensional pressure parameter.
            radius_scaling (float): The radius scaling factor. A cell with centre located within this factor of the lumen current radius will "undergo apoptosis" and be absorbed into the luminal space. f in report
            A_eq_star_scaling (float): The A_eq_star scaling factor. The preferred area of each simplex is calculated as the area of the convex hull divided by the number of simplices, multiplied by this scaling factor.
            volume_scaling (float, optional): The volume scaling factor. Defaults to 0.1. g in report
            write_results (bool, optional): Whether to write the simulation results. Defaults to False.
            write_path (str, optional): The path to write the results.
            run_number (int, optional): The run number. Defaults to 0.
            alter (str, optional): The alter parameter. Defaults to 'all'. Used as a string in output filenames
        """
        self.beta = beta
        self.alpha = alpha
        self.A_eq_star_scaling = A_eq_star_scaling
        self.P_star = P_star
        self.volume_scaling = volume_scaling
        self.radius_scaling = radius_scaling
        self.A_eq_star = (self.hull.area / (self.hull.simplices.shape[0])) * self.A_eq_star_scaling

        self.t_min = 0
        while self.t_min < self.t_max: 
            try:
                sol = solve_ivp( #https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
                    fun=self.r_dot, 
                    t_span=(self.t_min, self.t_max), 
                    y0=self.positions.flatten(), 
                    method='RK45', 
                    first_step=0.0003, 
                    max_step=self.delta_t_max, 
                    events=(self.events), 
                    args=(self,)
                )

                self.t_min = max(sol.t)
                
                if self.t_min > self.t_max:
                    break
                elif (self.event_trigger_reason == 'too_many_cells') or (self.event_trigger_reason == 'unphysical_area') or (self.event_trigger_reason == 'unphysical_hull'):
                    break
                elif self.event_trigger_reason == 'too_soon_event':
                    self.event_trigger_reason = None
                    continue
                elif self.event_trigger_reason == 'division':
                    self.perform_divisions()
                    self.event_trigger_reason = None
                elif self.event_trigger_reason == 'cell_joins_lumen':
                    self.cell_joins_lumen()
                    if self.event_trigger_reason == 'unphysical_hull':
                        break
                    self.event_trigger_reason = None
                else:
                    self.event_trigger_reason = "unknown"
                    break

            except:
                self.event_trigger_reason = 'unknown_uncaught'
                break

        del self.all_t_values[0:2]
        self.results = pd.DataFrame()        
        if self.event_trigger_reason != "unknown" and self.event_trigger_reason != "unknown_uncaught" and self.event_trigger_reason != "unphysical_hull":
            final_positions = self.positions
            final_N_bodies = self.N_bodies
            final_ages = self.ages
            self.calculate_radius_from_age()
            final_radii = self.radii
        else:
            final_positions = (self.all_positions[-1])
            final_N_bodies = self.all_positions[-1].shape[0]
            final_ages = self.all_ages[-1]
            final_radii = self.r_min * (1 + ((np.cbrt(2)-1)  * (final_ages/self.all_lifetimes[-1])))


        hull = sp.ConvexHull(final_positions)
        volume = hull.volume
        surface_area = hull.area
        sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
        dela = sp.Delaunay(final_positions)
        self.neighbours = np.empty((final_N_bodies, final_N_bodies))
        self.neighbours[:] = np.nan
        self.neighbours = get_neighbours(dela.simplices, final_N_bodies)
        distance_matrix = np.zeros((final_N_bodies, final_N_bodies)) 

        for ii in range(final_N_bodies):
            for jj in self.neighbours[ii]:
                if np.isnan(jj) or ii <= jj:
                    continue
                else:   
                    jj = int(jj)
                    r_ij_star = final_positions[jj,:] - final_positions[ii,:]
                    sum_radii = final_radii[ii][0] + final_radii[jj][0]
                    r_mag = np.sqrt(np.dot(r_ij_star,r_ij_star))
                    distance_matrix[ii][jj] = r_mag-sum_radii

        self.results['cluster_vol'] = self.all_volumes
        self.results["sphericity"] = sphericity
        self.results["mean_separation"] = np.nanmean(distance_matrix)
        self.results['t'] = self.all_t_values
        self.results['lumen_volume'] = self.all_lumen_volumes
        self.results["run_no"] = run_number
        self.results["r_min"] = self.r_min
        self.results["beta"] = self.beta
        self.results["alpha"] = self.alpha
        self.results["a_eq_star_scaling"] = self.A_eq_star_scaling
        self.results["p_star"] = self.P_star
        self.results["mean_lifetime"] = self.mean_lifetime
        self.results["lumen_volume_scaling"] = self.volume_scaling
        self.results["lumen_radius_scaling"] = self.radius_scaling
        self.results["end_reason"] = self.event_trigger_reason
        self.results['final_N_bodies'] = self.N_bodies
        try:
            self.results['hull_volume'] = self.hull.volume
        except:
            self.results['hull_volume'] = np.nan

        if write_results:
            self.results.to_parquet("{}\\Alter{}_Run{}.parquet".format(write_path, alter, run_number))
            np.save('{}\\Alter{}_positions_{}.npy'.format(write_path, alter, run_number), np.array(self.all_positions, dtype=object), allow_pickle=True)
            np.save('{}\\Alter{}_ages_{}.npy'.format(write_path, alter, run_number), np.array(self.all_ages, dtype=object), allow_pickle=True)
