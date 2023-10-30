import argparse
import numpy as np
import dimod
from dwave.system import LeapHybridSampler

import demo

def build_bqm(possible_metro_stations, num_poi, pois, num_existing_stations, existing_stations, num_new_metro_stations, poi_weights):
    """Build bqm that models our problem scenario using NumPy. 

    Args:
        possible_metro_stations (list of tuples of ints): 
            Potential new metro stations
        num_poi (int): Number of points of interest
        pois (list of tuples of ints): A fixed set of points of interest
        num_existing_stations (int): Number of existing charging stations
        existing_stations (list of tuples of ints): 
            A fixed set of current metro stations
        num_new_metro_stations (int): Number of new charging stations desired
    
    Returns:
        bqm_np (BinaryQuadraticModel): QUBO model for the input scenario
    """

    # Tunable parameters
    gamma1 = len(possible_metro_stations) * 4.
    gamma2 = len(possible_metro_stations) / 3.
    gamma3 = len(possible_metro_stations) * 1.7
    gamma4 = len(possible_metro_stations) ** 3

    # Build BQM using adjVectors to find best new metro stations s.t. min
    # distance to POIs and max distance to existing metro stations
    linear = np.zeros(len(possible_metro_stations))

    nodes_array = np.asarray(possible_metro_stations)
    pois_array = np.asarray(pois)
    cs_array = np.asarray(existing_stations)

    # Constraint 1: Min average distance to POIs
    if num_poi > 0:

        ct_matrix = (np.matmul(nodes_array, pois_array.T)*(-2.) 
                    + np.sum(np.square(pois_array), axis=1).astype(float) 
                    + np.sum(np.square(nodes_array), axis=1).reshape(-1,1).astype(float))

        # Initialize the linear term
        linear = np.zeros(len(possible_metro_stations))

        # Adjust the linear term based on the weights
        for i in range(len(poi_weights)):
            if poi_weights[i] >= 3:
            # If weight is three or more, prioritize minimizing distance
                linear += ct_matrix[:, i] * gamma1
            else:
            # If weight is less than three, consider other constraints
                linear += np.sum(ct_matrix * poi_weights, axis=1) / num_poi * gamma1

    # Adjust other constraints as needed

    # Constraint 2: Max distance to existing metro
    if num_existing_stations > 0:    

        dist_mat = (np.matmul(nodes_array, cs_array.T)*(-2.) 
                    + np.sum(np.square(cs_array), axis=1).astype(float) 
                    + np.sum(np.square(nodes_array), axis=1).reshape(-1,1).astype(float))

        linear += -1 * np.sum(dist_mat, axis=1) / num_existing_stations * gamma2 

    # Constraint 3: Max distance to other new metro stations
    if num_new_metro_stations > 1:

        dist_mat = -gamma3*((np.matmul(nodes_array, nodes_array.T)*(-2.) 
                    + np.sum(np.square(nodes_array), axis=1)).astype(float) 
                    + np.sum(np.square(nodes_array), axis=1).reshape(-1,1).astype(float))

    else:
        dist_mat = np.zeros((len(possible_metro_stations),len(possible_metro_stations)))

    # Constraint 4: Choose exactly num_new_metro_stations new metro stations
    linear += (1-2*num_new_metro_stations)*gamma4
    dist_mat += 2*gamma4
    dist_mat = np.triu(dist_mat, k=1).flatten()

    quad_col = np.tile(np.arange(len(possible_metro_stations)), len(possible_metro_stations))
    quad_row = np.tile(np.arange(len(possible_metro_stations)), 
                (len(possible_metro_stations),1)).flatten('F')

    q2 = quad_col[dist_mat != 0]
    q1 = quad_row[dist_mat != 0]
    q3 = dist_mat[dist_mat != 0]
    
    bqm_np = dimod.BinaryQuadraticModel.from_numpy_vectors(linear=linear, 
                                                            quadratic=(q1, q2, q3), 
                                                            offset=0, 
                                                            vartype=dimod.BINARY)

    return bqm_np

if __name__ == '__main__':

    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description='New Metro Locations')

    # Define command-line arguments
    parser.add_argument('--width', type=int, default=100, help='Width of the city grid')
    parser.add_argument('--height', type=int, default=100, help='Height of the city grid')
    parser.add_argument('--poi', type=int, default=10, help='Number of Points of Interest (POIs)')
    parser.add_argument('--metro', type=int, default=5, help='Number of Existing Metro Stations')
    parser.add_argument('--new_metro', type=int, default=3, help='Number of New Metro Stations')


    # Parse the command-line arguments
    args = parser.parse_args()

    # Build large grid graph for city
    G, pois, existing_stations, possible_metro_stations = demo.set_up_scenario(args.width, 
                                                                            args.height, 
                                                                            args.poi, 
                                                                            args.metro)
    
    poi_weights = np.random.rand(len(possible_metro_stations))


    poi_weights = poi_weights[:, np.newaxis]

    # Build BQM
    bqm = build_bqm(possible_metro_stations, 
                    args.poi, 
                    pois, 
                    args.metro, 
                    existing_stations, 
                    args.new_metro,
                    poi_weights)

    # Run BQM on HSS
    sampler = LeapHybridSampler()
    print("\nRunning scenario on", sampler.solver.id, "solver...")

    new_metro_nodes = demo.run_bqm_and_collect_solutions(bqm, sampler, possible_metro_stations)

    # Print results to commnand-line for user
    demo.printout_solution_to_cmdline(pois, 
                                    args.poi, 
                                    existing_stations, 
                                    args.metro, 
                                    new_metro_nodes, 
                                    args.new_metro)

    # Create scenario output image
    demo.save_output_image(G, pois, existing_stations, new_metro_nodes)
