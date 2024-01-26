import numpy as np
import pandas as pd
import os

def starting_state_generator(p_honest:float, p_coerced:float)->np.array:
    """Given the proabilities of an agent being honest and of being coerced, returns an array of the starting state probabilites"""
    p_dishonest = 1 - p_honest
    p_not_coerced = 1 - p_coerced


    a1 = p_dishonest * p_not_coerced * 1
    b1 = 0
    c1 = p_honest * p_not_coerced 
    d1 = p_honest * p_coerced
    e1 = p_dishonest * p_coerced * 1
    f1 = 0

    starting_state = np.array([a1, b1, c1, d1, e1, f1])
    return starting_state

def transition_matrix_generator(p_honest:float, p_coerced:float)->np.array:
    """Given the proabilities of an agent being honest and of being coerced,
    returns a transition matrix that determines the probability
    of picking a child in a given state, given that the parent is in another state. """

    p_dishonest = 1 - p_honest
    p_not_coerced = 1 - p_coerced

    m11 = 0
    m12 = p_dishonest * p_not_coerced * 1 
    m13 = p_honest * p_not_coerced * 1
    m14 = p_honest * p_coerced * 1
    m15 = 0
    m16 = p_dishonest * p_coerced * 1

    m21 = 0
    m22 = 0
    m23 = 0
    m24 = 0
    m25 = 0
    m26 = 0

    m31 = 0
    m32 = p_dishonest * p_not_coerced * 1
    m33 = p_honest * p_not_coerced * 1
    m34 = p_honest * p_coerced * 1
    m35 = 0
    m36 = p_dishonest * p_coerced * 1

    m41 = p_dishonest * p_not_coerced * 1
    m42 = 0
    m43 = p_honest * p_not_coerced * 1
    m44 = p_honest * p_coerced * 1
    m45 = p_dishonest * p_coerced * 1
    m46 = 0

    m51 = p_dishonest * p_not_coerced * 1
    m52 = 0
    m53 = p_honest * p_not_coerced * 1
    m54 = p_honest * p_coerced * 1
    m55 = p_dishonest * p_coerced * 1
    m56 = 0

    m61 = 0
    m62 = 0
    m63 = 0
    m64 = 0
    m65 = 0
    m66 = 0

    M = np.array([[m11, m12, m13, m14, m15, m16],
            [m21, m22, m23, m24, m25, m26],
            [m31, m32, m33, m34, m35, m36],
            [m41, m42, m43, m44, m45, m46],
            [m51, m52, m53, m54, m55, m56],
            [m61, m62, m63, m64, m65, m66]])
    
    return M

def platoon_initial_state_generator(p_honest:float, p_coerced:float) -> np.array:
    """Given the proabilities of an agent being honest and of being coerced, 
    returns an array of the starting state probabilites of a platoon"""

    p_dishonest = 1 - p_honest
    p_not_coerced = 1 - p_coerced

    #initial state can be only state A, D and E, and D is 0. Because we say a platoon is comandeered by a dishonest car. 
    ap1 = p_dishonest * p_not_coerced * 1
    dp1 = 0
    ep1 = p_dishonest * p_coerced * 1

    platoon_initial_state = np.array([ap1, dp1, ep1])
    return platoon_initial_state

def platoon_transition_matrix_generator(p_honest:float, p_coerced:float) -> np.array:
    """Given the proabilities of an agent being honest and of being coerced,
    returns a transition matrix that determines the probability
    of picking a child in a given state, given that the parent is in another state. """
    
    p_dishonest = 1 - p_honest
    p_not_coerced = 1 - p_coerced

    mp11 = 0
    mp12 = p_honest * p_coerced * 1
    mp13 = 0

    mp21 = p_dishonest * p_not_coerced * 1
    mp22 = p_honest * p_coerced * 1
    mp23 = p_dishonest * p_coerced * 1

    mp31 = p_dishonest * p_not_coerced * 1
    mp32 = p_honest * p_coerced * 1
    mp33 = p_dishonest * p_coerced * 1

    Mp = np.array([[mp11, mp12, mp13],
                [mp21, mp22, mp23], 
                [mp31, mp32, mp33]])
    
    return Mp



def edge_probabilities(starting_state:np.array, transition_matrix:np.array, approvals_kernel:np.array, depth:int, number_of_neighbours:list)-> np.array:
    """Function to compute the vector with the expected number of edge conections in a tree of a given depth, and number of neighbours.
    params: starting_state: inital probabilities of being in each state as a parent. Numpy array of 1 by 6.
            transition_matrix: matrix containing the probabilities of picking a child in a given state, given that the parent is in a state.
            Numpy array, (6 by 6)
            approvals_kernel: Matrix containing the probability that a child will approve a parent, 
            when the child and the parent are in a given state.
            Numpy array of (6 by 6).
            depth: depth of the tree. Int.
            number_of_neighbours: list containing the number of neighbours each parent has per depth level. list of length = depth

    returns: vector containing the expected number of edges. 
    The index of the vector entry corresponds to the state combination of parent and child edges.
            """
    #initalising an empty state space vector for the recursive sum
    new_state = np.zeros(len(starting_state))

    assert len(number_of_neighbours) == depth
    assert transition_matrix.size == approvals_kernel.size

    #Hadamard product of probability of picking a given child (Transition_matrix) 
    #and probability of that child approving their parent (approvals_kernel)
    P = np.multiply(transition_matrix, approvals_kernel)
    for d in range(1, depth + 1, 1):
        #Raising matrix P to the power of the depth, for each depth level.
        Pd = np.linalg.matrix_power(P, d)
        new_state += ((d * number_of_neighbours[d - 1]) * (np.matmul(starting_state, Pd)))

    #return np.sum(new_state)
    return new_state


def parser(simulation_number, probability_of_honest, probability_of_coerced, total_edge_connections):

    row_list = [simulation_number, probability_of_honest, probability_of_coerced, total_edge_connections]

    return row_list

def simulator(number_of_simulations:int, prob_coerced:float, prob_honest:float, 
            inital_state:np.array, state_transition_matrix:np.array, approvals_matrix:np.array, d:int, neighbours_list:list):
    
    data = []
    for simulation_id in range(number_of_simulations):    
        edge_probability_vector = edge_probabilities(starting_state=inital_state, transition_matrix=state_transition_matrix, 
                                                    approvals_kernel=approvals_matrix, depth=d, number_of_neighbours=neighbours_list)
        
        total_edge_connections = np.sum(edge_probability_vector)
        
        row = parser(simulation_id, prob_honest, prob_coerced, total_edge_connections)
        data.append(row)

    simulation_df = pd.DataFrame(data, 
    columns=['Simulation number', 'Probability of honest cars', 'Probability of coerced cars', 'Expected Number of Edges'])

    return simulation_df

def save_simulation(simulation_df, path, simulation_id):
    simulation_path = path + str(simulation_id) + '.txt'
    simulation_df.to_csv(simulation_path)

    return simulation_path

def make_directory(target_path):
    cwd = os.getcwd()
    path = cwd + target_path
    os.makedirs(path, exist_ok =True)
    return path

def full_csv(directory_path_string):
    """Given a directory pathfile with .txt files of simulation data, 
    loops through each one, reads them and creates one .csv file with 
    all the simulation data"""
    
    directory = os.fsencode(directory_path_string)
    dfs = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        
        if filename.endswith('.txt'):
            simulation_path = directory_path_string + filename
            data = pd.read_csv(simulation_path)
            dfs.append(data)

    return pd.concat(dfs)