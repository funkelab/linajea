import itertools
from linajea.tracking import TrackingParameters


#Utility functions for performing grid search for solver ILP

def parameters_to_setup_string(parameters):
    return create_setup_string(parameters.cost_appear,
            parameters.cost_disappear,
            parameters.cost_split,
            parameters.weight_distance_cost,
            parameters.threshold_node_score,
            parameters.weight_node_score,
            parameters.threshold_edge_score)

def create_setup_string(
        cost_appear, 
        cost_disappear, 
        cost_split, 
        weight_distance_cost, 
        threshold_node_score, 
        weight_node_score, 
        threshold_edge_score):
        
    s = "selected"
    s += "_a_" + str(cost_appear)
    s += "_d_" + str(cost_disappear)
    s += "_s_" + str(cost_split)
    s += "_dc_" + str(weight_distance_cost)
    s += "_n_" + str(threshold_node_score)
    s += "_w_" + str(weight_node_score)
    s += "_e_" + str(threshold_edge_score)
    
    return s.replace(".", "") 


def grid_search_to_parameters(grid_search_dict):
    grid_search_keys = list(grid_search_dict.keys())

    product = itertools.product(*[grid_search_dict[key] for key in grid_search_keys])
    parameters_list = []
    for configuration in product:
        parameters = TrackingParameters()
        parameters.cost_appear = configuration[grid_search_keys.index('cost_appear')]
        parameters.cost_disappear = configuration[grid_search_keys.index('cost_disappear')]
        parameters.cost_split = configuration[grid_search_keys.index('cost_split')]
        parameters.weight_distance_cost = configuration[grid_search_keys.index('weight_distance_cost')]
        parameters.threshold_node_score = configuration[grid_search_keys.index('threshold_node_score')]
        parameters.threshold_edge_score = configuration[grid_search_keys.index('threshold_edge_score')]
        parameters.weight_node_score = configuration[grid_search_keys.index('weight_node_score')]
        parameters_list.append(parameters)
    
    return parameters_list

