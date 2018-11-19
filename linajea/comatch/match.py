from __future__ import division
import pylp
import logging

logger = logging.getLogger(__name__)

def match_components(
        nodes_x, nodes_y,
        edges_xy,
        node_labels_x, node_labels_y,
        allow_many_to_many=False,
        edge_costs=None,
        no_match_costs=0):
    '''Match nodes from X to nodes from Y by selecting candidate edges x <-> y,
    such that the split/merge error induced from the labels for X and Y is
    minimized.

    Example::

        X:     Y:

        1      a
        |      |
        2      b
        |       \
        3      h c
        |      | |
        4    C i d
        |      | |
        5      j e
        |       /
        6      f
        |      |
        7      g

        A      B

    1-7: nodes in X labelled A; a-g: nodes in Y labelled B; h-j: nodes in Y
    labelled C.

    Assuming that all nodes in X can be matched to all nodes in Y in the same
    line (``edges_xy`` would be (1, a), (2, b), (3, h), (3, c), and so on), the
    solution would be to match:

        1 - a
        2 - b
        3 - c
        4 - d
        5 - e
        6 - f
        7 - g

    h, i, and j would remain unmatched, since matching them would incur a split
    error of A into B and C.

    Args:

        nodes_x, nodes_y (array-like of ``int``):

            A list of IDs of set X and Y, respectively.

        edges_xy (array-like of tuple):

            A list of tuples ``(id_x, id_y)`` of matching edges to chose from.

        node_labels_x, node_labels_y (``dict``):

            A dictionary from IDs to labels.

        allow_many_to_many (``bool``, optional):

            If ``True``, allow that one node in X can match to multiple nodes
            in Y and vice versa. Default is ``False``.

        edge_costs (array-like of ``float``, optional):

            If given, defines a preference for selecting edges from
            ``edges_xy`` by contributing costs ``edge_costs[i]`` for edge
            ``edges_xy[i]``.

            The edge costs form a secondary objective, i.e., the matching is
            still performed to minimize the total number of errors (splits,
            merges, FPs, and FNs). However, for matching problems where several
            solutions exist with the same number of errors, the edge costs
            define a preference (e.g., by favouring matches between nodes that
            are spatially close, if the edge costs represent distances).

            See also ``no_match_costs``.

        no_match_costs (``float``, optional):

            A cost for not matching a node either in X or Y. Complementary to
            ``edge_costs``.

    Returns:

        (label_matches, node_matches, num_splits, num_merges, num_fps, num_fns)

        ``label_matches``: A list of tuples ``(label_x, label_y)`` of labels
        that got matched.

        ``node_matches``: A list of tuples ``(id_x, id_y)`` of nodes that got
        matched. Subset of ``edges_xy``.

        ``num_splits``, ``num_merges``, ...: The number of label splits,
        merges, false positives (unmatched in X), and false negatives
        (unmatched in Y).
    '''

    if edge_costs is None and no_match_costs != 0:
        edge_costs = [ 0 ]*len(edges_xy)

    num_vars = 0

    # add "no match in X" and "no match in Y" dummy nodes
    logger.debug("node_labels_x.keys() " + str(node_labels_x.keys()))
    logger.debug("node_labels_y.keys() " + str(node_labels_y.keys()))
    no_match_node = max(nodes_x + nodes_y) + 1
    no_match_label = max(max(node_labels_x.keys()), max(node_labels_y.keys())) + 1

    node_labels_x = dict(node_labels_x)
    node_labels_y = dict(node_labels_y)
    node_labels_x.update({no_match_node: no_match_label})
    node_labels_y.update({no_match_node: no_match_label})

    labels_x = set(node_labels_x.values())
    labels_y = set(node_labels_y.values())

    # add additional edges to dummy nodes
    edges_xy += [ (n, no_match_node) for n in nodes_x ]
    edges_xy += [ (no_match_node, n) for n in nodes_y ]

    # create indicator for each matching edge
    edge_indicators = {}
    edges_by_node_x = {}
    edges_by_node_y = {}
    for edge in edges_xy:
        edge_indicators[edge] = num_vars
        num_vars += 1
        u, v = edge
        if u not in edges_by_node_x:
            edges_by_node_x[u] = []
        if v not in edges_by_node_y:
            edges_by_node_y[v] = []
        edges_by_node_x[u].append(edge)
        edges_by_node_y[v].append(edge)

    # Require that each node matches to exactly one (or at least one, depending
    # on the allow_many_to_many parameter) other node. Dummy nodes can match to
    # any number.

    constraints = pylp.LinearConstraints()

    for nodes, edges_by_node in zip(
            [nodes_x, nodes_y], [edges_by_node_x, edges_by_node_y]):

        for node in nodes:

            if node == no_match_node:
                continue

            constraint = pylp.LinearConstraint()
            for edge in edges_by_node[node]:
                constraint.set_coefficient(edge_indicators[edge], 1)
            if allow_many_to_many:
                constraint.set_relation(pylp.Relation.GreaterEqual)
            else:
                constraint.set_relation(pylp.Relation.Equal)
            constraint.set_value(1)
            constraints.add(constraint)

    # add indicators for label matches

    label_indicators = {}
    edges_by_label_pair = {}

    for edge in edges_xy:

        label_pair = node_labels_x[edge[0]], node_labels_y[edge[1]]

        if label_pair not in label_indicators:
            label_indicators[label_pair] = num_vars
            num_vars += 1

        if label_pair not in edges_by_label_pair:
            edges_by_label_pair[label_pair] = []
        edges_by_label_pair[label_pair].append(edge)

    label_indicators[(no_match_label, no_match_label)] = num_vars
    num_vars += 1

    # couple label indicators to edge indicators
    for label_pair, edges in edges_by_label_pair.items():

        # y == 1 <==> sum(x1, ..., xn) > 0
        #
        # y - sum(x1, ..., xn) <= 0
        # sum(x1, ..., xn) - n*y <= 0

        constraint1 = pylp.LinearConstraint()
        constraint2 = pylp.LinearConstraint()
        constraint1.set_coefficient(label_indicators[label_pair], 1)
        constraint2.set_coefficient(label_indicators[label_pair], -len(edges))
        for edge in edges:
            constraint1.set_coefficient(edge_indicators[edge], -1)
            constraint2.set_coefficient(edge_indicators[edge], 1)
        constraint1.set_relation(pylp.Relation.LessEqual)
        constraint2.set_relation(pylp.Relation.LessEqual)
        constraint1.set_value(0)
        constraint2.set_value(0)
        constraints.add(constraint1)
        constraints.add(constraint2)

    # pin no-match pair indicator to 1
    constraint = pylp.LinearConstraint()
    no_match_indicator = label_indicators[(no_match_label, no_match_label)]
    constraint.set_coefficient(no_match_indicator, 1)
    constraint.set_relation(pylp.Relation.Equal)
    constraint.set_value(1)
    constraints.add(constraint)

    # add integer for splits
    #   splits = sum of all label pair indicators - n
    # with n number of labels in x (including no-match)
    #   sum - splits = n
    splits = num_vars
    num_vars += 1
    constraint = pylp.LinearConstraint()
    for _, label_indicator in label_indicators.items():
        constraint.set_coefficient(label_indicator, 1)
    constraint.set_coefficient(splits, -1)
    constraint.set_relation(pylp.Relation.Equal)
    constraint.set_value(len(labels_x))
    constraints.add(constraint)

    # add integer for merges
    merges = num_vars
    num_vars += 1
    constraint = pylp.LinearConstraint()
    for _, label_indicator in label_indicators.items():
        constraint.set_coefficient(label_indicator, 1)
    constraint.set_coefficient(merges, -1)
    constraint.set_relation(pylp.Relation.Equal)
    constraint.set_value(len(labels_y))
    constraints.add(constraint)

    # set objective
    objective = pylp.LinearObjective(num_vars)
    objective.set_coefficient(splits, 1)
    objective.set_coefficient(merges, 1)

    min_edge_cost = None
    if edge_costs is not None:

        edge_costs, no_match_costs = normalize_matching_costs(
            len(nodes_x), len(nodes_y),
            edge_costs,
            no_match_costs)
    
        edge_costs += [ no_match_costs ]*(len(nodes_x) + len(nodes_y))
        min_edge_cost = min(edge_costs)

        for edge, cost in zip(edges_xy, edge_costs):
            objective.set_coefficient(
                edge_indicators[edge],
                cost)

    # solve

    logger.debug("Added %d constraints", len(constraints))
    for i in range(len(constraints)):
        logger.debug(constraints[i])

    logger.debug("Creating linear solver")
    solver = pylp.create_linear_solver(pylp.Preference.Any)
    variable_types = pylp.VariableTypeMap()
    variable_types[splits] = pylp.VariableType.Integer
    variable_types[merges] = pylp.VariableType.Integer

    if min_edge_cost is not None:
        logger.debug("Set optimality gap to lowest edge cost")
        epsilon = 10**(-4)
        solver.set_optimality_gap(max(min_edge_cost - epsilon, 0.0), True)

    logger.debug("Initializing solver with %d variables", num_vars)
    solver.initialize(num_vars, pylp.VariableType.Binary, variable_types)

    logger.debug("Setting objective")
    solver.set_objective(objective)

    logger.debug("Setting constraints")
    solver.set_constraints(constraints)

    logger.debug("Solving...")
    solution, message = solver.solve()

    logger.debug("Solver returned: %s", message)
    if 'NOT' in message:
        raise RuntimeError("No optimal solution found...")

    # get label matches

    label_matches = []
    for label_pair, label_indicator in label_indicators.items():
        if no_match_node not in label_pair:
            if solution[label_indicator] > 0.5:
                label_matches.append(label_pair)

    # get node matches

    node_matches = [
        e
        for e in edges_xy
        if solution[edge_indicators[e]] > 0.5 and no_match_node not in e
    ]

    # get error counts

    num_splits = solution[splits]
    num_merges = solution[merges]
    num_fps = 0
    num_fns = 0
    for label_pair, label_indicator in label_indicators.items():
        if label_pair[0] == no_match_label:
            num_fps += solution[label_indicator]
        if label_pair[1] == no_match_label:
            num_fns += solution[label_indicator]
    num_fps -= 1
    num_fns -= 1
    num_splits -= num_fps
    num_merges -= num_fns

    return (label_matches, node_matches, num_splits, num_merges, num_fps, num_fns)

def normalize_matching_costs(
        num_nodes_x, num_nodes_y,
        edge_costs,
        no_match_costs):
    '''Scale the edge costs and no-match costs such that they do not exceed 1
    in the worst case. This is to ensure that the ILP first minimizes
    topological errors, then matching costs.'''

    # the sum of all edge costs is an upper bound on the actual edge costs
    total_edge_costs = (
        sum(edge_costs) +
        (num_nodes_x + num_nodes_y)*no_match_costs
    )

    edge_costs = [ c/total_edge_costs for c in edge_costs ]
    no_match_costs /= total_edge_costs

    return edge_costs, no_match_costs
