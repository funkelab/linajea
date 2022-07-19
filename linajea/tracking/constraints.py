import pylp


def ensure_edge_endpoints(graph, indicators):
    """If edge is selected, u and v have to be selected.

    Constraint:
      2 * edge(u, v) - u - v <= 0

    Args
    ----
    graph: TrackGraph
        Graph containing the node and edge candidates, the ILP will be solved
        for this graph.
    indicators: dict str: dict (int or pair of int): int
        Contains a dict for every indicator type (str).
        Each dict maps from node (int) or edge (pair of int) candidate to
        the corresponding indicator variable/index (int). Each candidate can
        have indicators of different types associated to it.
    """
    constraints = []
    for edge in graph.edges():
        u, v = edge
        ind_e = indicators["edge_selected"][edge]
        ind_u = indicators["node_selected"][u]
        ind_v = indicators["node_selected"][v]

        constraint = pylp.LinearConstraint()
        constraint.set_coefficient(ind_e, 2)
        constraint.set_coefficient(ind_u, -1)
        constraint.set_coefficient(ind_v, -1)
        constraint.set_relation(pylp.Relation.LessEqual)
        constraint.set_value(0)
        constraints.append(constraint)

    return constraints


def ensure_one_predecessor(graph, indicators):
    """Every selected node has exactly one selected edge to the previous frame
    This includes the special "appear" edge.

    Constraint:
      sum(prev) - node = 0 # exactly one prev edge,
                             iff node selected

    Args
    ----
    graph: TrackGraph
        Graph containing the node and edge candidates, the ILP will be solved
        for this graph.
    indicators: dict str: dict (int or pair of int): int
        Contains a dict for every indicator type (str).
        Each dict maps from node (int) or edge (pair of int) candidate to
        the corresponding indicator variable/index (int). Each candidate can
        have indicators of different types associated to it.
    """
    constraints = []
    for node in graph.nodes():
        constraint = pylp.LinearConstraint()

        # all neighbors in previous frame
        for edge in graph.prev_edges(node):
            constraint.set_coefficient(indicators["edge_selected"][edge], 1)

        # plus "appear"
        constraint.set_coefficient(indicators["node_appear"][node], 1)

        # node
        constraint.set_coefficient(indicators["node_selected"][node], -1)

        # relation, value
        constraint.set_relation(pylp.Relation.Equal)

        constraint.set_value(0)
        constraints.append(constraint)

    return constraints


def ensure_at_most_two_successors(graph, indicators):
    """Every selected node has zero to two selected edges to the next frame.

    Constraint:
      sum(next) - 2*node <= 0 # at most two next edges

    Args
    ----
    graph: TrackGraph
        Graph containing the node and edge candidates, the ILP will be solved
        for this graph.
    indicators: dict str: dict (int or pair of int): int
        Contains a dict for every indicator type (str).
        Each dict maps from node (int) or edge (pair of int) candidate to
        the corresponding indicator variable/index (int). Each candidate can
        have indicators of different types associated to it.
    """
    constraints = []
    for node in graph.nodes():
        constraint = pylp.LinearConstraint()

        for edge in graph.next_edges(node):
            constraint.set_coefficient(indicators["edge_selected"][edge], 1)

        # node
        constraint.set_coefficient(indicators["node_selected"][node], -2)

        # relation, value
        constraint.set_relation(pylp.Relation.LessEqual)

        constraint.set_value(0)
        constraints.append(constraint)

    return constraints


def ensure_split_set_for_divs(graph, indicators):
    """Ensure that the split indicator is set for every cell that splits
    into two daughter cells.
    I.e., each node with two forwards edges is a split node.

    Constraint 1:
      sum(forward edges) - split   <= 1
      sum(forward edges) >  1 => split == 1

    Constraint 2:
      sum(forward edges) - 2*split >= 0
      sum(forward edges) <= 1 => split == 0

    Args
    ----
    graph: TrackGraph
        Graph containing the node and edge candidates, the ILP will be solved
        for this graph.
    indicators: dict str: dict (int or pair of int): int
        Contains a dict for every indicator type (str).
        Each dict maps from node (int) or edge (pair of int) candidate to
        the corresponding indicator variable/index (int). Each candidate can
        have indicators of different types associated to it.
    """
    constraints = []
    for node in graph.nodes():
        constraint_1 = pylp.LinearConstraint()
        constraint_2 = pylp.LinearConstraint()

        # sum(forward edges)
        for edge in graph.next_edges(node):
            constraint_1.set_coefficient(indicators["edge_selected"][edge], 1)
            constraint_2.set_coefficient(indicators["edge_selected"][edge], 1)

        # -[2*]split
        constraint_1.set_coefficient(indicators["node_split"][node], -1)
        constraint_2.set_coefficient(indicators["node_split"][node], -2)

        constraint_1.set_relation(pylp.Relation.LessEqual)
        constraint_2.set_relation(pylp.Relation.GreaterEqual)

        constraint_1.set_value(1)
        constraint_2.set_value(0)
        constraints.append(constraint_1)
        constraints.append(constraint_2)

    return constraints


def ensure_pinned_edge(graph, indicators, selected_key):
    """Ensure that if an edge has already been set by a neighboring block
    its state stays consistent (pin/fix its state).

    Constraint:
    If selected
      selected(e) = 1
    else:
      selected(e) = 0

    Args
    ----
    graph: TrackGraph
        Graph containing the node and edge candidates, the ILP will be solved
        for this graph.
    indicators: dict str: dict (int or pair of int): int
        Contains a dict for every indicator type (str).
        Each dict maps from node (int) or edge (pair of int) candidate to
        the corresponding indicator variable/index (int). Each candidate can
        have indicators of different types associated to it.
    selected_key: str
        Consider this property to determine pinned state of candidate.
    """
    constraints = []
    for edge in graph.edges():
        if selected_key in graph.edges[edge]:
            selected = graph.edges[edge][selected_key]

            ind_e = indicators["edge_selected"][edge]
            constraint = pylp.LinearConstraint()
            constraint.set_coefficient(ind_e, 1)
            constraint.set_relation(pylp.Relation.Equal)
            constraint.set_value(1 if selected else 0)
            constraints.append(constraint)

    return constraints


def ensure_one_state(graph, indicators):
    """Ensure that each selected node has exactly on state assigned to it

    Constraint:
      split(n) + child(n) + continuation(n) = selected(n)

    Args
    ----
    graph: TrackGraph
        Graph containing the node and edge candidates, the ILP will be solved
        for this graph.
    indicators: dict str: dict (int or pair of int): int
        Contains a dict for every indicator type (str).
        Each dict maps from node (int) or edge (pair of int) candidate to
        the corresponding indicator variable/index (int). Each candidate can
        have indicators of different types associated to it.
    """
    constraints = []
    for node in graph.nodes():
        constraint = pylp.LinearConstraint()
        constraint.set_coefficient(indicators["node_split"][node], 1)
        constraint.set_coefficient(indicators["node_child"][node], 1)
        constraint.set_coefficient(indicators["node_continuation"][node], 1)
        constraint.set_coefficient(indicators["node_selected"][node], -1)
        constraint.set_relation(pylp.Relation.Equal)
        constraint.set_value(0)
        constraints.append(constraint)

    return constraints


def ensure_split_child(graph, indicators):
    """If an edge is selected, the split (division) and child indicators
    are linked. Let e=(u,v) be an edge linking node u at time t + 1 to v
    in time t.

    Constraint 1:
      child(u) + selected(e) - split(v) <= 1
    Constraint 2:
      split(v) + selected(e) - child(u) <= 1

    Args
    ----
    graph: TrackGraph
        Graph containing the node and edge candidates, the ILP will be solved
        for this graph.
    indicators: dict str: dict (int or pair of int): int
        Contains a dict for every indicator type (str).
        Each dict maps from node (int) or edge (pair of int) candidate to
        the corresponding indicator variable/index (int). Each candidate can
        have indicators of different types associated to it.
    """
    constraints = []
    for edge in graph.edges():
        u, v = edge
        ind_e = indicators["edge_selected"][edge]
        split_v = indicators["node_split"][v]
        child_u = indicators["node_child"][u]

        constraint_1 = pylp.LinearConstraint()
        constraint_1.set_coefficient(child_u, 1)
        constraint_1.set_coefficient(ind_e, 1)
        constraint_1.set_coefficient(split_v, -1)
        constraint_1.set_relation(pylp.Relation.LessEqual)
        constraint_1.set_value(1)

        constraint_2 = pylp.LinearConstraint()
        constraint_2.set_coefficient(split_v, 1)
        constraint_2.set_coefficient(ind_e, 1)
        constraint_2.set_coefficient(child_u, -1)
        constraint_2.set_relation(pylp.Relation.LessEqual)
        constraint_2.set_value(1)
        constraints.append(constraint_1)
        constraints.append(constraint_2)

    return constraints


def get_default_constraints(config):
    solver_type = config.solve.solver_type

    if solver_type == "basic":
        pin_constraints_fn_list = [ensure_pinned_edge]
        constraints_fn_list = [ensure_edge_endpoints,
                               ensure_one_predecessor,
                               ensure_at_most_two_successors,
                               ensure_split_set_for_divs]
    elif solver_type == "cell_state":
        pin_constraints_fn_list = [ensure_pinned_edge]
        constraints_fn_list = [ensure_edge_endpoints, ensure_split_child,
                               ensure_one_state,
                               ensure_one_predecessor,
                               ensure_at_most_two_successors,
                               ensure_split_set_for_divs]
    else:
        raise RuntimeError("solver_type %s unknown for constraints",
                           solver_type)

    return (pin_constraints_fn_list,
            constraints_fn_list)
