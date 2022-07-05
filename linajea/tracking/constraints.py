import pylp


def ensure_edge_endpoints(edge, indicators):
    """if e is selected, u and v have to be selected
    """
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

    return [constraint]


def ensure_one_predecessor(node, indicators, graph, **kwargs):
    """Every selected node has exactly one selected edge to the previous frame
    This includes the special "appear" edge.

      sum(prev) - node = 0 # exactly one prev edge,
                             iff node selected
    """
    pinned_edges = kwargs["pinned_edges"]

    constraint_prev = pylp.LinearConstraint()

    # all neighbors in previous frame
    pinned_to_1 = []
    for edge in graph.prev_edges(node):
        constraint_prev.set_coefficient(indicators["edge_selected"][edge], 1)
        if edge in pinned_edges and pinned_edges[edge]:
            pinned_to_1.append(edge)
    if len(pinned_to_1) > 1:
        raise RuntimeError(
            "Node %d has more than one prev edge pinned: %s"
            % (node, pinned_to_1))
    # plus "appear"
    constraint_prev.set_coefficient(indicators["node_appear"][node], 1)

    #node
    constraint_prev.set_coefficient(indicators["node_selected"][node], -1)

    # relation, value
    constraint_prev.set_relation(pylp.Relation.Equal)

    constraint_prev.set_value(0)

    return [constraint_prev]


def ensure_at_most_two_successors(node, indicators, graph, **kwargs):
    """Every selected node has zero to two selected edges to the next frame.

      sum(next) - 2*node <= 0 # at most two next edges
    """
    constraint_next = pylp.LinearConstraint()

    for edge in graph.next_edges(node):
        constraint_next.set_coefficient(indicators["edge_selected"][edge], 1)

    # node
    constraint_next.set_coefficient(indicators["node_selected"][node], -2)

    # relation, value
    constraint_next.set_relation(pylp.Relation.LessEqual)

    constraint_next.set_value(0)

    return [constraint_next]


def ensure_split_set_for_divs(node, indicators, graph, **kwargs):
    """Ensure that the split indicator is set for every cell that splits
    into two daughter cells.
    I.e., each node with two forwards edges is a split node.

    Constraint 1
      sum(forward edges) - split   <= 1
      sum(forward edges) >  1 => split == 1

    Constraint 2
      sum(forward edges) - 2*split >= 0
      sum(forward edges) <= 1 => split == 0
    """
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

    return [constraint_1, constraint_2]


def ensure_pinned_edge(edge, indicators, selected):
    """Ensure that if an edge has already been set by a neighboring block
    its state stays consistent (pin/fix its state).

    Constraint:
    If selected
      selected(e) = 1
    else:
      selected(e) = 0
    """
    ind_e = indicators["edge_selected"][edge]
    constraint = pylp.LinearConstraint()
    constraint.set_coefficient(ind_e, 1)
    constraint.set_relation(pylp.Relation.Equal)
    constraint.set_value(1 if selected else 0)

    return [constraint]


def ensure_one_state(node, indicators):
    """Ensure that each selected node has exactly on state assigned to it

    Constraint:
      split(n) + child(n) + continuation(n) = selected(n)
    """
    constraint = pylp.LinearConstraint()
    constraint.set_coefficient(indicators["node_split"][node], 1)
    constraint.set_coefficient(indicators["node_child"][node], 1)
    constraint.set_coefficient(indicators["node_continuation"][node], 1)
    constraint.set_coefficient(indicators["node_selected"][node], -1)
    constraint.set_relation(pylp.Relation.Equal)
    constraint.set_value(0)

    return [constraint]


def ensure_split_child(edge, indicators):
    """If an edge is selected, the split (division) and child indicators
    are linked. Let e=(u,v) be an edge linking node u at time t + 1 to v
    in time t.

    Constraints:
      child(u) + selected(e) - split(v) <= 1
      split(v) + selected(e) - child(u) <= 1
    """
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

    return [constraint_1, constraint_2]


def get_constraints_default(config):
    solver_type = config.solve.solver_type

    if solver_type == "basic":
        pin_constraints_fn_list = [ensure_pinned_edge]
        edge_constraints_fn_list = [ensure_edge_endpoints]
        node_constraints_fn_list = []
        inter_frame_constraints_fn_list = [
            ensure_one_predecessor,
            ensure_at_most_two_successors,
            ensure_split_set_for_divs]
    elif solver_type == "cell_state":
        pin_constraints_fn_list = [ensure_pinned_edge]
        edge_constraints_fn_list = [ensure_edge_endpoints, ensure_split_child]
        node_constraints_fn_list = [ensure_one_state]
        inter_frame_constraints_fn_list = [
            ensure_one_predecessor,
            ensure_at_most_two_successors,
            ensure_split_set_for_divs]
    else:
        raise RuntimeError("solver_type %s unknown for constraints",
                           solver_type)

    return (pin_constraints_fn_list,
            edge_constraints_fn_list,
            node_constraints_fn_list,
            inter_frame_constraints_fn_list)
