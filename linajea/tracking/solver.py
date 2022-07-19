"""Provides a solver object encapsulating the ILP solver
"""
import logging

import pylp

logger = logging.getLogger(__name__)


class Solver(object):
    """ Class for initializing and solving an ILP problem for candidate nodes
    and edges using pylp.

    The set of indicators and the main constraints have to be set on
    initialization. The resulting problem can be solved multiple times
    (using the same pylp.Solver instance) for different objectives by
    calling `update_objective` in between successive calls to solve_and_set.
    In each call to `update_objective` different costs per indicator can be
    used and the value for a different set of indicators can be pinned.

    Attributes
    ----------
    graph: nx.DiGraph or TrackGraph
        Graph containing the node and edge candidates, the ILP will be solved
        for this graph.
    timeout: int
        Terminate solver after this time; if terminated early solution might
        be suboptimal; set to 0 to disable (default).
    num_threads: int
        Number of threads to use for solving. Default is 1. Depending on
        the used solver and the type of problem parallelization of ILPs
        often does not scale that well, one approach is to use different
        methods or heuristics in parallel and stop as soon as one of them
        was successful.
    num_vars: int
        Will be set to the total number of indicator variables.
    selected_key: str
        After solving, this property of node/edge candidates will be set
        depending on if they are part of the solution or not. Its value
        will be set by calling `update_objective`.
    objective: pylp.LinearObjective
        Will be set to the objective to be solved in `update_objective`
    main_constraints: list of pylp.LinearConstraint
        Will be filled by calling the Callables in constraints_fns on the
        graph
    pin_constraints: list of pylp.LinearConstraint
        Will be filled by calling the Callables in pin_constraints_fns on the
        graph
    solver: pylp.LinearSolver
        Will contain an instance of a pylp solver (e.g. Gurobi)
    node_indicator_names: list of str
        List of types of node indicators
    edge_indicator_names: list of str
        List of types of edge indicators
    constraints_fns: list of Callable
        Each Callable should handle a single type of constraint.
        It should create the respective constraints for all affected objects in
        the graph and return them.
        Add more Callable to this list to add additional constraints.
        See tracking/constraints.py for examples.
        Interface: fn(graph, indicators) -> constraints
          graph: Create constraints for nodes/edges in this graph
          indicators: The indicator dict created by this Solver object
          constraints: list of pylp.LinearConstraint
    pin_constraints_fns: list of Callable
        Each Callable should handle a single type of pin constraint.
        Use this to add constraints to pin indicators to specific states.
        Created only if indicator has already been set by neighboring blocks.
        Interface: fn(graph, indicators, selected) -> constraints
          graph: Create constraints for nodes/edges in this graph
          indicators: The indicator dict created by this Solver object
          selected_key: Consider this property to determine state of candidate
          constraints: list of pylp.LinearConstraint
    """
    def __init__(self, graph,
                 node_indicator_names, edge_indicator_names,
                 constraints_fns, pin_constraints_fns, timeout=0,
                 num_threads=1):
        """Constructs a Solver object.
        Sets the object attributes and calls `_create_indicators` to create
        all indicators, `_create_solver` to create an instance of a pylp
        solver object and `_create_constraints` to create the constraints.
        Call `update_objective` afterwards to set the objective, followed by
        `solve_and_set` to compute the solution and to set the appropriate
        attribute for the selected node and edge candidates.

        Args
        ----
        graph: nx.DiGraph or TrackGraph
            Graph containing the node and edge candidates, the ILP will be
            solved for this graph.
        node_indicator_names: list of str
            List of types of node indicators
        edge_indicator_names: list of str
            List of types of edge indicators
        constraints_fns: list of Callable
            Each Callable should handle a single type of constraint.
            It should create the respective constraints for all affected
            objects in the graph and return them.
            Add more Callable to this list to add additional constraints.
            See tracking/constraints.py for examples.
            Interface: fn(graph, indicators) -> constraints
              graph: Create constraints for nodes/edges in this graph
              indicators: The indicator dict created by this Solver object
              constraints: list of pylp.LinearConstraint
        pin_constraints_fns: list of Callable
            Each Callable should handle a single type of pin constraint.
            Use this to add constraints to pin indicators to specific states.
            Created only if indicator has already been set by neighboring
            blocks.
            Interface: fn(graph, indicators, selected) -> constraints
              graph: Create constraints for nodes/edges in this graph
              indicators: The indicator dict created by this Solver object
              selected_key: Consider this property to determine state of
                candidate
              constraints: list of pylp.LinearConstraint
        timeout: int
            Terminate solver after this time; if terminated early solution
            might be suboptimal; set to 0 to disable (default).
        num_threads: int
            Number of threads to use for solving. Default is 1. Depending on
            the used solver and the type of problem parallelization of ILPs
            often does not scale that well, one approach is to use different
            methods or heuristics in parallel and stop as soon as one of them
            was successful.
        """

        self.graph = graph
        self.timeout = timeout
        self.num_threads = num_threads

        self.num_vars = None
        self.selected_key = None
        self.objective = None
        self.main_constraints = []  # list of LinearConstraint objects
        self.pin_constraints = []  # list of LinearConstraint objects
        self.solver = None

        self.node_indicator_names = set(node_indicator_names)
        self.edge_indicator_names = set(edge_indicator_names)

        self.constraints_fns = constraints_fns
        self.pin_constraints_fns = pin_constraints_fns

        self._create_indicators()
        self._create_solver()
        self._create_constraints()

    def update_objective(self, node_indicator_costs, edge_indicator_costs,
                         selected_key):
        """Set/Update the objective using a new set of node and edge costs.

        Notes
        -----
        Has to be called before solving.

        Args
        ----
        node_indicator_costs: dict str: Callable
            Map from (node) indicator type to Callable. The Callable will be
            executed for every indicator of the respective type. It returns
            a list of costs for that indicator. The sum of costs will be
            added as a coefficient for that indicator to the objective.
            See tracking/cost_functions.py for examples.
            Interface: fn(obj: dict[str, Number]) -> cost: list[Number]
              fn:
                Callable that takes a dict and returns a list of Numbers
                (typically a (parameterized) closure).
              obj:
                The data associated with a node or edge
              cost:
                The computed cost that will be added to the objective for
                the respective indicator
        edge_indicator_costs: dict str: Callable
            Map from (edge) indicator type to Callable. The Callable will be
            executed for every indicator of the respective type. It returns
            a list of costs for that indicator. The sum of costs will be
            added as a coefficient for that indicator to the objective.
            See tracking/cost_functions.py for examples.
            Interface: fn(obj: dict[str, Number]) -> cost: list[Number]
              fn:
                Callable that takes a dict and returns a list of Numbers
                (typically a (parameterized) closure).
              obj:
                The data associated with a node or edge
              cost:
                The computed cost that will be added to the objective for
                the respective indicator
        selected_key: str
            After solving, this property of node/edge candidates will be set
            depending on if they are part of the solution or not. In addition
            it will be passed to the pin_constraints_fns Callables.
        """
        assert (
            set(node_indicator_costs.keys()) == self.node_indicator_names and
            set(edge_indicator_costs.keys()) == self.edge_indicator_names), \
            "cannot change set of indicators during one run!"
        self.node_indicator_costs = node_indicator_costs
        self.edge_indicator_costs = edge_indicator_costs

        self.selected_key = selected_key

        self._create_objective()
        self.solver.set_objective(self.objective)

        self.pin_constraints = []
        self._add_pin_constraints()
        all_constraints = pylp.LinearConstraints()
        for c in self.main_constraints + self.pin_constraints:
            all_constraints.add(c)
        self.solver.set_constraints(all_constraints)

    def _create_solver(self):
        self.solver = pylp.LinearSolver(
                self.num_vars,
                pylp.VariableType.Binary,
                preference=pylp.Preference.Any)
        self.solver.set_num_threads(self.num_threads)
        self.solver.set_timeout(self.timeout)

    def solve(self):
        """Solves the ILP

        Notes
        -----
        Called internally by `solve_and_set`, if access to the whole
        indicator solution vector is required, call this function directly.
        """
        assert self.objective is not None, (
            "objective has to be defined before solving by calling"
            "update_objective")
        solution, message = self.solver.solve()
        logger.info(message)
        logger.info("costs of solution: %f", solution.get_value())

        return solution

    def solve_and_set(self, node_key="node_selected",
                      edge_key="edge_selected"):
        """Solves the ILP and sets the selected_key property of
        node and edge candidates according to the solution

        Args
        ----
        node_key: str
            For node candidates, check solution state of this indicator.
        edge_key: str
            For edge candidates, check solution state of this indicator.
        """
        solution = self.solve()

        for v in self.graph.nodes:
            self.graph.nodes[v][self.selected_key] = solution[
                self.indicators[node_key][v]] > 0.5

        for e in self.graph.edges:
            self.graph.edges[e][self.selected_key] = solution[
                self.indicators[edge_key][e]] > 0.5

    def _create_indicators(self):

        self.indicators = {}
        self.num_vars = 0

        for k in self.node_indicator_names:
            self.indicators[k] = {}
            for node in self.graph.nodes:
                self.indicators[k][node] = self.num_vars
                self.num_vars += 1

        for k in self.edge_indicator_names:
            self.indicators[k] = {}
            for edge in self.graph.edges():
                self.indicators[k][edge] = self.num_vars
                self.num_vars += 1

    def _create_objective(self):

        logger.debug("setting objective")

        objective = pylp.LinearObjective(self.num_vars)

        # node costs
        for k, fn in self.node_indicator_costs.items():
            for n_id, node in self.graph.nodes(data=True):
                objective.set_coefficient(self.indicators[k][n_id],
                                          sum(fn(node)))

        # edge costs
        for k, fn in self.edge_indicator_costs.items():
            for u, v, edge in self.graph.edges(data=True):
                objective.set_coefficient(self.indicators[k][(u, v)],
                                          sum(fn(edge)))

        self.objective = objective

    def _create_constraints(self):

        self.main_constraints = []

        self._add_constraints()

    def _add_pin_constraints(self):

        logger.debug("setting pin constraints: %s",
                     self.pin_constraints_fns)

        for fn in self.pin_constraints_fns:
            self.pin_constraints.extend(
                fn(self.graph, self.indicators, self.selected_key))

    def _add_constraints(self):

        logger.debug("setting constraints: %s", self.constraints_fns)

        for fn in self.constraints_fns:
            self.main_constraints.extend(fn(self.graph, self.indicators))
