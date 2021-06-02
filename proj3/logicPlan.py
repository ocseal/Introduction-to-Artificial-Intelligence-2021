# logicPlan.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}


class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()


def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


A, B, C, D = Expr('A'), Expr('B'), Expr('C'), Expr('D')


def sentence1():
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    s1 = disjoin(A, B)
    s2 = ~A % disjoin(~B, C)
    s3 = disjoin(~A, ~B, C)
    return conjoin(s1, s2, s3)


def sentence2():
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    s1 = C % disjoin(B, D)
    s2 = A >> conjoin(~B, ~D)
    s3 = ~conjoin(B, ~C) >> A
    s4 = ~D >> C
    return conjoin(s1, s2, s3, s4)


def sentence3():
    """Using the symbols PacmanAlive[1], PacmanAlive[0], PacmanBorn[0], and PacmanKilled[0],
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    """
    A, B, C, D = Expr("PacmanAlive[1]"), Expr("PacmanAlive[0]"), Expr("PacmanBorn[0]"), Expr("PacmanKilled[0]")
    s1 = A % disjoin(conjoin(B, ~D), conjoin(~B, C))
    s2 = ~conjoin(B, C)
    s3 = C
    return conjoin(s1, s2, s3)


def modelToString(model):
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False"
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    cnf = to_cnf(sentence)
    return pycoSAT(cnf)


def atLeastOne(literals):
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    return disjoin(literals)


def atMostOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    exprs = []
    combs = list(itertools.combinations(literals, 2))
    for comb in combs:
        exprs.append(disjoin(~comb[0], ~comb[1]))
    return conjoin(exprs)


def exactlyOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    least = atLeastOne(literals)
    most = atMostOne(literals)
    return conjoin(least, most)


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, time = parsed
            plan[int(time)] = action
    # return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


def pacmanSuccessorStateAxioms(x, y, t, walls_grid, var_str=pacman_str):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    possibilities = []
    if not walls_grid[x][y + 1]:
        possibilities.append(PropSymbolExpr(var_str, x, y + 1, t - 1)
                             & PropSymbolExpr('South', t - 1))
    if not walls_grid[x][y - 1]:
        possibilities.append(PropSymbolExpr(var_str, x, y - 1, t - 1)
                             & PropSymbolExpr('North', t - 1))
    if not walls_grid[x + 1][y]:
        possibilities.append(PropSymbolExpr(var_str, x + 1, y, t - 1)
                             & PropSymbolExpr('West', t - 1))
    if not walls_grid[x - 1][y]:
        possibilities.append(PropSymbolExpr(var_str, x - 1, y, t - 1)
                             & PropSymbolExpr('East', t - 1))

    if not possibilities:
        return None

    return PropSymbolExpr(var_str, x, y, t) % disjoin(possibilities)


def pacmanSLAMSuccessorStateAxioms(x, y, t, walls_grid, var_str=pacman_str):
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    moved_tm1_possibilities = []
    if not walls_grid[x][y + 1]:
        moved_tm1_possibilities.append(PropSymbolExpr(var_str, x, y + 1, t - 1)
                                       & PropSymbolExpr('South', t - 1))
    if not walls_grid[x][y - 1]:
        moved_tm1_possibilities.append(PropSymbolExpr(var_str, x, y - 1, t - 1)
                                       & PropSymbolExpr('North', t - 1))
    if not walls_grid[x + 1][y]:
        moved_tm1_possibilities.append(PropSymbolExpr(var_str, x + 1, y, t - 1)
                                       & PropSymbolExpr('West', t - 1))
    if not walls_grid[x - 1][y]:
        moved_tm1_possibilities.append(PropSymbolExpr(var_str, x - 1, y, t - 1)
                                       & PropSymbolExpr('East', t - 1))

    if not moved_tm1_possibilities:
        return None

    moved_tm1_sent = conjoin(
        [~PropSymbolExpr(var_str, x, y, t - 1), ~PropSymbolExpr(wall_str, x, y), disjoin(moved_tm1_possibilities)])

    unmoved_tm1_possibilities_aux_exprs = []  # merged variables
    aux_expr_defs = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, t - 1)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, t - 1)
        unmoved_tm1_possibilities_aux_exprs.append(wall_dir_combined_literal)
        aux_expr_defs.append(wall_dir_combined_literal % wall_dir_clause)

    unmoved_tm1_sent = conjoin([
        PropSymbolExpr(var_str, x, y, t - 1),
        disjoin(unmoved_tm1_possibilities_aux_exprs)])

    return conjoin([PropSymbolExpr(var_str, x, y, t) % disjoin([moved_tm1_sent, unmoved_tm1_sent])] + aux_expr_defs)


def pacphysics_axioms(t, all_coords, non_outer_wall_coords):
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
    """
    s = []

    walls = []
    for xy in all_coords:
        walls.append(PropSymbolExpr(wall_str, xy[0], xy[1]) >> ~PropSymbolExpr(pacman_str, xy[0], xy[1], t))
    s.append(conjoin(walls))

    non_wall_literals = []
    for non_walls in non_outer_wall_coords:
        is_at = PropSymbolExpr(pacman_str, non_walls[0], non_walls[1], t)
        non_wall_literals.append(is_at)
    s.append(exactlyOne(non_wall_literals))

    action_literals = []
    for action in DIRECTIONS:
        chosen_action = PropSymbolExpr(action, t)
        action_literals.append(chosen_action)
    s.append(exactlyOne(action_literals))

    return conjoin(s)


def check_location_satisfiability(x1_y1, x0_y0, action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - problem = An instance of logicAgents.LocMapProblem
    Return:
        - a model proving whether Pacman is at (x1, y1) at time t = 1
        - a model proving whether Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    pos_t0 = PropSymbolExpr(pacman_str, x0, y0, 0)
    axioms_t0 = pacphysics_axioms(0, all_coords, non_outer_wall_coords)
    action_t0 = PropSymbolExpr(action0, 0)
    legal_axioms_t0 = allLegalSuccessorAxioms(1, walls_grid, non_outer_wall_coords)

    axioms_t1 = pacphysics_axioms(1, all_coords, non_outer_wall_coords)
    action_t1 = PropSymbolExpr(action1, 1)

    KB.append(pos_t0)
    KB.append(axioms_t0)
    KB.append(action_t0)
    KB.append(legal_axioms_t0)
    KB.append(axioms_t1)
    KB.append(action_t1)

    model1 = findModel(conjoin(conjoin(KB), PropSymbolExpr(pacman_str, x1, y1, 1)))
    model2 = findModel(conjoin(conjoin(KB), ~PropSymbolExpr(pacman_str, x1, y1, 1)))

    return model2, model1


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2),
                                        range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = ['North', 'South', 'East', 'West']
    KB = []

    pos_t0 = PropSymbolExpr(pacman_str, x0, y0, 0)
    KB.append(pos_t0)

    for t in range(50):
        nonwallpos = [PropSymbolExpr(pacman_str, xy[0], xy[1], t) for xy in non_wall_coords]
        KB.append(exactlyOne(nonwallpos))
        goal_assertion = PropSymbolExpr(pacman_str, xg, yg, t)
        model = findModel(conjoin(conjoin(KB), goal_assertion))
        if model:
            return extractActionSequence(model, actions)
        action_list = [PropSymbolExpr(action, t) for action in actions]
        KB.append(exactlyOne(action_list))
        for xy in non_wall_coords:
            KB.append(pacmanSuccessorStateAxioms(xy[0], xy[1], t+1, walls))



def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    # locations = list(filter(lambda loc : loc not in walls_list, all_coords))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = ['North', 'South', 'East', 'West']
    food_vars = [PropSymbolExpr(food_str, f[0], f[1], 0) for f in food]
    KB = []
    pos_t0 = PropSymbolExpr(pacman_str, x0, y0, 0)
    KB.append(pos_t0)
    KB.append(conjoin(food_vars))

    for t in range(50):
        nonwallpos = [PropSymbolExpr(pacman_str, xy[0], xy[1], t) for xy in non_wall_coords]
        KB.append(exactlyOne(nonwallpos))
        goal = []
        for coords in non_wall_coords:
            goal.append(~PropSymbolExpr(food_str, coords[0], coords[1], t))
        goal_assertion = conjoin(goal)
        model = findModel(conjoin(conjoin(KB), goal_assertion))
        if model:
            return extractActionSequence(model, actions)
        action_list = [PropSymbolExpr(action, t) for action in actions]
        KB.append(exactlyOne(action_list))
        for xy in non_wall_coords:
            KB.append(pacmanSuccessorStateAxioms(xy[0], xy[1], t+1, walls))
            food_present = PropSymbolExpr(food_str, xy[0], xy[1], t)
            food_t1 = PropSymbolExpr(food_str, xy[0], xy[1], t+1)
            pacman_present = PropSymbolExpr(pacman_str, xy[0], xy[1], t+1)

            KB.append(conjoin(food_present, ~pacman_present) % food_t1)


# to check whether something is true, just append it to the KB




# Helpful Debug Method
def visualize_coords(coords_list, problem):
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualize_bool_array(bool_arr, problem):
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)


def sensorAxioms(t, non_outer_wall_coords):
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, t, x + dx, y + dy)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                    PropSymbolExpr(pacman_str, x, y, t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def four_bit_percept_rules(t, percepts):
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], t)
        percept_unit_clauses.append(percept_unit_clause)  # The actual sensor readings
    return conjoin(percept_unit_clauses)


def num_adj_walls_percept_rules(t, percepts):
    """
    SLAM uses a weaker num_adj_walls sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    num_adj_walls = sum(percepts)
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t, non_outer_wall_coords):
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, t, x + dx, y + dy)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(
                combo_var % (PropSymbolExpr(pacman_str, x, y, t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t, walls_grid, non_outer_wall_coords):
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorStateAxioms(
            x, y, t, walls_grid, var_str=pacman_str)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t, walls_grid, non_outer_wall_coords):
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSLAMSuccessorStateAxioms(
            x, y, t, walls_grid, var_str=pacman_str)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def localization(problem, agent):
    '''
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    '''
    debug = False

    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))

    possible_locs_by_timestep = []
    KB = []
    for xy in all_coords:
        if xy in walls_list:
            KB.append(PropSymbolExpr(wall_str, xy[0], xy[1]))
        else:
            KB.append(~PropSymbolExpr(wall_str, xy[0], xy[1]))

    for t in range(agent.num_timesteps):
        KB.append(pacphysics_axioms(t, all_coords, non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t], t))
        KB.append(sensorAxioms(t, non_outer_wall_coords))
        percept_rules = four_bit_percept_rules(t, agent.getPercepts())
        KB.append(percept_rules)

        possible_locations_t = []
        for xy in non_outer_wall_coords:
            pacman_present = PropSymbolExpr(pacman_str, xy[0], xy[1], t)
            model1 = findModel(conjoin(conjoin(KB), pacman_present))
            model2 = findModel(conjoin(conjoin(KB), ~pacman_present))
            if model2 is False:
                possible_locations_t.append(xy)
                KB.append(pacman_present)
            if model1 is False:
                KB.append(~pacman_present)
            if model1 and model2:
                possible_locations_t.append(xy)
        possible_locs_by_timestep.append(possible_locations_t)
        agent.moveToNextState(agent.actions[t])
        KB.append(allLegalSuccessorAxioms(t+1, walls_grid, non_outer_wall_coords))

    return possible_locs_by_timestep

#  Helpful pseudocode from Project OH:


"""if model_at_xy != False :
    possible_locations_t.append((x,y))
if model_at_xy == False:
    KB.append(~pacman_pos)
if model_not_at_xy == False:
    KB.append(pacman_pos)"""

def mapping(problem, agent):
    '''
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    '''
    debug = False

    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight() + 2)] for x in range(problem.getWidth() + 2)]
    known_map_by_timestep = []

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, 0))

    for t in range(agent.num_timesteps):
        KB.append(pacphysics_axioms(t, all_coords, non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t], t))
        KB.append(sensorAxioms(t, non_outer_wall_coords))
        percept_rules = four_bit_percept_rules(t, agent.getPercepts())
        KB.append(percept_rules)

        for xy in non_outer_wall_coords:
            wall_present = PropSymbolExpr(wall_str, xy[0], xy[1])
            model1 = findModel(conjoin(conjoin(KB), wall_present))
            model2 = findModel(conjoin(conjoin(KB), ~wall_present))
            if model2 is False:
                known_map[xy[0]][xy[1]] = 1
                KB.append(wall_present)
            if model1 is False:
                known_map[xy[0]][xy[1]] = 0
                KB.append(~wall_present)
            if model1 and model2:
                known_map[xy[0]][xy[1]] = -1
        known_map_by_timestep.append(copy.deepcopy(known_map))

        agent.moveToNextState(agent.actions[t])
        KB.append(allLegalSuccessorAxioms(t+1, known_map_by_timestep[t], non_outer_wall_coords))

    return known_map_by_timestep


def slam(problem, agent):
    '''
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    '''
    debug = False

    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth() + 2), range(problem.getHeight() + 2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth() + 1), range(1, problem.getHeight() + 1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight() + 2)] for x in range(problem.getWidth() + 2)]
    known_map_by_timestep = []
    possible_locs_by_timestep = []

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, 0))

    for t in range(agent.num_timesteps):
        KB.append(pacphysics_axioms(t, all_coords, non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t], t))
        KB.append(SLAMSensorAxioms(t, non_outer_wall_coords))
        percept_rules = num_adj_walls_percept_rules(t, agent.getPercepts())
        KB.append(percept_rules)

        possible_locations_t = []

        for xy in non_outer_wall_coords:
            wall_present = PropSymbolExpr(wall_str, xy[0], xy[1])
            wall_model1 = findModel(conjoin(conjoin(KB), wall_present))
            wall_model2 = findModel(conjoin(conjoin(KB), ~wall_present))
            pacman_present = PropSymbolExpr(pacman_str, xy[0], xy[1], t)
            pm_model1 = findModel(conjoin(conjoin(KB), pacman_present))
            pm_model2 = findModel(conjoin(conjoin(KB), ~pacman_present))

            # evaluate wall models
            if wall_model2 is False:
                known_map[xy[0]][xy[1]] = 1
                KB.append(wall_present)
            if wall_model1 is False:
                known_map[xy[0]][xy[1]] = 0
                KB.append(~wall_present)
            if wall_model1 and wall_model2:
                known_map[xy[0]][xy[1]] = -1

            # evaluate pacman positional models
            if pm_model2 is False:
                possible_locations_t.append(xy)
                KB.append(pacman_present)
            if pm_model1 is False:
                KB.append(~pacman_present)
            if pm_model1 and pm_model2:
                possible_locations_t.append(xy)

        map_copy = copy.deepcopy(known_map)
        for x in map_copy:
            for k in x:
                if k == -1:
                    x[x.index(k)] = 0
        known_map_by_timestep.append(copy.deepcopy(known_map))
        possible_locs_by_timestep.append(possible_locations_t)

        agent.moveToNextState(agent.actions[t])
        KB.append(SLAMSuccessorAxioms(t+1, map_copy, non_outer_wall_coords))

    return known_map_by_timestep, possible_locs_by_timestep


# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
