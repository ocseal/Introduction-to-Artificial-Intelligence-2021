# factorOperations.py
# -------------------
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


from bayesNet import Factor
import operator as op
import util
import functools

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors, joinVariable):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


def joinFactors(factors):
    """
    Question 3: Your join implementation 

    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))

    factors_list = list(factors)
    # add all conditioned and unconditioned variables to lists.
    uncond_vars = [uncond_var for factor in factors for uncond_var in factor.unconditionedVariables()]
    cond_vars = [cond_var for factor in factors for cond_var in factor.conditionedVariables()]

    # remove overlapping variables from the RHS.
    overlap_removed_cond_vars = [var for var in cond_vars if var not in uncond_vars]

    # remove duplicates
    final_cond_vars = []
    [final_cond_vars.append(var) for var in overlap_removed_cond_vars if var not in final_cond_vars]

    var_dict = factors_list[0].variableDomainsDict()
    return_factor = Factor(uncond_vars, final_cond_vars, var_dict)

    # get all assignments + set their probability by multiplying out
    for assignment in return_factor.getAllPossibleAssignmentDicts():
        initial_prob = 1
        for factor in factors:
            initial_prob *= factor.getProbability(assignment)
        return_factor.setProbability(assignment, initial_prob)
    return return_factor


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor, eliminationVariable):
        """
        Question 4: Your eliminate implementation 

        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        final_uncond_vars = factor.unconditionedVariables()
        final_uncond_vars.remove(eliminationVariable)
        return_factor = Factor(final_uncond_vars, factor.conditionedVariables(), factor.variableDomainsDict())

        for assignment in return_factor.getAllPossibleAssignmentDicts():
            initial_prob = 0

            # find all assignment dicts in original factor (before elimination)
            possible_dicts = [d for d in factor.getAllPossibleAssignmentDicts() if assignment.items() <= d.items()]

            # sum out all of them
            for dictionary in possible_dicts:
                initial_prob += factor.getProbability(dictionary)
            return_factor.setProbability(assignment, initial_prob)
        return return_factor

    return eliminate

eliminate = eliminateWithCallTracking()


def normalize(factor):
    """
    Question 5: Your normalize implementation 

    Input factor is a single factor.

    The set of conditioned variables for the normalized factor consists 
    of the input factor's conditioned variables as well as any of the 
    input factor's unconditioned variables with exactly one entry in their 
    domain.  Since there is only one entry in that variable's domain, we 
    can either assume it was assigned as evidence to have only one variable 
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables 
    with single value domains, but that is alright since we have to assign 
    variables that only have one value in their domain to that single value.

    Return a new factor where the sum of the all the probabilities in the table is 1.
    This should be a new factor, not a modification of this factor in place.

    If the sum of probabilities in the input factor is 0,
    you should return None.

    This is intended to be used at the end of a probabilistic inference query.
    Because of this, all variables that have more than one element in their 
    domain are assumed to be unconditioned.
    There are more general implementations of normalize, but we will only 
    implement this version.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    variableDomainsDict = factor.variableDomainsDict()
    for conditionedVariable in factor.conditionedVariables():
        if len(variableDomainsDict[conditionedVariable]) > 1:
            print("Factor failed normalize typecheck: ", factor)
            raise ValueError("The factor to be normalized must have only one " + \
                            "assignment of the \n" + "conditional variables, " + \
                            "so that total probability will sum to 1\n" + 
                            str(factor))


    """The set of conditioned variables for the normalized factor consists
    of the input factor's conditioned variables as well as any of the
    input factor's unconditioned variables with exactly one entry in their
    domain.  Since there is only one entry in that variable's domain, we
    can either assume it was assigned as evidence to have only one variable
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables
    with single value domains, but that is alright since we have to assign
    variables that only have one value in their domain to that single value."""

    new_cond_vars = factor.conditionedVariables()
    new_uncond_vars = factor.unconditionedVariables()

    # finding and adding unconditioned vars with 1 domain entry to conditioned vars.
    for var in factor.unconditionedVariables():
        if len(factor.variableDomainsDict()[var]) == 1:
            new_cond_vars.add(var)
            new_uncond_vars.remove(var)

    return_factor = Factor(new_uncond_vars, new_cond_vars, factor.variableDomainsDict())
    initial_sum = 0
    for assignment in factor.getAllPossibleAssignmentDicts():
        initial_sum += factor.getProbability(assignment)
    if initial_sum == 0:
        return None
    elif initial_sum == 1:
        return factor
    else:
        # modifying all probs by scaling factor to allow sum to 1
        scaling_factor = 1 / initial_sum
        for assignment in factor.getAllPossibleAssignmentDicts():
            return_factor.setProbability(assignment, factor.getProbability(assignment) * scaling_factor)

    return return_factor

