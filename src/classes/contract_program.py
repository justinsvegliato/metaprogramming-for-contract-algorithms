import copy
# import math
from itertools import permutations

import numpy as np

from src.classes.performance_profile import PerformanceProfile
from src.classes.time_allocation import TimeAllocation


class ContractProgram(PerformanceProfile):
    """
    Structures a directed-acyclic graph (DAG) as a contract program by applying a budget on a DAG of
    contract algorithms.  The edges are directed from the leaves to the root


    :param: budget : non-negative int, required
        The budget of the contract program represented as seconds

    :param: dag : DAG, required
        The DAG that the contract program inherits
    """
    STEP_SIZE = 0.1
    POPULOUS_FILE_NAME = "populous.json"

    def __init__(self, dag, budget, scale, decimals, quality_interval=.05, time_interval=1):
        PerformanceProfile.__init__(self, file_name=self.POPULOUS_FILE_NAME, time_interval=time_interval, time_limit=budget,
                                    quality_interval=quality_interval, time_step_size=self.STEP_SIZE)
        self.budget = budget
        self.dag = dag
        self.allocations = self.uniform_budget()
        self.scale = scale
        self.decimals = decimals

    @staticmethod
    def global_utility(qualities):
        """
        Gives a utility given the qualities of the parents of the current node

        :param qualities: Qualities[], required
                The qualities that were outputted for each contract algorithm in the DAG
        :return: float
        """
        return sum(qualities)
        # return math.prod(qualities)

    def global_expected_utility(self, time_allocations):
        """
        Gives the expected utility of the contract program given the performance profiles of the nodes
        (i.e., the probability distribution of each contract program's conditional performance profile) and the
        global utility

        Assumption(s): 1) A time-allocation is given to each node in the contract program

        :param time_allocations: float[], required
                The time allocations for each contract algorithm
        :return: float
        """
        probability = 1
        average_qualities = []
        # The for loop should be a breadth-first search given that the time-allocations is ordered correctly
        for (id, time) in enumerate(time_allocations):
            node = self.find_node(id)
            parent_qualities = self.find_parent_qualities(node, time_allocations, depth=0)
            qualities = self.query_quality_list_on_interval(time.time, id, parent_qualities=parent_qualities)
            average_quality = self.average_quality(qualities)
            average_qualities.append(average_quality)

            probability = probability * self.query_probability_contract_expression(average_quality, qualities)

        expected_utility = probability * self.global_utility(average_qualities)
        return expected_utility

    def find_parent_qualities(self, node, time_allocations, depth):
        """
        Returns the parent qualities given the time allocations and node

        :param: depth: The depth of the recursive call
        :param: node: Node object, finding the parent qualities of this node
        :param: time_allocations: float[] (order matters), for the entire DAG
        :return: A list of parent qualities
        """
        # Recur down the DAG
        depth += 1
        if node.parents:
            parent_qualities = []
            for parent in node.parents:
                quality = self.find_parent_qualities(parent, time_allocations, depth)
                # Reset the parent qualities for the next node
                parent_qualities.append(quality)
            if depth == 1:
                return parent_qualities
            else:
                # Return a list of parent-dependent qualities (not a leaf or root)
                quality = self.query_average_quality(node.id, time_allocations[node.id], parent_qualities)

                return quality
        # Base Case (Leaf Nodes in a functional expression)
        else:
            # Leaf Node as a trivial functional expression
            if depth == 1:
                return []
            else:
                quality = self.query_average_quality(node.id, time_allocations[node.id], [])
                return quality

    def find_node(self, node_id):
        """
        Finds the node in the node list given the id

        :param node_id: The id of the node
        :return: Node object
        """
        for node in self.dag.nodes:
            if node.id == node_id:
                return node
        raise IndexError("Node not found with given id")

    def naive_hill_climbing(self, decay=1.2, threshold=.001, verbose=False):
        """
        Does naive hill climbing search by randomly replacing a set amount of time s between two different contract
        algorithms. If the expected value of the root node of the contract algorithm increases, we commit to the
        replacement; else, we divide s by a decay rate and repeat the above until s reaches some threshold by which we
        terminate

        :param verbose: Verbose mode
        :param threshold: float, the threshold of the temperature decay during annealing
        :type decay: float, the decay rate of the temperature during annealing
        :return: A stream of optimized time allocations associated with each contract algorithm
        """
        allocation = self.budget / self.dag.order
        time_switched = allocation
        while time_switched > threshold:
            possible_local_max = []

            for permutation in permutations(self.allocations, 2):
                # Avoids exchanging time with itself
                if permutation[0].node_id == permutation[1].node_id:
                    continue
                # Make a deep copy to avoid pointers to the same list
                adjusted_allocations = copy.deepcopy(self.allocations)

                # Avoids negative time allocation
                if adjusted_allocations[permutation[0].node_id].time - time_switched < 0:
                    continue
                else:
                    adjusted_allocations[permutation[0].node_id].time = adjusted_allocations[
                        permutation[0].node_id].time - time_switched
                    adjusted_allocations[permutation[1].node_id].time = adjusted_allocations[
                        permutation[1].node_id].time + time_switched
                    if self.global_expected_utility(adjusted_allocations) > self.global_expected_utility(
                            self.allocations):
                        possible_local_max.append(adjusted_allocations)

                    temp_time_switched = time_switched
                    eu_adjusted = self.global_expected_utility(adjusted_allocations) * self.scale
                    eu_original = self.global_expected_utility(self.allocations) * self.scale
                    print_allocations = [i.time for i in adjusted_allocations]

                    # Check for rounding
                    if self.decimals is not None:
                        print_allocations = [round(i.time, self.decimals) for i in adjusted_allocations]
                        eu_adjusted = round(eu_adjusted, self.decimals)
                        eu_original = round(eu_original, self.decimals)
                        self.global_expected_utility(self.allocations) * self.scale
                        temp_time_switched = round(temp_time_switched, self.decimals)
                    if verbose:
                        print("Amount of time switched: {:<12} ==> EU(adjusted): {:<12} EU(original): {:<12} ==> "
                              "Allocations: {}".format(
                                  temp_time_switched, eu_adjusted, eu_original, print_allocations))

            # arg max here
            if possible_local_max:
                best_allocation = max([self.global_expected_utility(j) for j in possible_local_max])
                for j in possible_local_max:
                    if self.global_expected_utility(j) == best_allocation:
                        # Make a deep copy to avoid pointers to the same list
                        self.allocations = copy.deepcopy(j)
            else:
                time_switched = time_switched / decay

        return self.allocations

    def uniform_budget(self):
        """
        Partitions the budget into equal partitions relative to the order of the DAG

        :return: TimeAllocation[]
        """
        allocation = self.budget / self.dag.order  # Divide the budget into equal allocations for every contract algo
        return [TimeAllocation(allocation, node_id) for node_id in range(0, self.dag.order)]

    def random_budget(self):
        """
        Partitions the budget into random partitions such that they add to the budget using a Dirichlet distribution

        :return: TimeAllocation
        """
        allocations_array = np.random.dirichlet(np.ones(self.dag.order), size=1).squeeze()
        allocations_list = allocations_array.tolist()
        # Multiply all elements by the budget
        allocations_list = [time * self.budget for time in allocations_list]
        return [TimeAllocation(time=time, node_id=id) for (id, time) in enumerate(allocations_list)]
