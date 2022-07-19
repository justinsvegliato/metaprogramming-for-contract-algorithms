import json
import numpy as np


class PerformanceProfile:
    """
    A performance profile attached to a node in the DAG via an id associated with the node

    :param file_name: the file name of the JSON file of performance profiles to be used
    :param time_interval: the interval w.r.t. time to query from in the quality mapping
    :param time_limit: the time limit for each quality mapping
    :param time_step_size: the step size for each time step
    :param quality_interval: the interval w.r.t. qualities to query from in the quality mapping
    """

    def __init__(self, file_name, time_interval, time_limit, time_step_size=.1, quality_interval=.05,
                 using_genetic_algorithm=False):
        self.dictionary = self.import_quality_mappings(file_name)
        self.time_interval = time_interval
        self.quality_interval = quality_interval
        self.time_limit = time_limit
        self.time_step_size = time_step_size
        self.using_genetic_algorithm = using_genetic_algorithm

    @staticmethod
    def import_quality_mappings(file_name):
        """
        Imports the performance profiles as dictionary via an external JSON file.

        :param file_name: the name of the file with quality mappings for each node
        :return: An embedded dictionary
        """
        f = open('{}'.format(file_name), "r")
        return json.loads(f.read())

    def query_quality_list_on_interval(self, time, id, parent_qualities):
        """
        Queries the quality mapping at a specific time, using some interval to create a distribution over qualities

        :param parent_qualities: List of qualities of the parent nodes
        :param id: The node id
        :param time: The time allocation by which the contract algorithm stops
        :return: A list of qualities for node with self.id
        """
        if self.dictionary is None:
            raise ValueError("The quality mapping for this node is null")
        else:
            # ["node_{}".format(id)]: The node
            # ['qualities']: The node's quality mappings
            dictionary = self.dictionary["node_{}".format(id)]['qualities']
            # Finding node quality given the parents' qualities
            if parent_qualities:
                for parent_quality in parent_qualities:
                    parent_quality = self.round_nearest(parent_quality, step=self.quality_interval)
                    dictionary = dictionary["{:.2f}".format(parent_quality)]
            qualities = []
            # Initialize the start and end of the time interval for descritization of the prior
            start_step = (time // self.time_interval) * self.time_interval
            end_step = start_step + self.time_interval
            # Check if time is equal to limit
            if time == self.time_limit:
                start_step = ((time - self.time_interval) // self.time_interval) * self.time_interval
                end_step = start_step + self.time_interval
            # Note: interval is [start_step, end_step) or [start_step, end_step] for time at limit
            num_decimals = self.find_number_of_decimals(self.time_step_size)
            # Round to get rid of rounding error in division of time
            for t in np.arange(start_step, end_step, self.time_step_size).round(num_decimals):
                # ["{}".format(t)]: The time allocation
                qualities += dictionary["{}".format(t)]
            return qualities

    def query_probability_contract_expression(self, queried_quality, quality_list):
        """
        The performance profile (contract expression): Queries the quality mapping at a specific time given the
        previous qualities of the contract algorithm's parents

        :param node: Node object, node being evaluated
        :param quality_list: A list of qualities from query_quality_list_on_interval()
        :param queried_quality: The conditional probability of obtaining the queried quality
        :return: [0,1], the probability of getting the current_quality, given the previous qualities and time
        allocation
        """
        # Sort in ascending order
        quality_list = sorted(quality_list)
        number_in_interval = 0
        # Initialize the start and end of the quality interval for the posterior
        start_quality = (queried_quality // self.quality_interval) * self.quality_interval
        end_quality = start_quality + self.quality_interval
        # Note: interval of [start_step, end_step)
        for quality in quality_list:
            if start_quality <= quality < end_quality:
                number_in_interval += 1
        probability = number_in_interval / len(quality_list)
        return probability

    def query_average_quality(self, id, time, parent_qualities):
        """
        Queries a single, estimated quality given a time allocation and possibly has parent qualities

        :param parent_qualities: float[] (order matters), the qualities of the parents given their time allocations
        :param id: non-negative int: the id of the Node object
        :param time: TimeAllocation object, The time allocation to the node
        :return: A quality
        """
        if self.dictionary is None:
            raise ValueError("The quality mapping for this node is null")
        # For leaf nodes
        elif not parent_qualities:
            # ["node_{}".format(id)]: The node
            # ['qualities']: The node's quality mappings
            dictionary = self.dictionary["node_{}".format(id)]['qualities']

            if not self.using_genetic_algorithm:
                estimated_time = self.round_nearest(time.time, self.time_interval)
            else:
                estimated_time = self.round_nearest(time, self.time_interval)

            # Use .1f to add a trailing zero
            qualities = dictionary["{:.1f}".format(estimated_time)]
            average_quality = self.average_quality(qualities)
            return average_quality
        # For intermediate or root nodes
        else:
            dictionary = self.dictionary["node_{}".format(id)]['qualities']

            if not self.using_genetic_algorithm:
                estimated_time = self.round_nearest(time.time, self.time_interval)
            else:
                estimated_time = self.round_nearest(time, self.time_interval)

            for parent_quality in parent_qualities:
                parent_quality = self.round_nearest(parent_quality, step=self.quality_interval)
                dictionary = dictionary["{:.2f}".format(parent_quality)]
            qualities = dictionary["{:.1f}".format(estimated_time)]
            average_quality = self.average_quality(qualities)
            return average_quality

    @staticmethod
    def average_quality(qualities):
        """
        Gets the average quality over a list of qualities

        :param qualities: float[]
        :return: float
        """
        average = sum(qualities) / len(qualities)
        return average

    def query_probability_conditional_expression(self, conditional_node, time_to_conditional,
                                                 queried_quality_branches, qualities_branches):
        """
        The performance profile (conditional expression): Queries the quality mapping at a specific time given the
        previous qualities of the contract algorithm's parents

        :param conditional_node: Node object, the conditional node being evaluated
        :param quality_list: A list of qualities from query_quality_list_on_interval()
        :param queried_quality: The conditional probability of obtaining the queried quality
        :return: [0,1], the probability of getting the current_quality, given the previous qualities and time
        allocation
        """
        # Sort in ascending order
        qualities_branches[0] = sorted(qualities_branches[0])
        qualities_branches[1] = sorted(qualities_branches[1])
        found_embedded_if = False

        # Query the probability of the condition being true
        rho = self.estimate_rho()

        # TODO: implement recursion for embedded if statements
        # TODO: For now assume that only two branches exist that are contract expressions
        for child in conditional_node.children:
            # Take into account branched if statements
            if child.expr_type == "conditional":
                found_embedded_if = True

        if not found_embedded_if:
            probability = rho * self.query_probability_contract_expression(queried_quality_branches[0], qualities_branches[0]) \
                + (1 - rho) * self.query_probability_contract_expression(queried_quality_branches[1],
                                                                         qualities_branches[1])
        else:
            # TODO: Finish this later
            pass
        return probability

    @staticmethod
    def estimate_rho():
        # Assume it's constant for now
        return .4

    @staticmethod
    def calculate_tau():
        # Assume it takes constant time
        return .1

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
            # Check that none of the parents are conditional expressions
            if not self.is_conditional_node(node, "parents"):
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
            else:
                # Assumption: Node only has one parent (the conditional)
                # Skip the conditional node since no relevant mapping exists
                node = node.parents[0]
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

    @staticmethod
    def round_nearest(number, step):
        """
        Finds the nearest element with respect to the step size

        :param number: A float
        :param step: A float
        :return: A float
        """
        return round(number / step) * step

    @staticmethod
    def find_number_of_decimals(number):
        """
        Finds the number of decimals in a float
        :param number: float
        :return: int
        """
        string_number = str(number)
        return string_number[::-1].find('.')

    @staticmethod
    def is_conditional_node(node, family_type):
        if family_type == "parents":
            for parent in node.parents:
                if parent.expr_type == "conditional":
                    return True
            return False
        elif family_type == "children":
            for child in node.parents:
                if child.expr_type == "conditional":
                    return True
            return False
        else:
            raise ValueError("Invalid family_type")
