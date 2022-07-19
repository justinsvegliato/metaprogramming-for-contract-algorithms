import json
import math
import numpy as np


class Generator:
    """
    A generator to create synthetic performance profiles

    :param instances: the number of instances to be produced
    :param dag: the dag to be used for performance profile simulation
    """

    def __init__(self, instances, dag, time_limit, step_size, uniform_low, uniform_high, quality_interval=.05):
        self.instances = instances
        self.dag = dag
        self.time_limit = time_limit
        self.step_size = step_size
        self.uniform_low = uniform_low
        self.uniform_high = uniform_high
        self.quality_interval = quality_interval

    def simulate_performance_profile(self, random_number, node):
        """
        Simulates a performance profile of a contract algorithm using synthetic data

        :param node: Node object
        :param random_number: a random number from a uniform distribution with some noise
        :return: dictionary
        """
        dictionary = self.recur_build(0, node, [], {}, random_number)
        return dictionary

    def recur_build(self, depth, node, qualities, dictionary, random_number):
        """
        Used to recursively generate the synthetic quality mappings

        :param depth: The depth of the recursion
        :param node: Node object, Synthetic quality mapping for this particular node given its parents
        :param qualities: The parent qualities as inputs of the node
        :param dictionary: The dictionary being generated
        :param random_number: To produce noise in the quality mappings
        :return:
        """
        potential_parent_qualities = [format(i, '.2f') for i in np.arange(
            0, 1 + self.quality_interval, self.quality_interval).round(2)]
        if not node.parents:
            velocity = self.parent_dependent_transform(node, qualities, random_number)
            for t in np.arange(0, self.time_limit + self.step_size, self.step_size).round(1):
                # Use this function to approximate the performance profile
                dictionary[t] = 1 - math.e ** (-velocity * t)
        else:
            for quality in potential_parent_qualities:
                dictionary[quality] = {quality: {}}
                # Base Case
                if depth == len(node.parents) - 1:
                    dictionary[quality] = {}
                    # To change the quality mapping with respect to the parent qualities
                    velocity = self.parent_dependent_transform(node, qualities, random_number)
                    for t in np.arange(0, self.time_limit + self.step_size, self.step_size).round(1):
                        # Use this function to approximate the performance profile
                        dictionary[quality][t] = 1 - math.e ** (-velocity * t)
                else:
                    self.recur_build(depth + 1, node, qualities.append(quality), dictionary[quality], random_number)
        return dictionary

    @staticmethod
    def parent_dependent_transform(node, qualities, random_number):
        if qualities:
            # Get the average parent quality (this may not be what we want)
            average_parent_quality = sum(qualities) / len(node.parents)
            velocity = (10**average_parent_quality) - 1
            return velocity
        else:
            # If node has no parents
            return random_number

    @staticmethod
    def import_performance_profiles(file_name):
        """
        Imports the performance profiles as dictionary via an external JSON file.

        :return: an embedded dictionary
        """
        f = open('{}'.format(file_name), "r")
        return json.loads(f.read())

    def create_dictionary(self, node):
        """
        Creates a dictionary for one instance of the performance profiles of the DAG using synthetic data

        :param node: A node object
        :return: dictionary
        """
        dictionary = {'instances': {}}
        # Take a random value from a uniform distribution; used for nodes without parents
        c = np.random.uniform(low=self.uniform_low, high=self.uniform_high)
        for i in range(self.instances):
            # Add some noise to the random value
            c = c + abs(np.random.normal(loc=0, scale=.05))  # loc is mean; scale is st. dev.
            # Make an embedded dictionary for each instance of the node in the DAG
            dictionary_inner = self.simulate_performance_profile(c, node)
            dictionary['instances']['instance_{}'.format(i)] = dictionary_inner
        dictionary['parents'] = [parent.id for parent in node.parents]
        return dictionary

    def generate_nodes(self):
        """
        Generates instances using the DAG and number of instances required
        :return: a list of the file names of the instances stored in JSON files
        """
        nodes = []  # file names of the nodes
        # Create a finite number of unique nodes and create JSON files for each
        for (i, node) in enumerate(self.dag.nodes):
            dictionary_temp = self.create_dictionary(node)
            with open('node_{}.json'.format(i), 'w') as f:
                nodes.append('node_{}.json'.format(i))
                json.dump(dictionary_temp, f, indent=2)
                print("New JSON file created for node_{}".format(i))
        return nodes

    def populate(self, nodes, out_file):
        """
        Populates a single file with the data from the quality mappings of the node JSON files

        :param nodes: a list of file names (strings) of the JSON performance profiles to be merged
        :param out_file: the file to be populated
        :return: An embedded dictionary
        """
        with open('{}'.format(out_file), 'w') as f:
            bundle = {}
            for (i, node) in enumerate(nodes):
                bundle["node_{}".format(i)] = {}
                bundle["node_{}".format(i)]['qualities'] = {}
                bundle["node_{}".format(i)]['parents'] = {}
                # Convert the JSON file into a dictionary
                temp_dictionary = self.import_performance_profiles(node)
                for instance in temp_dictionary['instances']:
                    # Loop through all the time steps
                    recursion_dictionary = temp_dictionary['instances'][instance]
                    populate_dictionary = bundle["node_{}".format(i)]['qualities']
                    self.recur_traverse(0, self.dag.nodes[i], [], recursion_dictionary, populate_dictionary)
                bundle["node_{}".format(i)]['parents'] = temp_dictionary['parents']
            json.dump(bundle, f, indent=2)
        print("Finished populating JSON file using nodes JSON files")

    def recur_traverse(self, depth, node, qualities, dictionary, populate_dictionary):
        """
        Used to recursively traverse the synthetic quality mappings to generate the populated JSON file of quality
        mappings

        :param populate_dictionary: Dictionary to be populated
        :param depth: The depth of the recursion
        :param node: Synthetic quality mapping for this particular node given its parents
        :param qualities: The parent qualities as inputs of the node
        :param dictionary: The dictionary being generated
        :return:
        """
        # Node without parents
        if not node.parents:
            for t in dictionary:
                try:
                    # See if a list object exists
                    if not isinstance(populate_dictionary["{}".format(t)], list):
                        populate_dictionary["{}".format(t)] = []
                except KeyError:
                    populate_dictionary["{}".format(t)] = []
                populate_dictionary["{}".format(t)].append(dictionary[t])
        # Node with parents
        else:
            # Loop through layer of the parent qualities
            for parent_quality in dictionary:
                try:
                    populate_dictionary[parent_quality]
                except KeyError:
                    populate_dictionary[parent_quality] = {"{}".format(parent_quality): {}}
                # Base Case
                if depth == len(node.parents) - 1:
                    # populate_dictionary[parent_quality] = {}
                    # To change the parent_quality mapping with respect to the parent qualities
                    for t in dictionary[parent_quality]:
                        try:
                            # See if a list object exists
                            if not isinstance(populate_dictionary[parent_quality]["{}".format(t)], list):
                                populate_dictionary[parent_quality]["{}".format(t)] = []
                        except KeyError:
                            populate_dictionary[parent_quality]["{}".format(t)] = []
                        populate_dictionary[parent_quality]["{}".format(t)].append(dictionary[parent_quality][t])
                else:
                    self.recur_traverse(depth + 1, node, qualities.append(parent_quality),
                                        dictionary[parent_quality], populate_dictionary[parent_quality])
        return populate_dictionary

    @staticmethod
    def adjust_dag_with_conditionals(dag):
        """
        Changes the structure of the DAG by removing any conditional nodes and appending its parents to its children
        temporarily for generation. Note that the original structure of the DAG remains intact

        :param dag: directedAcyclicGraph Object, original version
        :return: directedAcyclicGraph Object, a trimmed version
        """
        for node in dag.nodes:
            if node.expr_type == "conditional":
                # Append its parents to the children
                # Then remove the node
                for child in node.children:
                    child.parents.extend(node.parents)
                    child.parents.remove(node)
                for parent in node.parents:
                    parent.children.extend(node.children)
                    parent.children.remove(node)
        return dag
