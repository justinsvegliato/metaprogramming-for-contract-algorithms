class DirectedAcyclicGraph:
    """
    Creates a DAG from a list of Node objects and a root node

    Parameters
    ----------
    root : Node, required
        The root of the DAG

    nodes : Nodes[], required
        The list of nodes in the DAG, including the root
    """

    def __init__(self, nodes, root):
        self.nodes = nodes
        self.order = len(self.nodes)
        self.root = root
        # Checks if the input root is valid
        if self.__find_root() != self.root:
            raise ValueError("Input root node is not the root node of the DAG")
        # Checks that all nodes have a unique id
        self.__unique_id("list")
        # Checks that the structure of the DAG is valid
        self.check_structure()
        self.performance_profiles = None

    def check_structure(self):
        """
        Checks the following properties in order: a unique root exists, connectedness, no self-loops, no directed cycles
        :return: Error; else, continue
        """
        # Check for a unique root
        # Done in constructor
        # Check for connectedness
        if self.__is_disconnected(self.root):
            raise ValueError("Received an invalid directed acyclic graph: has disconnectedness")
        # Reset the traversed pointers for the nodes in nodes to False
        self.__reset_traversed()
        # Check for self-loops
        if self.__has_self_loops():
            raise ValueError("Received an invalid directed acyclic graph: contains self-loops")
        # Check for directed cycles
        if self.__is_cyclic():
            raise ValueError("Received an invalid directed acyclic graph: contains directed cycles")

    def __is_disconnected(self, node):
        """
        Checks for disconnectedness in the provided DAG by doing a DFS from the root and checks that all nodes have been
        traversed in the nodes

        :assumption: The unique root has been found and pointed to self.root
        :param: node: a Node object
        :return: True, if the DAG is disconnected; else False
        """
        node.traversed_connectedness = True
        # Hit a leaf node or trivial DAG with one root node
        if len(node.parents) == 0:
            if node == self.root:
                # Return False since trivially connected
                return False
            # Pop the call stack
            return
        else:
            # Do a recursive call on all the parents
            for parent in node.parents:
                self.__is_disconnected(parent)
        for temp_node in self.nodes:
            if not temp_node.traversed_connectedness:
                # Found a node that wasn't traversed, so the DAG is disconnected
                return True
        return False

    def __reset_traversed(self):
        """
        Reset the traversed attribute/pointer in all the nodes to False for future traversals
        """
        for node in self.nodes:
            node.traversed_connectedness = False

    def __has_self_loops(self):
        """
        Checks for self-loops in the provided DAG by doing a DFS from the root

        :assumption: The unique root has been found and pointed to self.root
                     and the DAG is connected
        :param: node: a Node object
        :return: True, if the DAG has self-loops; else False
        """
        for node in self.nodes:
            if node in node.parents:
                # Has self-loops
                return True
        # No self-loops
        return False

    def __check_cyclicity(self, node, visited, stack):
        """
        An adaptation from https://www.geeksforgeeks.org/detect-cycle-in-a-graph/#:~:text=To%20detect%20cycle%2C
        %20check%20for,a%20cycle%20in%20the%20tree to detect a cycle in the given DAG

        :param node: The current node being evaluated
        :param visited: The visited Nodes
        :param stack: The recursion stack
        :return: True, if node has been visited; else False
        """
        # Mark current node as visited and add to recursion stack
        visited[node.id] = True
        stack[node.id] = True

        # Recur for all parents if any parent is visited and in stack then graph is cyclic
        for parent in node.parents:
            if not visited[parent.id]:
                if self.__check_cyclicity(parent, visited, stack):
                    return True
            elif stack[parent.id]:
                return True

        # The node needs to be popped from recursion stack before function ends
        stack[node.id] = False
        return False

    def __is_cyclic(self):
        """
        An adaptation from https://www.geeksforgeeks.org/detect-cycle-in-a-graph/#:~:text=To%20detect%20cycle%2C
        %20check%20for,a%20cycle%20in%20the%20tree to detect a cycle in the given DAG

        :assumption: The DAG is connected
        :return: True, if the DAG has a cycle; else False
        """
        visited = [False] * (len(self.nodes) + 1)
        stack = [False] * (len(self.nodes) + 1)
        for node in self.nodes:
            if not visited[node.id]:
                if self.__check_cyclicity(node, visited, stack):
                    return True
        return False

    def __find_root(self):
        """
        An exhaustive search over the nodes is used to find possible roots

        :return: Node (root) or Error
        """
        possible_roots = []
        # This double for-loop checks to see if the node is a parent of any other nodes for potential roots
        for node in self.nodes:
            is_parent = False
            for inner_node in self.nodes:
                # Check if node is a parent of any nodes, a property not of the root
                if node in inner_node.parents:
                    is_parent = True
            if not is_parent:
                possible_roots.append(node)
        if len(possible_roots) == 1:
            # Return the only possible root
            return possible_roots[0]
        if len(possible_roots) > 1:
            raise ValueError("More than one possible root found: restructure the DAG")
        if len(possible_roots) == 0:
            raise ValueError("No possible roots found: a cycle is likely present, restructure the DAG")

    def add_node(self, node):
        """
        Adds a node to the nodes and verifies the structure of the DAG

        :param node: a Node object to be appended to the self.nodes
        :return: None
        """
        # Checks that the node has a unique id relative to nodes
        self.__unique_id("node", node)
        self.nodes.append(node)
        # Checks that adding the node doesn't ruin the DAG's structure
        self.check_structure()

    def __unique_id(self, data_type, data=None):
        """
        Checks to see if the given list or element either has all unique elements or if the element appended
        to the nodes is unique, where, if not unique, will infringe on other functions

        :param data_type: string ("list" or "node")
        :param data: Node object or None
        :return: True if valid; else, error
        """
        # Check that all nodes in the list have unique ids
        if data_type == "list":
            for node in self.nodes:
                for inner_node in self.nodes:
                    # Check that the ids are different
                    if node.id == inner_node.id:
                        # Check that the inner node and outer node are different
                        if node != inner_node:
                            raise ValueError("The same id is applied to more than one node")
            return True
        # Check that the appended node has a unique id relative to the other nodes
        elif data_type == "node":
            if data is None:
                raise ValueError("A node must be provided")
            else:
                for node in self.nodes:
                    if node.id == data.id:
                        raise ValueError("The same id is applied to more than one node")
                return True
        else:
            raise ValueError("Received an invalid data type")
