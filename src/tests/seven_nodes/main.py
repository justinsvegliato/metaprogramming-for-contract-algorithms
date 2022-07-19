from src.classes.directed_acyclic_graph import DirectedAcyclicGraph
from src.classes.node import Node
from src.classes.contract_program import ContractProgram
from src.profiles.generator import Generator
from src.tests.test import Test
from os.path import exists


if __name__ == "__main__":
    BUDGET = 10
    INSTANCES = 5
    TIME_LIMIT = BUDGET
    STEP_SIZE = 0.1
    QUALITY_INTERVAL = .05
    VERBOSE = False

    # Create a DAG manually for testing
    # Leaf nodes
    node_3 = Node(3, [], [], expression_type="contract")
    node_4 = Node(4, [], [], expression_type="contract")
    node_5 = Node(5, [], [], expression_type="contract")
    node_6 = Node(6, [], [], expression_type="contract")

    # Intermediate nodes
    node_1 = Node(1, [node_3, node_4], [], expression_type="contract")
    node_2 = Node(2, [node_5, node_6], [], expression_type="contract")

    # Root node
    root = Node(0, [node_1, node_2], [], expression_type="contract")

    # Add the children
    node_1.children = [root]
    node_2.children = [root]
    node_3.children = [node_1]
    node_4.children = [node_1]
    node_5.children = [node_2]
    node_6.children = [node_2]

    # Nodes
    nodes = [root, node_1, node_2, node_3, node_4, node_5, node_6]

    # Create and verify the DAG from the node list
    dag = DirectedAcyclicGraph(nodes, root)

    # Used to create the synthetic data as instances and a populous file
    generate = False
    if not exists("populous.json") or generate:
        # Initialize a generator
        generator = Generator(INSTANCES, dag, time_limit=TIME_LIMIT, step_size=STEP_SIZE, uniform_low=.05,
                              uniform_high=.9)

        # Generate the nodes' quality mappings
        nodes = generator.generate_nodes()  # Return a list of file names of the nodes

        # populate the nodes' quality mappings into one populous file
        generator.populate(nodes, "populous.json")

    # Create the program with some budget
    program = ContractProgram(dag, BUDGET, scale=10**6, decimals=3, time_interval=1)

    test = Test(program)

    # Test a random distribution on the initial allocations
    # print(test.test_initial_allocations(iterations=20, initial_is_random=True, verbose=False))

    # Test initial vs optimal expected utility and allocations
    test.find_utility_and_allocations(allocation_type="initial", initial_is_random=False)
    test.find_utility_and_allocations(allocation_type="optimal", initial_is_random=False, verbose=False)
