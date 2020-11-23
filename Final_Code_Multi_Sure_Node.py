from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
import pickle


def main():
    # defining bbn variable to create a bayesian belief network
    global machine_name
    global join_tree

    # enter 1 or 2 as per your choice
    machine_type = int(input("Choose : \n 1. Use Existing Machine \n 2. Configure a new machine \n\n"))

    # if existing machine then ask for machine name and open it and process further
    if machine_type == 1:
        machine_name = str(input("Please input your machine name: "))
        machine_name_file = '%s.sav' % machine_name

        try:
            join_tree = pickle.load(open(machine_name_file, 'rb'))
            for node in join_tree.get_bbn_nodes():
                print(node)
            check(machine_name)

            potential_func()

        except:
            print("Machine name does not exists")
            main()

    else:
        machine_name = str(input("Please input your machine name: "))

        globals()['machine_%s' % machine_name] = Bbn()

        # create the nodes
        create_bbn_nodes()

        # create the network structure by edges and nodes
        create_bbn_edges()

        join_tree = InferenceController.apply(globals()['machine_%s' % machine_name])

        filename = '%s.sav' % machine_name
        pickle.dump(join_tree, open(filename, 'wb'))

        print(globals()['machine_%s' % machine_name])

        check(machine_name)

        potential_func()


def create_bbn_nodes():
    #    globals()['machine_%s' %machine_name]
    global num_of_nodes

    try:
        num_of_nodes = int(input("How many nodes your network has: "))

        if num_of_nodes > 1:

            for i in range(num_of_nodes):
                node_id = int(input("Please input the node ID: "))
                node_name = str(input("Please input the node name: "))
                node_values_input = input(
                    "Please input the node values (seperated by comma, e.g. low,high or on,off): ")
                node_values = node_values_input.split(",")
                prob_array_input = input("Please input the probability array (seperated by comma, e.g. 0.5,0.5): ")
                prob_array_str = prob_array_input.split(",")
                prob_array = []

                for x in prob_array_str:
                    prob_array.append(float(x))

                globals()['%s' % node_name] = BbnNode(Variable(node_id, str(node_name), node_values), prob_array)

                globals()['machine_%s' % machine_name] = globals()['machine_%s' % machine_name].add_node(
                    globals()['%s' % node_name])

        else:
            print("Number of nodes should be atleast 2")
            create_bbn_nodes()
    except:
        print("Name of nodes is out of range")
        create_bbn_nodes()


def create_bbn_edges():
    #    globals()['machine_%s' %machine_name]
    global num_of_edges
    global join_tree

    try:
        num_of_edges = int(input("How many edges your network has: "))

    except:
        print("Number of edges should be integer")
        create_bbn_edges()

    if num_of_edges < ((num_of_nodes * (num_of_nodes - 1)) / 2) + 1:

        add_bbn_nodes()

    else:
        print("Number of edges is out of range")
        create_bbn_edges()


"""  

"""


def add_bbn_nodes():
    global num_of_edges
    global join_tree

    try:
        for i in range(num_of_edges):
            edges_name_input = input("Please input the edge name (seperated by comma, e.g. P1,C1): ")
            edges_str_list = edges_name_input.split(",")
            globals()['machine_%s' % machine_name] = globals()['machine_%s' % machine_name].add_edge(
                Edge(globals()['%s' % edges_str_list[0]], globals()['%s' % edges_str_list[1]], EdgeType.DIRECTED))

    except:
        print("Name of edges is out of range")
        add_bbn_nodes()


def check(machine_name):
    global num_of_nodes
    # print(globals()['machine_%s' %machine_name])
    sure_node_func()


def sure_node_func():
    global node
    global sure_node
    global machine_name
    global join_tree
    global num_of_nodes
    global num_of_sure_nodes
    global node_name_list

    node_name_list = []
    for node in join_tree.get_bbn_nodes():
        node_name_list.append(node.variable.name)

    node_values_list = []
    for node in join_tree.get_bbn_nodes():
        node_values_list.append(node.variable.values)

    try:
        num_of_sure_nodes = int(input("How many nodes you're sure of: "))

    except:
        print("Number of sure nodes should be integer")
        sure_node_func()
    try:
        loop()

    except:
        print("Number of sure nodes is out of range")
        sure_node_func()


def loop():
    global node_name_list
    global sure_node
    global num_of_sure_nodes

    for i in range(num_of_sure_nodes):
        sure_node = input("Which node you're sure of: ")
        if sure_node in node_name_list:
            sure_val_func()

        else:
            print("Node name is not correct.")
            loop()


def sure_val_func():
    global sure_node
    global sure_val

    try:

        sure_val = input("What is the current value of node %s: " % sure_node)

        # insert an observation evidence
        ev = EvidenceBuilder()
        ev = ev.with_node(join_tree.get_bbn_node_by_name(str(sure_node)))
        ev = ev.with_evidence(str(sure_val), 1.0)
        ev = ev.build()
        join_tree.set_observation(ev)

        # print the marginal probabilities

    except:
        print("Node value is not correct.")
        sure_val_func()


def potential_func():
    global join_tree

    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print(potential)
        print("")


if __name__ == '__main__':
    main()