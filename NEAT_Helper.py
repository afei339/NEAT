import numpy as np
from queue import *

from NEAT_Brain import *
from NEAT_Config import Config
config = Config()

def mutate(Network):#chooses what type of mutation the policy will undergo
    r = np.random.rand(4)
    if r[0] < config.mut_weight_p:
        weight_mutate(Network)
    if r[1] < config.mut_connection_p:
        add_connection(Network)
    if r[2] < config.mut_node_p:
        add_node(Network)
    if r[3] < config.mut_enabled_p:
        flip_enabled(Network)

def weight_mutate(Network): #add random noise to all the weights in the network 
    for i in range(len(Network.Connections)):
        Network.Connections[i].Weight += np.random.randn()*config.sigma

def add_connection(Network):#adds a connection between two nodes
    if len(Network.AvailConnections) == 0:
        return 
    InNodeID, OutNodeID = Network.AvailConnections[np.random.randint(len(Network.AvailConnections))]
    Network.AvailConnections.remove([InNodeID, OutNodeID])
    new_connection = Connection(InNodeID, OutNodeID, np.random.randn())
    Network.Connections.append(new_connection) 
    Network.get_Node(InNodeID).OutputConnections.append(OutNodeID)
    Network.get_Node(OutNodeID).InputConnections.append(InNodeID)

    update_layers(Network, Network.get_Node(OutNodeID))

#select a connection and replace it with a node
def add_node(Network):
    #randomly select a connection in the network
    selected_connection = np.random.choice(Network.Connections)
    selected_connection.enable = False
    
    #create the new node where the selected connection is 
    layer = Network.get_Node(selected_connection.InNode).Layer   
    new_node = Node(Network.currID,'HID',layer)    
    new_node.InputConnections.append(selected_connection.InNode)
    new_node.OutputConnections.append(selected_connection.OutNode)
    Network.Nodes.append(new_node)
    
    update_layers(Network, new_node)#update topological numbers

    #Add the connections with the new node to the network 
    Network.Connections.append(Connection(selected_connection.InNode, Network.currID, 1, 0))
    Network.Connections.append(Connection(Network.currID, selected_connection.OutNode, selected_connection.Weight, 0))
    
    #add the connections to the nodes on either side of the new node 
    Network.get_Node(selected_connection.OutNode).InputConnections.append(Network.currID)
    Network.get_Node(selected_connection.InNode).OutputConnections.append(Network.currID)

    #add the new connections that are possible given the new node 
    for i in Network.InNodes:
        Network.AvailConnections.append([i.NodeID, Network.currID])
    for i in Network.OutNodes:
        Network.AvailConnections.append([Network.currID, i.NodeID])

    Network.currID += 1

#changes the enabled/disabled state of a random connection in the net
def flip_enabled(Network):
    selected_connection = np.random.choice(Network.Connections)
    selected_connection.Enable = not(selected_connection.Enable)


#performs a breadth first search to update the topological number of the nodes
def update_layers(Network, initial_node):
    frontier = Queue()
    frontier.put(initial_node)
    while not frontier.empty():
        current = frontier.get()
        current.Layer += 1
        for i in current.OutputConnections:
            frontier.put(Network.get_Node(i))

