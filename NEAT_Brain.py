import numpy as np

from NEAT_Config import Config
config = Config()


class Node(object): #singular vertex of the graph
    def __init__(self, nodeID, nodeType, layer):
        self.NodeID = nodeID
        self.Type = nodeType
        self.InputConnections = []
        self.OutputConnections = []
        self.Layer = layer #keeps track of topological order that the graph must be evaluated in 

    def __repr__(self):
        return str(self.NodeID) + ': ' + str(self.Type)

class Connection(object): #connection between two nodes, defined by NodeID
    def __init__(self, InNode, OutNode, Weight, Innovation=0):
        self.InNode = InNode
        self.OutNode = OutNode
        self.Weight = Weight
        self.Enable = True
        #self.innovation = Innovation 

    def __repr__(self):
        return str(self.InNode) + '->' + str(self.OutNode)

class Network(object):#graph made up on nodes and connections that forms the neural net 
    def __init__(self):
        self.currID = 0#ID number of the node to be place, also how many nodes are in the network
        self.InNodes = []#inputs (the state)
        self.OutNodes = []#outputs (the actions)

        #add the input and output nodes 
        for _ in range(config.s_size):
            self.InNodes.append(Node(self.currID,'IN', 0))
            self.currID += 1
        for _ in range(config.a_size):
            self.OutNodes.append(Node(self.currID,'OUT', 1))
            self.currID += 1

        #the network starts off with all the nodes fully connected
        self.Connections = []
        for i in range(config.s_size):
            for j in range(config.a_size):
                #add the conenection
                self.Connections.append(Connection(self.InNodes[i].NodeID, self.OutNodes[j].NodeID,\
                    np.random.randn()/np.sqrt(config.s_size), 0))

                #the nodes also keep track of connections that affect them
                self.OutNodes[j].InputConnections.append(self.InNodes[i].NodeID)
                self.InNodes[i].OutputConnections.append(self.OutNodes[j].NodeID)
        self.Nodes = self.InNodes+self.OutNodes

        #will be used to keep track of connections that can be made in the network
        self.AvailConnections = []
    
    def get_Node(self, NodeID):#returns node object given an ID
        return self.Nodes[NodeID]

    def get_Connection(self, node1, node2):#return connection between two nodes
        for x in self.Connections:
            if x.InNode == node1 and x.OutNode == node2:
                return x
            if x.OutNode == node1 and x.InNode == node2:
                return x

    #feed forward pass of the network
    def predict(self, s):
        layered = list(self.Nodes)
        layered.sort(key=lambda x: x.Layer)#sort by topological layer
        node_values = [None]*len(layered)
        for i in range(config.s_size):
            
            node_values[i] = s[i]
        for i in range(len(s), len(layered)):
            values = []
            connections = []
            
            for NodeID in layered[i].InputConnections:
                values.append(node_values[NodeID])
                con = self.get_Connection(NodeID, layered[i].NodeID)
                if con.Enable:
                    connections.append(float(con.Weight))
                else:
                    connections.append(0)
            node_values[layered[i].NodeID] = np.dot(values,connections)
       

        output_nodes = []
        output_values = []
        for i in range(len(layered)):
            if layered[i].Type == 'OUT':
                output_nodes.append(layered[i])

        for i in range(config.s_size, config.a_size + config.s_size):
            output_values.append(node_values[i])

        return output_values

    #plays a game until completion 
    def playthrough(self, env):
        s = env.reset()
        total_reward = 0
        while True:
            a = self.predict(s.flatten())
            a = np.argmax(a)
            s, reward, done, _ = env.step(a)

            total_reward = reward

            if done:
                return total_reward
