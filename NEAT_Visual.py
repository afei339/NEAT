import matplotlib.pyplot as plt
import pickle
import numpy as np

from NEAT_Config import Config
config = Config()

with open('Champion_Network.pk1', 'rb') as input:
    network = pickle.load(input)

xpos = np.zeros(len(network.Nodes))
ypos = np.zeros(len(network.Nodes))

hidden = network.Nodes[config.s_size + config.a_size:]
hidden.sort(key=lambda x: x.Layer)

max_width = 0
curr_width = 0
for i in range(len(hidden)):
    curr_layer = 1
    if hidden[i].Layer == curr_layer:
        curr_width += 1
    else:
        curr_layer = hidden[i].Layer
        if curr_width > max_width:
            max_width == curr_width
        curr_width = 0

max_width = max([curr_width, max_width, config.s_size, config.a_size])

max_depth = 0
for x in network.OutNodes:
    if x.Layer > max_depth:
        max_depth = x.Layer


tmp = np.linspace(0,max_width, config.s_size+2)
for i in range(config.s_size):
    xpos[network.InNodes[i].NodeID] = 0
    ypos[network.InNodes[i].NodeID] = tmp[i+1]

tmp = np.linspace(0,max_width, config.a_size+2)
for i in range(config.a_size):
    xpos[network.OutNodes[i].NodeID] = max_depth
    ypos[network.OutNodes[i].NodeID] = tmp[i+1]

tmp = np.linspace(0,max_width, len(hidden)+2)
for i in range(len(hidden)):
    xpos[hidden[i].NodeID] = hidden[i].Layer
    ypos[hidden[i].NodeID] = tmp[i+1]


for C in network.Connections:
    xi = xpos[C.InNode]
    yi = ypos[C.InNode]
    xf = xpos[C.OutNode]
    yf = ypos[C.OutNode]
    alpha = 1
    colour = 'g'

    if C.Weight < 0:
        colour = 'r'
    if C.Enable == False:
        alpha = 0.1

    plt.arrow(xi,yi, (xf-xi), (yf-yi), linewidth = C.Weight*2, color = colour, alpha = alpha)


plt.scatter(xpos[:config.s_size],ypos[:config.s_size],\
    s = 200, alpha = 0.6, c = 'xkcd:mango')
for i in network.InNodes:
    plt.annotate('S' + str(i.NodeID), (xpos[i.NodeID], ypos[i.NodeID]))
plt.scatter(xpos[config.s_size:config.s_size+config.a_size],ypos[config.s_size:config.s_size+config.a_size],\
    s = 200, alpha = 0.6, c = 'xkcd:deep rose')
for i in network.OutNodes:
    plt.annotate('A' + str(i.NodeID), (xpos[i.NodeID], ypos[i.NodeID]))

plt.scatter(xpos[config.s_size+config.a_size:], ypos[config.s_size+config.a_size:],\
    s = 150, alpha = 0.6, c = 'xkcd:clear blue')

plt.axis('off')
plt.xlim([-0.5, max_depth+0.5])
plt.ylim([-0.5, max_width+0.5])
plt.show()
