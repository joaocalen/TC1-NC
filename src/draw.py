import matplotlib.pyplot as plt
import networkx as nx

def draw_neural_network(input_size, hidden_size, output_size=1, show_hidden=10):
    G = nx.DiGraph()
    
    # Add nodes
    input_nodes = [f'I{i}' for i in range(input_size)]
    hidden_nodes = [f'H{i}' for i in range(hidden_size)]
    output_node = ['O']
    
    G.add_nodes_from(input_nodes, layer='input')
    G.add_nodes_from(hidden_nodes[:show_hidden], layer='hidden')
    G.add_nodes_from(hidden_nodes[-show_hidden:], layer='hidden')
    G.add_node('...', layer='hidden')
    G.add_nodes_from(output_node, layer='output')
    
    # Add edges
    for i in input_nodes:
        for h in hidden_nodes[:show_hidden] + hidden_nodes[-show_hidden:]:
            G.add_edge(i, h)
        G.add_edge(i, '...')
    
    for h in hidden_nodes[:show_hidden] + hidden_nodes[-show_hidden:]:
        G.add_edge(h, output_node[0])
    G.add_edge('...', output_node[0])
    
    # Define the position of nodes
    pos = {}
    layer_dist = 1.0
    node_dist = 1.0
    
    pos.update((node, (0, -i * node_dist)) for i, node in enumerate(input_nodes))
    pos.update((node, (layer_dist, -i * node_dist)) for i, node in enumerate(hidden_nodes[:show_hidden]))
    pos.update((node, (layer_dist, -(show_hidden + 1) * node_dist)) for node in ['...'])
    pos.update((node, (layer_dist, -(show_hidden + 1 + i) * node_dist)) for i, node in enumerate(hidden_nodes[-show_hidden:]))
    pos.update((node, (2 * layer_dist, 0)) for i, node in enumerate(output_node))
    
    # Draw the network
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=10, font_weight="bold", arrowsize=20)
    plt.title('Neural Network Architecture')
    plt.show()
