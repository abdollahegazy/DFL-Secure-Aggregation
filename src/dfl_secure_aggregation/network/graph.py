import random
import json
from typing import Any
import networkx
import matplotlib.pyplot as plt
from networkx.classes.coreviews import AtlasView
import math
# seed
random.seed(42)

class Topology(networkx.Graph):
    def __init__(self):
        super().__init__()
        self.network_type = None

    def add_node(self, node_for_adding, malicious=False, **attr):
        if node_for_adding in self.nodes:
            return
        super().add_node(node_for_adding, malicious=malicious, **attr)

    def add_edge(self, u_of_edge, v_of_edge,**attr):
        super().add_edge(u_of_edge, v_of_edge,**attr)

    def get_neighbors(self, node):
        return set(self.neighbors(node))
    
    def load_from_graph(self, graph: networkx.Graph):
        for node in graph.nodes:
            self.add_node(node)
        for node1, node2 in graph.edges:
            self.add_edge(node1, node2)

    def create_random_graph(self, num_nodes, edge_density, malicious_nodes=[]):
        self.network_type = 'random'
        for i in range(num_nodes):
            is_malicious = i in malicious_nodes
            self.add_node(i, malicious=is_malicious)
        
        for node1 in range(num_nodes):
            total_deg = math.ceil(int(edge_density*num_nodes))
            total_deg -= len(self.get_neighbors(node1))
            connections = set(self.get_neighbors(node1))
            while len(connections) < total_deg:
                node2 = random.randint(0, num_nodes-1)
                if node2 != node1 and node2 not in connections:
                    connections.add(node2)
            for node2 in connections:
                self.add_edge(node1, node2)
            # print(len(self.get_neighbors(node1)))
        
    def create_small_world_graph(self, num_nodes, k, b, malicious_nodes = []):
        self.network_type = 'small_world'
        ws = networkx.watts_strogatz_graph(num_nodes, k, b)
        self.load_from_graph(ws)
        self.set_malicous(malicious_nodes)

    def create_scale_free_graph(self, num_nodes, m, malicious_nodes = []):
        self.network_type = 'scale_free'
        ba = networkx.barabasi_albert_graph(n=num_nodes, m=m)
        self.load_from_graph(ba)
        self.set_malicous(malicious_nodes)

    def save(self, filename):
        # save as json
        with open(filename, 'w') as f:
            json.dump(networkx.node_link_data(self), f,indent=4)
    
    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.directed = data['directed']
        for node in data['nodes']:
            self.add_node(node['id'], malicious=node['malicious'])
            
        for link in data['links']:
            self.add_edge(link['source'], link['target'])
    def draw(self):
        pos = None
        
        if self.network_type is None:
            raise ValueError("Network not created yet")
        if self.network_type == 'random' or self.network_type == 'scale_free':
            pos = networkx.spring_layout(self, k=0.2, iterations=100)
        elif self.network_type == 'small_world':
            pos = networkx.circular_layout(self)
        fig, ax = plt.subplots(figsize=(10, 10))
        networkx.draw(self, pos, with_labels=False, node_size=20, 
                  node_color=['r' if self.nodes[node]['malicious'] else 'b' for node in self.nodes],
                  ax=ax)
        return fig, ax

    def set_malicous(self, nodes):
        for node in nodes:
            self.nodes[node]['malicious'] = True
    
    def __getitem__(self, n: Any) -> AtlasView[Any, str, Any]:
        return super().__getitem__(n)

if __name__== '__main__':
    n=128
    topology = Topology()

    beta = .005
    k=5
    topology.create_small_world_graph(n, k, beta)

    # get all rewires
    rewires = set()
    malicious=[]
    for node in topology.nodes:
        neighbors = topology.get_neighbors(node)
        for neighbor in neighbors:
            edge = (node, neighbor) if node < neighbor else (neighbor, node)
            if abs(node%n - neighbor%n) > k and edge not in rewires:
                rewires.add(edge)
                malicious.append(node)
                break
    topology.set_malicous(malicious)
    print(malicious)
    print(len(malicious))

    pos = networkx.circular_layout(topology)
    fig, ax = plt.subplots(figsize=(10, 10))
    networkx.draw(topology, pos, with_labels=False, node_size=20, 
                  node_color=['r' if topology.nodes[node]['malicious'] else 'b' for node in topology.nodes],
                  ax=ax)
    
    plt.show()