from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class GraphBuilder:

    def __init__(self, g=None, pos=None, dist_matrix=None):
        if g==None:
            g = nx.Graph()
        self.G = g
        self.options = {'node_size':8}
        self.pos = pos
        self.dist_matrix = dist_matrix

    @property
    def innerG(self):
        return self.G

    def copy(self):
        g = self.G.copy()
        return GraphBuilder(g, self.pos, self.dist_matrix)

    def add_optimal_edges(self, opt_tour):
        prev = None
        first = None
        edges = []
        g = self.G
        for node in opt_tour:
            inode = int(node) - 1
            if prev is None:
                first = prev = inode
                continue
            edge_tpl = (prev, inode)
            g.add_edge(*edge_tpl)
            edges.append(edge_tpl)
            prev = inode
        g.add_edge(first, prev)
        edges.append((first, prev))

        self.options['edge_color']='lightgreen'
        nx.draw(g, self.pos, **self.options)
        return edges

    def add_minimum_spanning_tree(self):
        dist_matrix = self.dist_matrix
        g = self.G
        X = np.triu(dist_matrix,0)
        Tcsr = minimum_spanning_tree(csr_matrix(X))

        edges = []
        for edge in np.argwhere(Tcsr>0):
            edge_tpl = (edge[0], edge[1])
            g.add_edge(*edge_tpl)
            edges.append(edge_tpl)

        self.options['edge_color']='lightgray'
        nx.draw(g, self.pos, **self.options)
        return edges

    def add_shortest_edges(self):
        g = self.G; dist_matrix = self.dist_matrix
        size = distance_matrix.shape[0]

        G_min = g.copy()
        for edge in np.argsort(dist_matrix, axis=None)[size:size*3]:
            #print(edge, edge % size, edge // size)
            G_min.add_edge(edge % size, edge // size)

        options['edge_color']='red'
        nx.draw(G_min, pos, **options)
    
    def add_local_shortest_edges(g, dist_matrix):
        j = 0
        G_min2 = g.copy()
        edges = []
        for i in np.argsort(dist_matrix, axis=1):
            edge_tpl = (j, i[1])
            G_min2.add_edge(*edge_tpl)
            edges.append(edge_tpl)
            
            #G_min2.add_edge(j, i[2])
            j += 1
            
        options['edge_color']='red'
        nx.draw(G_min2, pos, **options)
        return edges