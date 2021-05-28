import os
import sys
import pandas as pd
from mpmath import *
import numpy as np

netfile = os.path.join("C:\\Users\\eliza\\Downloads\\TransportationNetworks-master\\TransportationNetworks-master", 'SiouxFalls', 'SiouxFalls_net.tntp')
net = pd.read_csv(netfile, skiprows=8, sep='\t')

trimmed = [s.strip().lower() for s in net.columns]
net.columns = trimmed
net.drop(['~', ';'], axis=1, inplace=True)
F = 600

mp.dps = 30
mp.pretty = False

class Network:
    def __init__(self):
        self.V = 24
        self.E = len(net.index)
        self.t = matrix(self.V)
        self.c = matrix(self.V)
        self.edges = list([0, 0] for i in range(self.E))

    def generate_network(self, v, e):
        ind = 0
        for index, row in net.iterrows():
            self.t[int(row['init_node'])-1, int(row['term_node'])-1] = int(row['free_flow_time'])
            self.c[int(row['init_node'])-1, int(row['term_node'])-1] = row['capacity']
            self.edges[ind] = [int(row['init_node'])-1, int(row['term_node'])-1]

    def dijkstra_algorithm(self, graph, start):
        dist = matrix(self.V, 2)
        for i in range(self.V):
            dist[i, 0] = sys.maxsize
            dist[i, 1] = -1
        dist[start, 0], dist[start, 1] = 0, start
        sptSet = [False]*self.V
        for cout in range(self.V):
            min = sys.maxsize
            for v in range(self.V):
                if dist[v, 0] < min and sptSet[v] == False:
                    min = dist[v, 0]
                    u = v
            sptSet[u] = True
            for v in range(self.V):
                if graph[u, v] > 0 and sptSet[v] == False and \
                dist[v, 0] > dist[u, 0] + graph[u, v]:
                    dist[v, 0] = dist[u, 0] + graph[u, v]
                    dist[v, 1] = u

        return dist

    def changed_graph(self, graph, dist, start, finish):
        c_graph = graph.copy()
        cur_node = finish
        path = np.array([cur_node])
        while cur_node != start:
            c_graph[dist[cur_node, 1], cur_node] = 0
            cur_node = int(dist[cur_node, 1])
            path = np.append(path, cur_node)
        return c_graph, path[::-1]

    def vector_x(self, n, x_edges, path):
        for i in range(len(path) - 1):
            x_edges = np.append(x_edges, [[path[i], path[i+1]]], axis=0)
        x = matrix(len(x_edges), 1)
        for i in range(len(x_edges)):
            x[i] = F / n
        return x, x_edges

    def vector_g(self, x, x_edges):
        g = matrix(len(x), 1)
        for i in range(len(x)):
            g[i] = (self.t[int(x_edges[i][0]), int(x_edges[i][1])])*(1 + 0.15 * ((x[i]/self.c[int(x_edges[i][0]), int(x_edges[i][1])])**4))
        # print("g = ", g)
        return g

    def matrix_g(self, x, x_edges):
        G = matrix(len(x))
        for i in range(len(x)):
            G[i, i] = (self.c[int(x_edges[i][0]), int(x_edges[i][1])] ** 4) / (self.t[int(x_edges[i][0]), int(x_edges[i][1])] * 0.6 * (x[i] ** 3))
        # print("G = ", G)
        return G

    def matrix_a(self, x, x_edges, start, finish):
        v_set = set()
        for row in x_edges:
            v_set.add(row[0])
            v_set.add(row[1])

        len_x = len(x)
        a = matrix(len(v_set), len_x)
        d = matrix(len(v_set), 1)

        ind = 0
        v_pos = list([-1]*self.V)

        for i in range(len_x):
            if v_pos[int(x_edges[i][0])] == -1:
                v_pos[int(x_edges[i][0])] = ind
                ind += 1
            if v_pos[int(x_edges[i][1])] == -1:
                v_pos[int(x_edges[i][1])] = ind
                ind += 1
            a[v_pos[int(x_edges[i][0])], i] = -1
            a[v_pos[int(x_edges[i][1])], i] = 1
            if x_edges[i][0] == start:
                d[v_pos[int(x_edges[i][0])]] = -F
            if x_edges[i][1] == finish:
                d[v_pos[int(x_edges[i][1])]] = F

        ind = v_pos[finish]
        res_a = matrix(len(v_set)-1, len_x)
        for i in range(ind):
            for j in range(len_x):
                res_a[i, j] = a[i, j]
        for i in range(ind+1, len(v_set)):
            for j in range(len_x):
                res_a[i-1, j] = a[i, j]
        return res_a

    def iterations(self, x, x_edges, A):
        I = eye(len(x))
        A_transposed = A.T
        while True:
            G = self.matrix_g(x, x_edges)
            g = self.vector_g(x, x_edges)
            AGAtr_inv = (A*G*A_transposed)**-1
            tmp = I - A_transposed*AGAtr_inv*A*G
            new_x = x - G*tmp*g
            if norm(new_x-x, 2) < 10**-3:
                return new_x
            x = new_x.copy()

    def algorithm(self):
        dist = sf_network.dijkstra_algorithm(sf_network.t, 7)
        c_graph, path = sf_network.changed_graph(sf_network.t, dist, 7, 12)
        x_edges = np.empty([len(path)-1, 2])
        x = matrix(len(path)-1, 1)
        for i in range(len(path) - 1):
            x_edges[i] = [path[i], path[i+1]]
            x[i] = F
        num_paths = 1
        A = self.matrix_a(x, x_edges, 7, 12)
        # print(A)
        x = self.iterations(x, x_edges, A)

        while True:
            dist = self.dijkstra_algorithm(c_graph, 7)
            if dist[12, 0] != sys.maxsize:
                c_graph, path = sf_network.changed_graph(sf_network.t, dist, 7, 12)
                num_paths += 1
                x, x_edges = self.vector_x(num_paths, x_edges, path)
                # print("x = ", x)
                # print("x edges = ", x_edges)
                # print("path = ", path)
                A = self.matrix_a(x, x_edges, 7, 12)
                # print("A = ", A)
                x = self.iterations(x, x_edges, A)
                # print("new x = ", x)
            else:
                break
        return x, x_edges


sf_network = Network()
sf_network.generate_network(sf_network.V, sf_network.E)
x, x_edges = sf_network.algorithm()
print(x, x_edges)