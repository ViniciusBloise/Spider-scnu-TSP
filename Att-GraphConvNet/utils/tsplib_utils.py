import numpy as np
import math

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from utils.tsplib import read_tsplib_coor, read_tsplib_opt, write_tsplib_prob

def dist_mesh(dist:np.array, mesh:list):
    size = mesh[0].shape[0]
    output = np.zeros((size,size), dtype=np.float16)

    for i in range(size**2):
        #print(mesh[0][i // size, i % size], mesh[1][i // size, i % size])
        output[i // size, i % size] = dist[mesh[0][i // size, i % size], mesh[1][i // size, i % size]]
    return output

def _test_mesh():
    mg = np.meshgrid([0,1,2],[0,1,2])
    print(mg[0].shape)
    size = 4
    a = np.ones(size, dtype=int) + np.eye(size, dtype=int) - np.eye(size, dtype=int)[:,::-1]
    print(a)
    print(mg[0][:,0].shape)
    print(a[mg[0][:,0]].shape)
    print(dist_mesh(a, mg))

#Replace tsp_source for (coor, opt)
def test_one_tsp(tsp_source, coor_buff, node_num=20, 
                    cluster_center = 0, top_k = 19, top_k_expand = 19):

    coor, opt = tsp_source
        #tsp_instance_reader(tspinstance=tsp_source, buff = coor_buff, num_node=node_num)
    coors = [coor]
    
    distA = pdist(coors[0], metric='euclidean')
    distB_raw = squareform(distA)
    distB = squareform(distA) + 10.0 * np.eye(N = node_num, M =node_num, dtype = np.float64)
    
    pre_edges = np.ones(shape = (top_k + 1, top_k + 1), dtype = np.int32) + np.eye(N = top_k + 1, M = top_k + 1)
    pre_node = np.ones(shape = (top_k + 1, ))
    
    pre_node_target = np.arange(0, top_k + 1)
    pre_node_target = np.append(pre_node_target, 0)
    pre_edge_target = np.zeros(shape = (top_k + 1, top_k + 1)) 
    pre_edge_target[pre_node_target[:-1], pre_node_target[1:]] = 1
    pre_edge_target[pre_node_target[1:], pre_node_target[:-1]] = 1

    neighbor = np.argpartition(distB, kth = top_k, axis=1)
    
    neighbor_expand = np.argpartition(distB, kth = top_k_expand, axis=1)
    Omega_w = np.zeros(shape=(node_num, ), dtype = np.int32)
    Omega = np.zeros(shape=(node_num, node_num), dtype = np.int32)
    
    edges, edges_values = [], []
    nodes, nodes_coord = [], []
    edges_target, nodes_target = [], []
    meshs = []
    num_clusters = 0
    if node_num==20:
        num_clusters_threshold = 1
    else:
        num_clusters_threshold = math.ceil((node_num / (top_k+1) ) * 5)
    all_visited = False
    
    while num_clusters < num_clusters_threshold or all_visited == False:
        if all_visited==False:
            
            cluster_center_neighbor = neighbor[cluster_center, :top_k]
            cluster_center_neighbor = np.insert(cluster_center_neighbor,
                                                0, cluster_center)
        else:
            np.random.shuffle(neighbor_expand[cluster_center, :top_k_expand])
            cluster_center_neighbor = neighbor_expand[cluster_center, :top_k]
            cluster_center_neighbor = np.insert(cluster_center_neighbor,
                                                0, cluster_center)
        
        Omega_w[cluster_center_neighbor] += 1

        # case 4
        node_coord = coors[0][cluster_center_neighbor]
        x_y_min = np.min(node_coord, axis=0)
        scale = 1.0 / np.max(np.max(node_coord, axis=0)-x_y_min)
        node_coord = node_coord - x_y_min
        node_coord *= scale
        nodes_coord.append(node_coord)

        # case 1-2
        edges.append(pre_edges)
        mesh = np.meshgrid(cluster_center_neighbor, cluster_center_neighbor)
        #print(len(mesh), mesh[0].shape)
        
        edges_value = distB_raw[mesh[0], mesh[1]] #dist_mesh(distB_raw, mesh) #distB_raw[mesh].copy()
        edges_value *= scale
        edges_values.append(edges_value)
        meshs.append(mesh)
        Omega[mesh] += 1
        #print(distB_raw.shape, edges_value.shape)
        # case 3
        nodes.append(pre_node)

        # case 5-6
        edges_target.append(pre_edge_target)
        nodes_target.append(pre_node_target[:-1])

        num_clusters += 1
        
        if 0 not in Omega_w:
            all_visited = True
        
        cluster_center = np.random.choice(np.where(Omega_w==np.min(Omega_w))[0])
    
    return edges, edges_values, nodes, nodes_coord, edges_target, nodes_target, meshs, Omega, opt

def multiprocess_write(sub_prob, meshgrid, omega, node_num = 20,
                       tsplib_name = './sample.txt', statistics = False, opt = None):
    edges_probs = np.zeros(shape = (node_num, node_num), dtype = np.float32)
    print(len(meshgrid), sub_prob.shape)
    for i in range(len(meshgrid)):
        #edges_probs[list(meshgrid[i])] += sub_prob[i, :, :, 1]
        edges_probs[meshgrid[i][0], meshgrid[i][1]] += sub_prob[i, :, :, 1]
        
    edges_probs = edges_probs / (omega + 1e-8)#[:, None]
    # normalize the probability in an instance 
    edges_probs = edges_probs + edges_probs.T
    edges_probs_norm = edges_probs/np.reshape(np.sum(edges_probs, axis=1),
                                              newshape=(node_num, 1))
    
    if statistics:
        mean_rank = 0
        for i in range(node_num-1):
            mean_rank += len(np.where(edges_probs_norm[opt[i], :]>=edges_probs_norm[opt[i], opt[i+1]])[0]) 
        mean_rank /= (node_num-1)
        
        false_negative_edge = opt[np.where(edges_probs_norm[opt[:-1], opt[1:]]<1e-5)]
        # false negative edges in an instance
        num_fne = len(false_negative_edge)
        
        greater_zero_edges = len(np.where(edges_probs_norm>1e-6)[0])
        greater_zero_edges /= node_num
        
        write_tsplib_prob(tsplib_name, edge_prob = edges_probs_norm,
                  num_node=node_num, mean=mean_rank, fnn = num_fne, greater_zero=greater_zero_edges)
    else:
        write_tsplib_prob(tsplib_name, edge_prob = edges_probs_norm,
                          num_node=node_num, mean=0, fnn = 0, greater_zero=0)
    return mean_rank

def load_prob_matrix(filename:str):
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    dim = 0
    while True:
        line = lines[i]
        i += 1
        if(line.startswith('DIMENSION:')):
            dim = int(line.split(':')[1])
            break

    prob_mtx = np.zeros((dim, dim), dtype=np.float16)

    start = i
    i = 0
    for line in lines[start:]:
        line = line.strip()
        l_arr = np.array([float(j) for j in line.split(' ')], dtype=np.float16)
        prob_mtx[i,:] = l_arr 
        i += 1
    return prob_mtx

