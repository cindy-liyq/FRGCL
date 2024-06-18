import numpy as np
import pandas as pd




def process_graph_sh(edge_data,num_a,num_b):
    edge_data = edge_data.tolist()
    graph_onehot = np.zeros((num_a,num_b))
    for edge in edge_data:
        if edge[0]<390:
            if num_a==805:
                edge[0] = edge[0]-390
            if num_b==805:
                edge[1] = edge[1]-390
            graph_onehot[edge[0]][edge[1]] = 1
        else:
            break
    return graph_onehot


def process_graph_ss_hh(edge_data,num_a,num_b):
    edge_data = edge_data.tolist()
    graph_onehot = np.zeros((num_a,num_b))

    for edge in edge_data:
        if num_a==805:
            edge[0] = edge[0]-390
        if num_b==805:
            edge[1] = edge[1]-390
        graph_onehot[edge[0]][edge[1]] = 1
    return graph_onehot

def save_onehot(data,filename):
    save_data = pd.DataFrame(data)
    save_data.to_csv(filename)

if __name__ == '__main__':
    sh_edge = np.load('./KDHR/sh_graph.npy')
    sh_onehot = process_graph_sh(sh_edge,390,805)
    save_onehot(sh_onehot,'./KDHR/sh_train_onehot.csv')

    hh_edge = np.load('./KDHR/hh_graph.npy')
    hh_onehot = process_graph_ss_hh(hh_edge, 805, 805)
    save_onehot(hh_onehot, './KDHR/herbPair_onehot.csv')

    ss_edge = np.load('./KDHR/ss_graph.npy')
    ss_onehot = process_graph_ss_hh(ss_edge, 390, 390)
    save_onehot(ss_onehot, './KDHR/symPair_onehot.csv')