import numpy as np
from scipy import io
from idk import idkmap, gdkmap
from dkstc import expand_dataset_seed_auto_filter_recenter
from evaluation import centerlization, eval_overall_distance, eval_sse
from utils import idx2trajs, cluster_info, plot_clusters


np.random.seed(610)

if __name__ == "__main__":
    fname = 'traffic'
    # load dataset
    data_mat = io.loadmat('./dataset/TRAFFIC_trans.mat')
    data = data_mat['data']
    label = data_mat['class'][0]
    file_name = 'TRAFFIC_auto_filter'
    path_name = '.'
    all_points = []
    for i in range(len(data)):
        all_points.extend(data[i])
    psi = 16
    pth = 10
    l = 10

    print('Building feature map...')
    all_ikmap, idk_map1, D_idx = idkmap(data, psi=psi, t=100)
    # all_ikmap, idk_map1 = gdkmap(data, gma1=0.0001)
    represent_idk_all = np.array(all_ikmap, dtype=np.float64)
    represent_points = np.array(all_points)
    print('Clustering...')
    all_overlap, all_center = expand_dataset_seed_auto_filter_recenter(data, represent_points, idk_map1, represent_idk_all, pth=pth, l=l)
    cluster_results, centers = idx2trajs(represent_points, all_overlap, all_center, f'json_{file_name}_recenter_{psi}_{pth}', 5)
    cluster_info(f'json_{file_name}_recenter_{psi}_{pth}_cluster.json')
    plot_clusters(data, cluster_results, centers, 7, f'{file_name}_recenter_{psi}_{pth}.png')
    # evaluation
    # od & sse
    c_cluster_results = centerlization(cluster_results, centers)
    od_hd_score = eval_overall_distance(c_cluster_results, centers)
    sse_hd_score = eval_sse(c_cluster_results)
    print(f'OD_hd: {od_hd_score}   SSE_hd: {sse_hd_score} \n')
