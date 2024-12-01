import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from centerExtract import *
from density_segmentation import idkmap
import time
from evaluation import centerlization, eval_overall_distance, eval_sse
from plot_result import plot_every_cluster, plot_clusters
from subtraj_clustering import *
from subtraj_clustering_new import expand_dataset_seed_auto_filter_recenter
from cluster_result_info import cluster_info


if __name__ == "__main__":
    fname = 'traffic'
    # load dataset
    data_mat = io.loadmat('./datasets/TRAFFIC_trans.mat')
    data = data_mat['data']
    label = data_mat['class'][0]
    label_dic = dataset_info(data, label)
    file_name = 'TRAFFIC_auto_filter_cdtw'
    path_name = './1119TRAFFIC'
    all_points = []
    for i in range(len(data)):
        all_points.extend(data[i])
    # idk = IDKernel(n_estimators=100, max_samples=psi, iso_type='anne')
    # all_ikmap, idk_map1 = idk.fit_transform(data)
    psi_list = [16]
    pth_list = [7, 8, 9, 10, 11, 12, 13, 14]
    l = 10
    for psi in psi_list:
        print(f'===================== psi={psi} ========================')
        t1 = time.time()
        all_ikmap, idk_map1, D_idx = idkmap(data, psi=psi, t=100)
        represent_idk_all = np.array(all_ikmap, dtype=np.float64)
        represent_points = np.array(all_points)
        for pth in pth_list:
            print(f'+++++++++++++++++++++++ pth={pth} +++++++++++++++++++++')
            print('recenter')
            all_overlap, all_center = expand_dataset_seed_auto_filter_recenter(data, represent_points, idk_map1, represent_idk_all, pth=pth, l=l)
            t2 = time.time()
            print(len(all_overlap), len(all_center))
            cluster_results, centers = idx2trajs(represent_points, all_overlap, all_center, f'json_{file_name}_recenter_{psi}_{pth}', 5)
            cluster_info(f'json_{file_name}_recenter_{psi}_{pth}_cluster.json')
            # plot_overlap_array(data, represent_points, all_overlap, f'{file_name}_{pth}_{tth}')
            # plot_every_cluster(data, cluster_results, 5, f'{file_name}_{pth}_{tth}')
            plot_clusters(data, cluster_results, centers, 7, f'{file_name}_recenter_{psi}_{pth}.png')
            print("TIME COST:", t2 - t1)
            # 对聚类结果进行打分：
            # od & sse
            od_hd_score = eval_overall_distance(cluster_results, centers)
            sse_hd_score = eval_sse(cluster_results)
            print(f'OD_hd: {od_hd_score}   SSE_hd: {sse_hd_score} \n')
    
            c_cluster_results = centerlization(cluster_results, centers)
            od_hd_score = eval_overall_distance(c_cluster_results, centers)
            sse_hd_score = eval_sse(c_cluster_results)
            print(f'OD_hd: {od_hd_score}   SSE_hd: {sse_hd_score} \n')
