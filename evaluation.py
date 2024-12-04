'''
对子轨迹聚类的结果进行评估，直接使用类内距离之和
提供多种距离供选择，包括传统的 Frechet, DTW, Hausdorff, 以及使用相似度计算的 GDK
子轨迹聚类结果格式: [[cluster1], [cluster2]...]
[cluster]:[subtraj1,subtraj2,...]以轨迹的形式而非点集的形式出现
需要注意的是如果是用gdk来算的话要先在整个数据集上算embedding
'''
from scipy.spatial.distance import directed_hausdorff
from fastdtw import fastdtw
import numpy as np
# from dtaidistance import dtw
from scipy.spatial.distance import euclidean
from frechetdist import frdist


def hausdorff_distance(traj1, traj2):
    u = np.array(traj1)
    v = np.array(traj2)
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])


def dtw_distance(traj1, traj2):
    distance, path = fastdtw(traj1, traj2, dist=euclidean)
    return distance


def frechet_distance(traj1, traj2):
    return frdist(traj1, traj2)


def eval_sse(cluster_results, measure='hausdorff'):
    # sum of square error
    # sse = sum(mean([d(i,j)]^2))
    # 选择距离度量方法
    if measure == 'hausdorff':
        distance_function = hausdorff_distance
    elif measure == 'dtw':
        distance_function = dtw_distance
    elif measure == 'frechet':
        distance_function = frechet_distance
    else:
        raise ValueError("Unsupported measure. Use 'hausdorff', 'dtw', or 'frechet'.")

    total_distance = 0
    # 遍历每个聚类
    for cluster in cluster_results:
        n = len(cluster)
        if n == 0:
            return -1
        cluster_dis_sum = 0
        # 计算每个聚类内轨迹之间的距离
        for i in range(n):
            for j in range(i, n):
                cluster_dis_sum += distance_function(cluster[i], cluster[j])**2
        cluster_dis_mean = cluster_dis_sum / n
        total_distance += cluster_dis_mean
    return total_distance


def eval_overall_distance(cluster_results, centers, measure='hausdorff'):
    # 计算聚类中每条轨迹到代表性轨迹/中心的距离和
    # 选择距离度量方法
    if measure == 'hausdorff':
        distance_function = hausdorff_distance
    elif measure == 'dtw':
        distance_function = dtw_distance
    elif measure == 'frechet':
        distance_function = frechet_distance
    else:
        raise ValueError("Unsupported measure. Use 'hausdorff', 'dtw', or 'frechet'.")

    total_distance = 0
    # 遍历每个聚类
    for idx, cluster in enumerate(cluster_results):
        center_traj = centers[idx]
        if len(center_traj) == 0:
            continue
        n = len(cluster)
        for sub_traj in cluster:
            if len(sub_traj) == 0:
                continue
            total_distance += distance_function(center_traj, sub_traj)
    # 除以总的子轨迹数量
    total_num = sum([len(cluster) for cluster in cluster_results])
    if total_num == 0:
        return -1

    return total_distance/total_num


def centerlization(cluster_results, centers):
    # 返回根据center平移后的聚类结果
    aligned_trajectories = []
    for i, cluster in enumerate(cluster_results):
        center = centers[i]
        # 计算中心轨迹的中点
        center_midpoint = np.mean(center, axis=0)
        
        centralized_cluster = []
        for traj in cluster:
            # 计算当前轨迹的中点
            traj_midpoint = np.mean(traj, axis=0)
            # 计算平移向量
            translation_vector = center_midpoint - traj_midpoint
            # 对轨迹进行平移
            centralized_traj = [list(np.array(point) + translation_vector) for point in traj]
            centralized_cluster.append(centralized_traj)
        aligned_trajectories.append(centralized_cluster)
    return aligned_trajectories
