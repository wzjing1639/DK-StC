import json
import numpy as np
import matplotlib.pyplot as plt

def idx2trajs(rep_points, all_overlaps, all_center, filename, num_th):
    '''
    根据all_overlaps中的聚类结果（索引），转化成实际上的轨迹聚类结果（轨迹点）
    :param rep_points:
    :param all_overlaps:
    :param filename:
    :param num_th: 聚类中最少要有多少条轨迹，否则忽略该聚类
    :return:
    '''
    cluster_results = []
    centers = []
    # 步骤1和步骤2：转换索引并对轨迹点进行排序
    for tidx, cluster in enumerate(all_overlaps):
        cluster_trajs = []
        cnt = 0
        for i in range(len(cluster)):
            if cluster[i].size > 0:
                cnt += 1
        if cnt < num_th:
            continue
        for trajectory in cluster:
            # 根据索引获取实际的坐标点
            if trajectory.size > 0:
                sorted_traj = sorted(trajectory)
                actual_traj = [rep_points[idx].tolist() for idx in sorted_traj]
                cluster_trajs.append(actual_traj)
        cluster_results.append(cluster_trajs)
        centers.append(all_center[tidx].tolist())
    # 步骤3：将结果保存到文件中
    with open(f"{filename}_cluster.json", 'w') as f:
        json.dump(cluster_results, f)
    with open(f"{filename}_center.json", 'w') as f:
        json.dump(centers, f)
    # 返回结果
    return cluster_results, centers

def cluster_info(filename):
    # 从JSON文件中读取数据
    with open(filename, 'r') as f:
        cluster_results = json.load(f)

    # 计算聚类的数量
    num_clusters = len(cluster_results)

    # 初始化统计变量
    total_trajectories = 0
    total_length = 0

    # 统计每个聚类中的轨迹数量
    trajectories_per_cluster = []

    for cluster in cluster_results:
        num_trajectories_in_cluster = len(cluster)
        trajectories_per_cluster.append(num_trajectories_in_cluster)
        for trajectory in cluster:
            total_trajectories += 1
            total_length += len(trajectory)

    average_length = total_length / total_trajectories if total_trajectories != 0 else 0
    average_trajectories_per_cluster = total_trajectories / num_clusters if num_clusters != 0 else 0

    # 打印结果
    print(f"Number of clusters: {num_clusters}")
    print(f"Average number of trajectories per cluster: {average_trajectories_per_cluster}")
    print(f"Average trajectory length: {average_length}")
    return cluster_results


def plot_clusters(all_trajectories, cluster_results, centers, num, file_name='result.png'):
    num = min(num, len(cluster_results))
    # 绘制所有轨迹，设置为黑色，透明度为0.2
    plt.figure(figsize=(10, 8), dpi=250)
    for traj in all_trajectories:
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], color='black', alpha=0.1)

    # 统计每个cluster中的轨迹数量
    cluster_sizes = [(i, len(cluster)) for i, cluster in enumerate(cluster_results)]

    # 按照轨迹数量从大到小排序，并选取前num个聚类
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    selected_clusters = cluster_sizes[:num]

    # 打印选取的各个聚类中轨迹的数量
    for i, size in selected_clusters:
        print(f"Cluster {i} contains {size} trajectories")

    # 为选中的聚类绘制轨迹，每个聚类使用不同的颜色
    colors = plt.cm.get_cmap('Set2', len(selected_clusters))
    for cluster_index, (i, size) in enumerate(selected_clusters):
        cluster = cluster_results[i]
        cluster_color = colors(cluster_index)

        # 绘制每条轨迹
        for traj in cluster:
            traj = np.array(traj)
            plt.plot(traj[:, 0], traj[:, 1], color=cluster_color)
    # 绘制聚类中心轨迹
    # for cluster_index, (i, size) in enumerate(selected_clusters):
    #     cluster_color = colors(cluster_index)
    #     # 绘制聚类中心轨迹
    #     center_traj = np.array(centers[i])
    #     plt.plot(center_traj[:, 0], center_traj[:, 1], color=cluster_color, linewidth=1)
        # 获取每个聚类的质心（所有轨迹点的平均位置），用于放置标签
        all_points = np.concatenate(cluster)
        centroid = np.mean(all_points, axis=0)

        # 添加文本标签，标签内容为轨迹数量
        plt.text(centroid[0], centroid[1], str(size), fontsize=15, color='black', ha='center')

    plt.axis('off')
    plt.savefig(file_name)
    plt.show()