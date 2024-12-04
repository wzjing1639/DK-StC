import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from idk import idkmap

path_name = './'

def ssd(array):
    '''
    计算数组的离差平方和
    :param array: ndarray
    :return: float
    '''
    return np.sum(np.square(array - np.average(array)))


def find_optimal_split_ssd_gradient(D, title):
    '''
    取前后五个点的均值到当前点的差作为一小段的平均梯度
    小于五个点？那别分了，全塞回去吧
    :param D:
    :param title:
    :return:
    '''
    n = len(D)
    if n < 11:
        return 0
    pre_ssd = [float('inf')] * n
    post_ssd = [float('inf')] * n
    ssd_sum_list = [float('inf')] * n
    gradient_var_list = [0.0] * n
    for k in range(n):  # 遍历所有可能的分割点，避免空数组情况
        ssd_1 = ssd(D[:k])
        ssd_2 = ssd(D[k:])
        ssd_sum = ssd_1 + ssd_2
        pre_ssd[k] = ssd_1
        post_ssd[k] = ssd_2
        ssd_sum_list[k] = ssd_sum
    for k in range(5, n - 5):
        pre = np.mean(ssd_sum_list[k - 5:k])
        post = np.mean(ssd_sum_list[k + 1:k + 6])
        gradient_var_list[k] = abs(2 * ssd_sum_list[k] - pre - post)

    best_k = np.argmax(gradient_var_list)

    # 进行相似度曲线以及ssd和曲线的绘制
    plt.figure()
    plt.plot(D)
    plt.plot([best_k, best_k], [0, max(D)])
    plt.savefig(f'{path_name}{title}_similarity_curve.png')
    fig, ax1 = plt.subplots()
    ax1.plot(pre_ssd, label='pre_ssd')
    ax1.plot(post_ssd, label='post_ssd')
    ax1.plot(ssd_sum_list, label="sum_ssd")
    ax2 = ax1.twinx()
    ax2.plot([i for i in range(5, n - 5)], gradient_var_list[5:n - 5], label='gradient_diff', color='tab:purple')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig(f'{path_name}{title}_ssd_curve.png')
    plt.close()
    return best_k


def find_optimal_split_ssd(D, title):
    n = len(D)
    if n == 0:
        print("当前相似度序列全部为0")
        return -1
    min_ssd_sum = float('inf')
    best_k = -1
    pre_ssd = []
    post_ssd = []
    ssd_sum_list = []
    for k in range(1, n):  # 遍历所有可能的分割点，避免空数组情况
        ssd_1 = ssd(D[:k])
        ssd_2 = ssd(D[k:])
        ssd_sum = ssd_1 + ssd_2
        pre_ssd.append(ssd_1)
        post_ssd.append(ssd_2)
        ssd_sum_list.append(ssd_sum)

        if ssd_sum < min_ssd_sum:
            min_ssd_sum = ssd_sum
            best_k = k
        # 如果当前元素已经是0了，就不继续分割了
        if D[k] < 1e-3:
            break
    # 进行相似度曲线以及ssd和曲线的绘制
    # plt.figure()
    # plt.plot(D)
    # plt.plot([best_k, best_k], [0, max(D)])
    # plt.savefig(f'{path_name}{title}_similarity_curve.png')
    # plt.close()
    # plt.figure()
    # plt.plot(pre_ssd, label='pre_ssd')
    # plt.plot(post_ssd, label='post_ssd')
    # plt.plot(ssd_sum_list, label="sum_ssd")
    # plt.legend()
    # plt.savefig(f'{path_name}{title}_ssd_curve.png')
    # plt.close()
    return best_k


def find_optimal_split_std(D):
    '''
    对已经排序好的相似度序列进行分割，使得分割后两部分的std之和最小
    :param D:
    :return:
    '''
    n = len(D)
    min_std_sum = float('inf')
    best_k = -1

    for k in range(1, n):  # 遍历所有可能的分割点，避免空数组情况
        std_1 = np.std(D[:k])
        std_2 = np.std(D[k:])
        std_sum = std_1 + std_2

        if std_sum < min_std_sum:
            min_std_sum = std_sum
            best_k = k
        # 如果当前元素已经是0了，就不继续分割了
        if D[k] < 1e-3:
            break
    return best_k


def plot_auto_filter(data, rep_points, cluster, peak_idx, cut_idx, sort_sim_idx, sort_sims, iter_idx, title):
    plt.figure(figsize=(10, 8), dpi=250)
    # 先把所有轨迹画出来
    for tj in data:
        plt.plot(tj.T[0], tj.T[1], alpha=0.1, color='black')
    # 删除的子轨迹（tab:orange)
    for i in sort_sim_idx[cut_idx:len(sort_sims)]:
        if i != peak_idx and cluster[i].shape[0] != 0:
            sub_traj = rep_points[sorted(cluster[i])]
            plt.plot(sub_traj[:, 0], sub_traj[:, 1], color='tab:orange')
    # 保留的子轨迹
    for i in sort_sim_idx[0:cut_idx]:
        if i != peak_idx and cluster[i].shape[0] != 0:
            sub_traj = rep_points[sorted(cluster[i])]
            plt.plot(sub_traj[:, 0], sub_traj[:, 1], color='tab:blue')
    center_traj = rep_points[sorted(cluster[peak_idx])]
    plt.plot(center_traj[:, 0], center_traj[:, 1], color='red')
    plt.savefig(f"{path_name}{title}_filter_cluster_{iter_idx}.png")
    plt.close()


def auto_filter_idk_sim(rep_idk_all, C_cluster, peak_idx, title):
    # 在原本的idk表示上计算每子轨迹与中心的相似度（不需要再次计算idk
    peak_trj_idk = rep_idk_all[C_cluster[peak_idx]]
    peak_avg = np.average(peak_trj_idk, axis=0)
    sims = []
    idx_list = []
    for i in range(len(C_cluster)):
        if C_cluster[i].shape[0] != 0:
            i_idk = rep_idk_all[C_cluster[i]]
            sim_pi = np.dot(np.average(i_idk, axis=0), peak_avg)
            sims.append(sim_pi)
            idx_list.append(i)
    print(sims)
    sort_sim_idx, sort_sims = sort_by_values(idx_list, sims)
    cut_idx = find_optimal_split_ssd(sort_sims, title)
    return cut_idx, sort_sim_idx, sort_sims


def build_subtraj_cluster(rep_points, cluster, length):
    # 将聚类结果由索引转化为轨迹
    # cluster 中含有空轨迹（保证索引正确用于占位）
    subtraj_cluster = []
    new_cluster_idx = []
    cnt = 0
    for i, point_idx in enumerate(cluster):
        if point_idx.shape[0] >= length:  # 将长度不够的轨迹删除
            subtraj = rep_points[sorted(point_idx)]
            # print(subtraj)
            subtraj_cluster.append(subtraj[:])
            new_cluster_idx.append(point_idx)
            cnt += 1
        else:
            subtraj_cluster.append(np.array([]))
            new_cluster_idx.append(np.array([]))
    # print(subtraj_cluster)
    return subtraj_cluster, new_cluster_idx, cnt


def auto_filter_dtw(subtraj_cluster, center, title):
    # subtraj_cluster: 未被过滤的子轨迹聚类
    # 计算每条子轨迹与center的距离
    dtw_distances = []
    idx_list = []
    for i, subtraj in enumerate(subtraj_cluster):
        if subtraj.shape[0] != 0:
            distance, path = fastdtw(center, subtraj, dist=euclidean)
            dtw_distances.append(distance)
            idx_list.append(i)
    sort_sim_idx, sort_sims = sort_by_values(idx_list, dtw_distances)
    cut_idx = find_optimal_split_ssd(sort_sims, title)
    return cut_idx, sort_sim_idx, sort_sims


def align_trajectories(trajectories, center):
    # print(center)
    # 删除了所有空轨迹，记录有实际子轨迹的轨迹索引
    aligned_trajectories = []
    # center_mid_idx = center.shape[0] // 2
    center_mid_idx = 0
    center_point = np.array(center_mid_idx)
    idx_list = []
    for idx, traj in enumerate(trajectories):
        # print(traj.shape)
        if traj.shape[0] != 0:
            # 计算每条轨迹中点与center的差
            offset = traj[0] - center[0]
            # 将轨迹中的所有点减去这个差值，使得中点与center重合
            aligned_traj = traj - offset
            aligned_trajectories.append(aligned_traj)
            idx_list.append(idx)

    return idx_list, aligned_trajectories


def sort_by_values(idxs, values):
    combined = list(zip(idxs, values))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)  # 升序排列
    sorted_idxs, sorted_values = zip(*sorted_combined)
    # print(sorted_idxs, sorted_values)
    return list(sorted_idxs), list(sorted_values)


def auto_filter_center_idk(subtraj_cluster, center, peak_idx, psi, t=50, title='cidk'):
    # 首先进行中心化
    idx_list, aligned_cluster = align_trajectories(subtraj_cluster, center)
    # print(len(aligned_cluster))
    # 对新的轨迹计算idkmap
    all_ikmap, idk_map1, D_idx = idkmap(aligned_cluster, psi=psi, t=t)
    # print(idk_map1.shape)
    center_idk = idk_map1[idx_list.index(peak_idx)]
    similarity = np.dot(idk_map1, center_idk).tolist()
    # print(similarity)
    sort_sim_idx, sort_sims = sort_by_values(idx_list, similarity)
    cut_idx = find_optimal_split_ssd(sort_sims, title)
    return cut_idx, sort_sim_idx, sort_sims


def auto_filter_center_dtw(subtraj_cluster, center, title):
    idx_list, aligned_cluster = align_trajectories(subtraj_cluster, center)
    dtw_distances = [0.0] * len(aligned_cluster)
    for i, subtraj in enumerate(aligned_cluster):
        if subtraj.shape[0] != 0:
            distance, path = fastdtw(center, subtraj, dist=euclidean)
            dtw_distances[i] = distance
    sort_sim_idx, sort_sims = sort_by_values(idx_list, dtw_distances)
    cut_idx = find_optimal_split_ssd(sort_sims, title)
    return cut_idx, sort_sim_idx, sort_sims


def identify_center_subtraj(point_ik_map, cluster, length):
    '''
    通过idk相似度来确定中心子轨迹的索引
    '''
    subtraj_map = []
    subtraj_idx = []
    for i, point_idxs in enumerate(cluster):
        if point_idxs.shape[0] >= length:  # 只记录非空的子轨迹（算平均）
            subtraj_map.append(np.average(point_ik_map[point_idxs], axis=0))
            subtraj_idx.append(i)

    # 计算与整个 sub-traj cluster 之间的相似度
    # print(np.array(subtraj_map).shape)
    similarities = np.dot(np.array(subtraj_map), np.average(subtraj_map, axis=0))
    # print(similarities.shape)
    center = np.argmax(similarities)
    return subtraj_idx[center]

