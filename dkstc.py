import numpy as np
from filtering import build_subtraj_cluster, auto_filter_center_dtw, identify_center_subtraj
# from plot_result import plot_auto_filter, plot_auto_filter_dis
import time


def expand_dataset_seed_auto_filter_recenter(rep_trajs, rep_points, rep_idk, rep_idk_all, pth, l, filter='center_dtw'):
    '''
    根据th来进行overlap提取
    :param rep_trajs:
    :param rep_points:
    :param rep_idk:
    :param rep_idk_all: ndarray
    :param pth: point threshold，随着迭代的过程不断下降
    :param tth: trajectory threshold, 在迭代结束之后进行子轨迹之间的相似度计算
    :return: 每个overlap中属于每条轨迹的点
    '''
    # 过滤模块耗时统计
    filter_time_cost = {
        'build_subtraj': 0.0,
        'idk_nobuild': 0.0,
        'center_idk': 0.0,
        'dtw': 0.0,
        'center_dtw': 0.0
    }
    # 首先构建每条Tr中的点所属的类别数组
    point2cluster = []
    cluster2point = [0]  # 每条轨迹的开始、结束点的索引。第i条轨迹[i:i+1)
    for i in range(len(rep_trajs)):
        point2cluster.extend([i]*rep_trajs[i].shape[0])
        cluster2point.append(cluster2point[-1]+len(rep_trajs[i]))
    # 储存结果
    all_overlaps = []
    all_centers = []
    max_iter = 20
    max_iter_point = 500
    rho = 0.1  # pth的迭代下降速度 pth < (1-rho)*init_pth
    D = np.arange(0, len(rep_points))
    for iter_idx in range(max_iter):
        # 初始化
        # print("ITER:", iter_idx)
        if len(D) < 10:
            break
        sim_matrix = np.dot(rep_idk_all[D], np.average(rep_idk_all[D], axis=0).T)
        # print(sim_matrix.shape)
        cp1 = np.argmax(sim_matrix)
        cp = D[cp1]
        C = np.array([cp])  # 当前overlap, 在expand结束之后再进行整理（吗？
        # C_cluster = [np.array([]) for i in range(len(rep_trajs))]
        peak_idx = point2cluster[cp]
        # C_cluster[point2cluster[cp]].append(cp)  # 使用rep_idk_all[C_cluster[i]]来获取第i条代表性轨迹在当前overlap中的idkmap
        D = np.delete(D, cp1)  # 通过索引删除
        # 扩张
        # 使用最近邻的相似度来作为threshold迭代的开始(先expand一个点)
        c_idk = rep_idk_all[C]
        d_idk = rep_idk_all[D]
        sim_d_c = np.dot(d_idk, np.average(c_idk, axis=0))
        cp_idx = np.argmax(sim_d_c)
        cp = D[cp_idx]
        init_pth = (1 - rho) * max(sim_d_c)
        C = np.concatenate([C, [cp]])
        D = np.delete(D, cp_idx)
        expand_cnt = 0
        while init_pth > pth:
            expand_cnt += 1
            if D.shape[0] < 5: break
            c_idk = rep_idk_all[C]
            d_idk = rep_idk_all[D]
            sim_d_c = np.dot(d_idk, np.average(c_idk, axis=0))
            # 选择大于tth*rho的所有点进行扩张
            # print("max_sim_d_c: %.2f" % max(sim_d_c))
            if max(sim_d_c) < init_pth:  #如果当前最大相似度要小于阈值，则迭代减小阈值之后继续扩张（直到init_pth<pth）
                init_pth = (1 - rho) * init_pth
                continue
            # print(np.where(sim_d_c > init_pth)[0])
            mask = np.where(sim_d_c > init_pth)[0]
            cp = D[mask]

            C = np.concatenate([C, cp])
            # print(C)
            D = np.delete(D, mask)
            '''C_cluster[point2cluster[cp]].extend(cp)'''
            # print(init_pth)
            init_pth = (1 - rho) * init_pth
        # 一轮扩张结束之后，绘制图像，并储存到all_overlap中
        if len(C) < 5:
            continue
        # 后处理：把与peak所在轨迹相似度太低的塞回去
        # 首先将C中的点分配到每条轨迹中(条件切片), 在这里一次性构建C_cluster
        C_cluster = []
        for i in range(len(rep_trajs)):
            C_cluster.append(C[np.where((cluster2point[i]<=C)&(C<cluster2point[i+1]))[0]])
            # print(C_cluster[i])
        # 增加dtw filter
        t1 = time.time()
        peak_length = C_cluster[peak_idx].shape[0]
        subtraj_cluster, new_cluster_idx, cnt = build_subtraj_cluster(rep_points, C_cluster, l)
        # print(cnt)
        if cnt != 0:
            # 重新确定聚类中心
            old_peak = peak_idx
            peak_idx = identify_center_subtraj(rep_idk_all, new_cluster_idx, l)
            # print([t.shape for t in subtraj_cluster])
            filter_time_cost['build_subtraj'] += time.time() - t1
            center = subtraj_cluster[peak_idx]
            # 中心化dtw filter
            t1 = time.time()
            cut_idx, sort_sim_idx, sort_sims = auto_filter_center_dtw(subtraj_cluster, center, f"cdtw_{iter_idx}")
            filter_time_cost['center_dtw'] += time.time() - t1         
            #plot_auto_filter_dis(rep_trajs, rep_points, C_cluster, peak_idx, cut_idx, sort_sim_idx, sort_sims, f"cdtw_{iter_idx}")

            if filter == 'center_idk'or filter == 'idk_nobuild':
                for i in range(len(C_cluster)):
                    if i not in sort_sim_idx[::cut_idx] and i != peak_idx:
                        if C_cluster[i].shape[0] != 0:
                            D = np.concatenate([D, C_cluster[i]])
                            C_cluster[i] = np.array([])
            else:
                for i in range(len(C_cluster)):
                    if i not in sort_sim_idx[cut_idx::] and i != peak_idx:
                        if C_cluster[i].shape[0] != 0:
                            D = np.concatenate([D, C_cluster[i]])
                            C_cluster[i] = np.array([])     
            peak_idx = identify_center_subtraj(rep_idk_all, C_cluster, l)       
            all_overlaps.append(C_cluster)  # 点的索引
            all_centers.append(subtraj_cluster[peak_idx])  # 实际坐标
        else:
            for i in range(len(C_cluster)):
                if C_cluster[i].shape[0] != 0:
                    D = np.concatenate([D, C_cluster[i]])
                    C_cluster[i] = np.array([])
    # print(filter_time_cost)
    return all_overlaps, all_centers