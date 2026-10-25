import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


class DynamicSpectralClustering:
    def __init__(self, min_clusters, max_clusters, n_agents):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_agents = n_agents
        self.current_groups = None

    def cluster(self, state_history):
        # print("State History Shape:", state_history.shape)
        # print("Sample of State History:")
        # print(state_history[0, :, :5])  # 打印第一个批次的所有智能体的前5个状态维度
        # print("State Dimensions:", state_history.shape[2])

        # 假设 state_history 的形状为 (batch_size, n_agents, state_dim)
        batch_size, time_steps, state_dim = state_history.shape

        # 重塑状态历史以包含时间信息
        reshaped_states = state_history.view(batch_size, -1).cpu().numpy()

        # 找到最佳聚类数量
        best_n_clusters, best_labels = self._find_best_clustering(reshaped_states)

        # 将标签转换为组列表
        new_groups = [[] for _ in range(best_n_clusters)]
        for i in range(self.n_agents):
            group = best_labels[i]
            new_groups[group].append(i)

        # 移除空组
        new_groups = [group for group in new_groups if group]

        return new_groups

    def _find_best_clustering(self, data):
        best_score = -1
        best_n_clusters = self.min_clusters
        best_labels = None

        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
            labels = spectral_clustering.fit_predict(data)
            score = silhouette_score(data, labels)

            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels

        return best_n_clusters, best_labels

    def update_groups(self, state_history, stability_threshold):
        new_groups = self.cluster(state_history)

        if self.current_groups is None:
            self.current_groups = new_groups
            return True, new_groups, self.n_agents

        num_moved = self._count_moved_agents(self.current_groups, new_groups)

        if num_moved == 0:
            return False, self.current_groups, num_moved

        if num_moved / self.n_agents < stability_threshold:
            return False, self.current_groups, num_moved

        self.current_groups = new_groups
        return True, new_groups, num_moved

    def _count_moved_agents(self, old_groups, new_groups):
        old_set = {frozenset(group) for group in old_groups}
        new_set = {frozenset(group) for group in new_groups}

        if old_set == new_set:
            return 0  # 如果组成相同（忽略顺序），则没有智能体移动

        moved = 0
        old_group_map = {agent: frozenset(group) for group in old_groups for agent in group}
        new_group_map = {agent: frozenset(group) for group in new_groups for agent in group}

        for agent in range(self.n_agents):
            if old_group_map[agent] != new_group_map[agent]:
                moved += 1

        return moved