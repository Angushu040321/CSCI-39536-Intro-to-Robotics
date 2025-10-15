import numpy as np
from RRTTree import RRTTree
import time

class PRMPlanner(object):
    def __init__(self, planning_env, num_samples, num_neighbors):
        self.env = planning_env
        self.tree = RRTTree(self.env)  #RRTTree for storing nodes and edges
        self.num_samples = num_samples
        self.num_neighbors = num_neighbors

    def Plan(self, start_config, goal_config):
        plan_time = time.time()

        # --- Phase 1: Roadmap construction ---
        all_nodes = [start_config, goal_config]
        for _ in range(self.num_samples):
            all_nodes.append(self.env.sample())

        # Add all nodes to the "tree"
        for node in all_nodes:
            self.tree.AddVertex(node)

        # Connect each node to its nearest neighbors
        for i, node in enumerate(self.tree.vertices):
            vids, vertices = self.tree.GetNNInRad(node, np.inf)
            dists = [self.env.compute_distance(node, v) for v in vertices]
            neighbors = [vids[j] for j in np.argsort(dists)[:self.num_neighbors] if vids[j] != i]
            for n in neighbors:
                self.tree.AddEdge(i, n)
                self.tree.AddEdge(n, i)

        # --- Phase 2: Path search (list-based Dijkstra) ---
        start_id = 0
        goal_id = 1
        g_score = {i: np.inf for i in range(len(self.tree.vertices))}
        g_score[start_id] = 0
        came_from = {}
        visited = set()
        open_list = [(start_id, 0)]  # list of (node_id, cost)

        while open_list:
            # select node with lowest cost
            open_list.sort(key=lambda x: x[1])
            current, current_cost = open_list.pop(0)

            if current == goal_id:
                break
            if current in visited:
                continue
            visited.add(current)

            for neighbor_id, parent_id in self.tree.edges.items():
                if parent_id == current and neighbor_id not in visited:
                    tentative_g = g_score[current] + self.env.compute_distance(
                        self.tree.vertices[current], self.tree.vertices[neighbor_id])
                    if tentative_g < g_score[neighbor_id]:
                        g_score[neighbor_id] = tentative_g
                        came_from[neighbor_id] = current
                        open_list.append((neighbor_id, tentative_g))

        # --- Reconstruct path ---
        plan = []
        current = goal_id
        while current != start_id:
            plan.append(self.tree.vertices[current])
            current = came_from.get(current, start_id)
        plan.append(self.tree.vertices[start_id])
        plan.reverse()

        state_count = len(self.tree.vertices)
        cost = g_score.get(goal_id, 0)
        plan_time = time.time() - plan_time

        print("States Expanded: %d" % state_count)
        print("Cost: %f" % cost)
        print("Planning Time: %ss" % plan_time)

        return np.concatenate(plan, axis=1)