import numpy as np
import time
import sys

sys.setrecursionlimit(10000)
class IDDFSPlanner(object):
    def __init__(self, planning_env):
        self.env = planning_env
        self.visited = np.zeros(self.env.map.shape)

    def depth_limited_search(self, current, goal_config, limit, parent):
        if self.env.goal_criterion(np.array(current).reshape(2,1), goal_config):
            return True, current

        if limit <= 0:
            return False, None

        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        for dx, dy in moves:
            nx, ny = current[0]+dx, current[1]+dy
            if (0 <= nx < self.env.map.shape[0] and
                0 <= ny < self.env.map.shape[1] and
                not self.visited[nx, ny] and
                self.env.map[nx, ny] == 0):
                self.visited[nx, ny] = True
                parent[(nx, ny)] = current
                found, node = self.depth_limited_search((nx, ny), goal_config, limit-1, parent)
                if found:
                    return True, node
        return False, None

    def Plan(self, start_config, goal_config):
        plan_time = time.time()

        start = (int(start_config[0]), int(start_config[1]))
        goal = (int(goal_config[0]), int(goal_config[1]))

        depth = 0
        found = False
        parent = {}
        final_node = None
        state_count = 0

        while not found:
            self.visited = np.zeros(self.env.map.shape, dtype=bool)
            parent = {start: None}
            self.visited[start] = True
            found, final_node = self.depth_limited_search(start, goal_config, depth, parent)
            depth += 1
            state_count += np.sum(self.visited)
            if depth > 10000:  # safety cap
                break

        # reconstruct plan
        plan = []
        if found:
            node = final_node
            while node is not None:
                plan.append(np.array(node).reshape(2,1))
                node = parent[node]
            plan.reverse()

        cost = len(plan)
        plan_time = time.time() - plan_time

        print("IDDFS → States Expanded:", state_count)
        print("Depth reached:", depth)
        print("Cost:", cost)
        print("Planning Time: %.4fs" % plan_time)
        return np.hstack(plan) if plan else None
