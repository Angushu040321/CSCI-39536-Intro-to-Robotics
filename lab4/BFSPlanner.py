import numpy as np
import time
from collections import deque

class BFSPlanner(object):
    def __init__(self, planning_env):
        self.env = planning_env
        self.visited = np.zeros(self.env.map.shape)

    def Plan(self, start_config, goal_config):
        plan_time = time.time()

        start = (int(start_config[0]), int(start_config[1]))
        goal = (int(goal_config[0]), int(goal_config[1]))

        queue = deque([start])
        parent = {start: None}
        self.visited[start] = True
        state_count = 0
        found = False

        # 4-connected grid (up, down, left, right)
        moves = [(-1,0),(1,0),(0,-1),(0,1)]

        while queue:
            current = queue.popleft()
            state_count += 1

            # Goal check
            if self.env.goal_criterion(np.array(current).reshape(2,1), goal_config):
                found = True
                break

            # Explore neighbors
            for dx, dy in moves:
                nx, ny = current[0]+dx, current[1]+dy
                if (0 <= nx < self.env.map.shape[0] and
                    0 <= ny < self.env.map.shape[1] and
                    not self.visited[nx, ny] and
                    self.env.map[nx, ny] == 0):
                    self.visited[nx, ny] = True
                    parent[(nx, ny)] = current
                    queue.append((nx, ny))

        # Reconstruct path
        plan = []
        if found:
            node = current
            while node is not None:
                plan.append(np.array(node).reshape(2,1))
                node = parent[node]
            plan.reverse()

        cost = len(plan)
        plan_time = time.time() - plan_time

        print("BFS → States Expanded:", state_count)
        print("Cost:", cost)
        print("Planning Time: %.4fs" % plan_time)
        return np.hstack(plan) if plan else None
