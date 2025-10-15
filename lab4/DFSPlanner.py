import numpy as np
import time

class DFSPlanner(object):
    def __init__(self, planning_env):
        self.env = planning_env
        self.visited = np.zeros(self.env.map.shape)

    def Plan(self, start_config, goal_config):
        plan_time = time.time()

        start = (int(start_config[0]), int(start_config[1]))
        goal = (int(goal_config[0]), int(goal_config[1]))

        stack = [start]
        parent = {start: None}
        self.visited[start] = True
        state_count = 0
        found = False

        moves = [(-1,0),(1,0),(0,-1),(0,1)]

        while stack:
            current = stack.pop()
            state_count += 1

            if self.env.goal_criterion(np.array(current).reshape(2,1), goal_config):
                found = True
                break

            for dx, dy in moves:
                nx, ny = current[0]+dx, current[1]+dy
                if (0 <= nx < self.env.map.shape[0] and
                    0 <= ny < self.env.map.shape[1] and
                    not self.visited[nx, ny] and
                    self.env.map[nx, ny] == 0):
                    self.visited[nx, ny] = True
                    parent[(nx, ny)] = current
                    stack.append((nx, ny))

        plan = []
        if found:
            node = current
            while node is not None:
                plan.append(np.array(node).reshape(2,1))
                node = parent[node]
            plan.reverse()

        cost = len(plan)
        plan_time = time.time() - plan_time

        print("DFS → States Expanded:", state_count)
        print("Cost:", cost)
        print("Planning Time: %.4fs" % plan_time)
        return np.hstack(plan) if plan else None
