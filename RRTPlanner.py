import numpy as np
from RRTTree import RRTTree
import time

class RRTPlanner(object):

    def __init__(self, planning_env, bias = 0.05, eta = 1.0, max_iter = 10000):
        self.env = planning_env         # Map Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                # Goal Bias
        self.max_iter = max_iter        # Max Iterations
        self.eta = eta                  # Distance to extend

    def Plan(self, start_config, goal_config):

        plan_time = time.time()

        #tart with adding the start configuration to tree
        self.tree.AddVertex(start_config)

        for i in range(self.max_iter):
            x_rand = self.sample(goal_config)
            vid_near, x_near = self.tree.GetNearestVertex(x_rand)
            x_new = self.extend(x_near, x_rand)

            #if not in tree then skip
            if np.allclose(x_new, x_near):
                continue

            new_id = self.tree.AddVertex(x_new)
            self.tree.AddEdge(vid_near, new_id)

            #goal reached checker
            if self.env.compute_distance(x_new, goal_config) < self.eta:
                goal_id = self.tree.AddVertex(goal_config)
                self.tree.AddEdge(new_id, goal_id)
                break

        #path reconstruct (from the goal all the way back to start)
        plan = []
        current = goal_config
        while True:
            plan.append(current)
            vid = None
            for k, v in self.tree.edges.items():
                if np.allclose(self.tree.vertices[k], current):
                    current = self.tree.vertices[v]
                    vid = v
                    break
            if vid is None:
                break  # reached root

        plan.reverse()

        state_count = len(self.tree.vertices)
        cost = 0  # Could compute cumulative distance if needed
        plan_time = time.time() - plan_time

        print("States Expanded: %d" % state_count)
        print("Cost: %f" % cost)
        print("Planning Time: %ss" % plan_time)

        return np.concatenate(plan, axis=1)

    def extend(self, x_near, x_rand):
        # Move from x_near toward x_rand by distance eta
        direction = x_rand - x_near
        dist = np.linalg.norm(direction)

        if dist < self.eta:
            return x_rand

        return x_near + (direction / dist) * self.eta

    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()

