import numpy as np
import time

class AStarPlanner(object):    
    def __init__(self, planning_env, epsilon):
        self.env = planning_env
        self.nodes = {}
        self.epsilon = epsilon
        self.visited = np.zeros(self.env.map.shape)

    def Plan(self, start_config, goal_config):
        plan_time = time.time()

        #priority queue: stores (f_score, state)
        open_set = [(0, tuple(start_config.flatten()))] #tuple for hash
        came_from = {}
        g_score = {tuple(start_config.flatten()): 0}

        goal_tuple = tuple(goal_config.flatten())

        while open_set:
            #get node with smallest f score
            open_set.sort(key=lambda x: x[0])  #simple? priority queue
            _, current = open_set.pop(0) #pop node assigned current
            if current == goal_tuple:
                break  #found goal yay

            self.visited[current] = 1

            #more neighbor
            for neighbor in self.env.GetSuccessors(np.array(current).reshape(start_config.shape)):
                n_tuple = tuple(neighbor.flatten())
                if self.visited[n_tuple]:
                    continue

                tentative_g = g_score[current] + self.env.ComputeDistance(np.array(current), neighbor)

                if n_tuple not in g_score or tentative_g < g_score[n_tuple]:
                    came_from[n_tuple] = current
                    g_score[n_tuple] = tentative_g
                    f_score = tentative_g + self.epsilon * self.env.h(neighbor, goal_config)
                    open_set.append((f_score, n_tuple))

        #reconstruct path
        plan = [goal_tuple]
        while plan[-1] in came_from:
            plan.append(came_from[plan[-1]])
        plan = plan[::-1]  # reverse

        #convert to numpy
        plan = [np.array(p).reshape(start_config.shape) for p in plan]

        state_count = len(g_score)
        cost = g_score.get(goal_tuple, 0)
        plan_time = time.time() - plan_time

        print("States Expanded: %d" % state_count)
        print("Cost: %f" % cost)
        print("Planning Time: %ss" % plan_time)

        return np.concatenate(plan, axis=1)



