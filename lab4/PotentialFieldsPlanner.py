import numpy as np
import time


class PotentialFieldsPlanner(object):
    def __init__(self, planning_env):
        self.env = planning_env

    def Plan(self, start_config, goal_config):
        plan_time = time.time()

        step_size = 0.8 #movement per iteration 
        k_att = 5.0 #pull to the goal
        k_rep_base = 100.0 #push from walls 
        d0 = 6.0 #how far repulsion reaches
        goal_threshold = 1.5 #distance to goal for stopping
        max_iters = 4000 #max number iterations

        #flatten inputs into arrays
        current = np.array(start_config, dtype=float).flatten()
        goal = np.array(goal_config, dtype=float).flatten()
        plan = [current.copy()]

        for i in range(max_iters):
            ##compute attractive force toward goal
            F_att = -k_att * (current - goal)

            #initialize repulsive force to zero 
            F_rep = np.zeros(2)

            #define area to check for obstacles 
            x, y = int(round(current[0])), int(round(current[1]))
            x_min = max(int(x - d0), self.env.xlimit[0])
            x_max = min(int(x + d0), self.env.xlimit[1])
            y_min = max(int(y - d0), self.env.ylimit[0])
            y_max = min(int(y + d0), self.env.ylimit[1])

            #check each cell in area for obstacles 
            for xi in range(x_min, x_max + 1):
                for yi in range(y_min, y_max + 1):
                    #if obstacle, compute repulsive force 
                    if self.env.map[xi, yi] == 1:
                        obs = np.array([xi, yi], dtype=float)
                        diff = current - obs #vector from obstacle to current position 
                        dist = np.linalg.norm(diff) #distance to obstacle 
                        if 1e-5 < dist <= d0:
                            F_rep += k_rep_base * (1.0/dist - 1.0/d0) * (1.0/(dist**2)) * (diff / dist) #repulsive force formula 

            # combine forces and normalize
            F_total = F_att + F_rep
            F_norm = np.linalg.norm(F_total)
            #if no force, stop 
            if F_norm < 1e-5:
                print(f"Stopped early at iteration {i}: zero resultant force.")
                break

            #take a step in the direction of the force 
            direction = F_total / F_norm
            new_pos = current + step_size * direction

            # stop if we hit an obstacle
            if not self.env.state_validity_checker(new_pos.reshape(2, 1)):
                print(f"Hit obstacle at iteration {i}.")
                break

            #record new position. update current position 
            plan.append(new_pos.copy())
            current = new_pos

            #stop when close enough to goal
            if np.linalg.norm(current - goal) < goal_threshold:
                print(f"Reached goal in {i+1} iterations!")
                break

        #finalize and return 
        plan = np.array(plan)
        cost = np.sum(np.linalg.norm(np.diff(plan, axis=0), axis=1)) if len(plan) > 1 else 0.0
        plan_time = time.time() - plan_time

        print(f"Cost: {cost:.2f}")
        print(f"Planning Time: {plan_time:.2f}s")
        return plan.T
