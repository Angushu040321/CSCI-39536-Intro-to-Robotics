import numpy as np
import time


class BugPlanner:
    def __init__(self, planning_env, step_size=1):
        self.env = planning_env
        self.step_size = step_size

   #generate m-line using bresenham's algorithm 
    def _bresenham(self, start, goal):
        x0, y0 = int(round(start[0])), int(round(start[1])) #grab x and y of start goal 
        x1, y1 = int(round(goal[0])), int(round(goal[1])) #grab x and y of goal 

        points = []
        dx = abs(x1 - x0) #difference in x's
        dy = abs(y1 - y0) #difference in y's
        sx = 1 if x0 < x1 else -1 #which way to step in x direction
        sy = 1 if y0 < y1 else -1 #which way to step in y direction 
        err = dx - dy #error value (how far we're off from perfect line)

        while True:
            points.append((x0, y0)) #add current cell to list 
            if x0 == x1 and y0 == y1: #reach goal 
                break
            e2 = 2 * err #double error to compare w.o fractions
            if e2 > -dy: #error in y is too big
                err -= dy #adjust
                x0 += sx #step in x direction 
            if e2 < dx: #error in x is too big
                err += dx #adjust
                y0 += sy #step in y direction
        return points

#check if point is on m-line 
    def _on_mline(self, point, start, goal, threshold=1.0):
        start_to_goal = np.array(goal) - np.array(start) #vector from start to goal 
        start_to_point = np.array(point) - np.array(start) #vector from start to point
        cross = np.cross(start_to_goal, start_to_point) #cross product of the two vectors
        dist_to_line = abs(cross) / np.linalg.norm(start_to_goal) #distance from point to line
        return dist_to_line < threshold #if true, close enough to the line

    #check if theres a clear line of sight to the goal
    def _has_clear_line_to_goal(self, current, goal):
        if not self.env.edge_validity_checker(
            np.array(current).reshape(2, 1),
            np.array(goal).reshape(2, 1),
        ):
            return False #false if obstacle in the way 

            #check points along line to be sure they are valid 
        steps = int(np.ceil(np.linalg.norm(np.array(current) - np.array(goal)))) 
        x_vals = np.linspace(current[0], goal[0], steps)
        y_vals = np.linspace(current[1], goal[1], steps)
        for (x, y) in zip(x_vals, y_vals):
            if not self.env.state_validity_checker(np.array([x, y]).reshape(2, 1)):
                return False
        return True #clear path to goal 

#returns the next valid move on wall
    def _get_next_wall_move(self, prev, current):
    
    #check all 8 directions
        directions = [
            np.array([1, 0]), np.array([1, 1]), np.array([0, 1]),
            np.array([-1, 1]), np.array([-1, 0]), np.array([-1, -1]),
            np.array([0, -1]), np.array([1, -1])
        ]

        #determine movement direction
        move_dir = np.array(current) - np.array(prev)
        if np.linalg.norm(move_dir) == 0: #default to right if no movement
            move_dir = np.array([1, 0])

        # pick current direction index
        idx = np.argmin([np.linalg.norm(move_dir - d / np.linalg.norm(d)) for d in directions]) 

            #try directions in clockwise order 
        for offset in range(len(directions)):
            new_dir = directions[(idx + offset) % len(directions)]
            candidate = np.round(np.array(current) + self.step_size * new_dir).astype(int)

            # skip if out of bounds
            if not (self.env.xlimit[0] <= candidate[0] <= self.env.xlimit[1] and
                    self.env.ylimit[0] <= candidate[1] <= self.env.ylimit[1]):
                continue

            # check obstacle
            if self.env.state_validity_checker(candidate.reshape(2, 1)):
                return candidate

        # fallback random nudge to get unstuck 
        rand_dir = directions[np.random.randint(0, len(directions))]
        return np.round(np.array(current) + self.step_size * rand_dir).astype(int)

 #main planning algorithm 
    def Plan(self, start, goal):
        t0 = time.time()
        start = np.array(start).flatten() #make sure start and goal 1d arrays
        goal = np.array(goal).flatten()

        current = start.copy() #current position 
        prev = start.copy() #previous position 
        plan = [current.copy()] #list of points in path 
        mode = "mline"
        hit_point = None 

        mline_points = self._bresenham(start, goal) #get m-line points
        goal_threshold = 0.5 #how close to goal to stop 
        max_iters = 10000

        print("Starting Bug Planner...") 
        print(f"Start: {start}, Goal: {goal}")

        for i in range(max_iters):
            if np.linalg.norm(current - goal) <= goal_threshold: #check if reached goal 
                print(f"Reached goal in {i} steps.")
                break

            if mode == "mline":
                if not mline_points: #no more m-line points, no path found 
                    print("M-line exhausted — no path found.")
                    break

                next_point = np.array(mline_points.pop(0)) #next point on the m-line 

                # obstacle hit
                if not self.env.state_validity_checker(next_point.reshape(2, 1)): 
                    print(f"[Step {i}] Hit obstacle at {current}, switching to wall-following.")
                    hit_point = current.copy()
                    mode = "wall" #start following wall 
                    next_point = self._get_next_wall_move(prev, current) #get next wall move 

            elif mode == "wall":
                next_point = self._get_next_wall_move(prev, current)

                # try to rejoin M-line if possible
                if self._on_mline(next_point, start, goal, threshold=1.0) \
                        and self._has_clear_line_to_goal(next_point, goal):
                    print(f"[Step {i}] Rejoined M-line, switching back to goal mode.")
                    mode = "mline"

                # stop if looping around obstacle
                if hit_point is not None and np.linalg.norm(current - hit_point) < 0.5 and i > 200:
                    print(f"[Step {i}] Returned to hit point — path not found.")
                    break

            prev = current.copy()
            current = next_point.copy()
            plan.append(current.copy())

        # Clean up path (remove duplicates, invalid points)
        filtered_plan = []
        for p in plan:
            if not filtered_plan or not np.array_equal(p, filtered_plan[-1]):
                if self.env.state_validity_checker(np.array(p).reshape(2, 1)):
                    filtered_plan.append(p)

        plan = np.array(filtered_plan)
        plan[:, 0] = np.clip(plan[:, 0], self.env.xlimit[0], self.env.xlimit[1])
        plan[:, 1] = np.clip(plan[:, 1], self.env.ylimit[0], self.env.ylimit[1])

        cost = len(plan)
        t_elapsed = time.time() - t0
        print(f"Cost: {cost}")
        print(f"Planning time: {t_elapsed:.2f}s")

        return plan.T
