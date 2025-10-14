import numpy as np
import time
import itertools
import heapq #for dijkstras algorithm 


class VisibilityGraphPlanner(object):
    def __init__(self, planning_env):
        self.env = planning_env

    def get_obstacle_vertices(self):

        #find corners on the map by identifying 2*2 blocks with free cells next to obstacle cells
       
        map_ = self.env.map 
        vertices = set() 
        # Iterate over all possible 2x2 block top-left corners
        for r in range(map_.shape[0] - 1): 
            for c in range(map_.shape[1] - 1): 
                block = map_[r:r+2, c:c+2]
                # A corner is where free space meets obstacle space.
                # Sum of 1 or 3 indicates a corner in the 2x2 block.
                if np.sum(block) == 1 or np.sum(block) == 3:
                    # Find the free-space cell(s) in this block
                    for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        check_r, check_c = r + dr, c + dc
                        if map_[check_r, check_c] == 0:
                            # Add the coordinates of the valid, free-space cell
                            vertices.add((check_r, check_c))

        # convert to list of numpy arrays
        return [np.array(v).reshape(2, 1) for v in vertices]

 
    def _visible(self, p1, p2):
       
       #check if the line segement between two verticies is free of obstacles 
        return self.env.edge_validity_checker(
            
            np.array(p1).reshape(2, 1),
            np.array(p2).reshape(2, 1)
        )

#dijkstra's algorithm implementation 
    def _dijkstra(self, graph, start, goal):
       
     #priority queue based on dijstra's to find short path 
        start, goal = tuple(start), tuple(goal)
        dist = {v: np.inf for v in graph} #initialize distances to infinity 
        prev = {v: None for v in graph} #define prev for backtracking 
        dist[start] = 0
        pq = [(0, start)] #queue of (distance, vertex)

        while pq:
            d, u = heapq.heappop(pq) #get vertex with smallest distance so far 
            #if reached goal, stop
            if u == goal:  
                break
            #if shorter path exists, skip 
            if d > dist[u]: 
                continue
            for v, w in graph[u]:
                alt = d + w
                if alt < dist[v]:
                    #found shorter path to v
                    dist[v] = alt
                    prev[v] = u
                    #add to queue 
                    heapq.heappush(pq, (alt, v))

        # reconstruct the final path by backtracking from goal to start
        path = []
        node = goal
        while node is not None:
            path.insert(0, node)
            node = prev[node]
        return np.array(path).T if path else None

#main planning function
    def Plan(self, start_config, goal_config):
      
        plan_time = time.time()

        start = np.array(start_config).flatten()
        goal = np.array(goal_config).flatten()

        #get all vertices including start and goal
        vertices = [tuple(start), tuple(goal)]
        for v in self.get_obstacle_vertices():
            vertices.append(tuple(v.flatten()))

        # build the graph by checking visibility between all vertices pairs 
        graph = {v: [] for v in vertices}
        for v1, v2 in itertools.combinations(vertices, 2):
            if self._visible(v1, v2):
                w = np.linalg.norm(np.array(v1) - np.array(v2))
                graph[v1].append((v2, w))
                graph[v2].append((v1, w))

        #use dijkstra's to find the shortest path 
        path = self._dijkstra(graph, start, goal)

        #calculate stats
        plan_time = time.time() - plan_time
        cost = 0 if path is None else sum(
            np.linalg.norm(path[:, i+1] - path[:, i]) for i in range(path.shape[1]-1)
        )

        state_count = len(vertices)

        print("States Expanded: %d" % state_count)
        print("Cost: %.2f" % cost)
        print("Planning Time: %.2fs" % plan_time)

        # Return as a 2×N array for compatibility with the existing visualization
        return path if path is not None else np.array([start, goal]).T
