import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq
import random

# Display the maze with the path in red
def display_maze(maze, path=None):
    maze_copy = np.array(maze)
    if path:
        for (x, y) in path:
            maze_copy[x][y] = 2  # Highlight the path with a different value
    # Plot with colors: 0=white, 1=black, 2=red
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red'])
    plt.imshow(maze_copy, cmap=cmap)
    plt.show()

# Get valid neighbors for a given position
def get_neighbors(maze, position):
    x, y = position
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

# Reconstruct the path from start to goal using parent pointers
def reconstruct_path(parent, start, goal):
    path = []
    current = goal
    while current:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path

# Depth-First Search (DFS) algorithm
def dfs(maze, start, goal):
    stack = [start]
    visited = set()
    parent = {start: None}
    
    while stack:
        current = stack.pop()
        if current == goal:
            return reconstruct_path(parent, start, goal)
        if current in visited:
            continue
        visited.add(current)
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited and neighbor not in stack:
                stack.append(neighbor)
                parent[neighbor] = current
    return None

# Breadth-First Search (BFS) algorithm
def bfs(maze, start, goal):
    queue = deque([start])
    visited = set()
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        if current == goal:
            return reconstruct_path(parent, start, goal)
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = current
    return None

# Uniform Cost Search (UCS) algorithm
def ucs(maze, start, goal):
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    visited = set()
    parent = {start: None}
    cost = {start: 0}
    
    while priority_queue:
        current_cost, current = heapq.heappop(priority_queue)
        if current == goal:
            return reconstruct_path(parent, start, goal)
        if current in visited:
            continue
        visited.add(current)
        for neighbor in get_neighbors(maze, current):
            new_cost = current_cost + 1  # Assuming uniform cost
            if neighbor not in visited or new_cost < cost.get(neighbor, float('inf')):
                cost[neighbor] = new_cost
                heapq.heappush(priority_queue, (new_cost, neighbor))
                parent[neighbor] = current
    return None

# Manhattan distance heuristic for A* and Best-First Search
def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

# A* Search algorithm
def a_star(maze, start, goal):
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    visited = set()
    parent = {start: None}
    cost = {start: 0}
    
    while priority_queue:
        _, current = heapq.heappop(priority_queue)
        if current == goal:
            return reconstruct_path(parent, start, goal)
        if current in visited:
            continue
        visited.add(current)
        for neighbor in get_neighbors(maze, current):
            new_cost = cost[current] + 1
            if neighbor not in visited or new_cost < cost.get(neighbor, float('inf')):
                cost[neighbor] = new_cost
                priority = new_cost + manhattan_distance(neighbor, goal)
                heapq.heappush(priority_queue, (priority, neighbor))
                parent[neighbor] = current
    return None

# Best-First Search algorithm
def best_first_search(maze, start, goal):
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    visited = set()
    parent = {start: None}
    
    while priority_queue:
        _, current = heapq.heappop(priority_queue)
        if current == goal:
            return reconstruct_path(parent, start, goal)
        if current in visited:
            continue
        visited.add(current)
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                priority = manhattan_distance(neighbor, goal)
                heapq.heappush(priority_queue, (priority, neighbor))
                parent[neighbor] = current
    return None

# Automatically generate a random maze
def generate_maze(size, density=0.3):
    maze = np.random.choice([0, 1], size=(size, size), p=[1-density, density])
    maze[0][0] = maze[size-1][size-1] = 0  # Ensure start and goal are open
    return maze.tolist()

# Main function for user interaction
def main():
    algorithms = {
        '1': dfs,
        '2': bfs,
        '3': ucs,
        '4': a_star,
        '5': best_first_search
    }

    print("Choose a search algorithm:")
    print("1. Depth-First Search (DFS)")
    print("2. Breadth-First Search (BFS)")
    print("3. Uniform Cost Search (UCS)")
    print("4. A* Search")
    print("5. Best-First Search")
    
    choice = input("Enter the number of the algorithm you want to use: ")
    if choice not in algorithms:
        print("Invalid choice. Exiting.")
        return

    maze_size = 15
    
    maze_option = input("Generate maze automatically (a) or enter manually (m)?: ").lower()
    if maze_option == 'a':
        maze = generate_maze(maze_size)
    elif maze_option == 'm':
        maze = []
        print("Enter the maze row by row (0 for open, 1 for wall):")
        for i in range(maze_size):
            row = list(map(int, input(f"Row {i+1}: ").strip().split()))
            if len(row) != maze_size:
                print("Invalid row length. Exiting.")
                return
            maze.append(row)
    else:
        print("Invalid option. Exiting.")
        return
    
    start = (0, 0)
    goal = (maze_size - 1, maze_size - 1)
    
    display_maze(maze)
    
    algorithm = algorithms[choice]
    path = algorithm(maze, start, goal)
    
    if path:
        print("Path found:", path)
        display_maze(maze, path)
    else:
        print("No path found.")
    
    if input("Solve another maze? (y/n): ").lower() == 'y':
        main()

if __name__ == "__main__":
    main()
