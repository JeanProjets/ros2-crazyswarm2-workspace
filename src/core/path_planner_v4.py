"""
Dynamic A* Path Planner for Scenario 4
Implements obstacle-aware path planning for drone navigation through cluttered environments.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict
import time


class GridMap:
    """
    Grid-based occupancy map for obstacle representation and collision checking.
    """

    def __init__(self, config: Dict):
        """
        Initialize the grid map from configuration.

        Args:
            config: Dictionary containing:
                - resolution: Grid cell size in meters (e.g., 0.25)
                - width: Map width in meters
                - height: Map height in meters
                - obstacles: List of obstacle definitions
        """
        self.resolution = config.get('resolution', 0.25)
        width = config.get('width', 10.0)
        height = config.get('height', 10.0)

        # Calculate grid dimensions
        self.width_cells = int(np.ceil(width / self.resolution))
        self.height_cells = int(np.ceil(height / self.resolution))

        # Initialize empty grid (0 = free, 1 = occupied)
        self.grid = np.zeros((self.width_cells, self.height_cells), dtype=np.uint8)

        # Store dimensions for coordinate conversion
        self.width_m = width
        self.height_m = height

        # Add obstacles from config
        if 'obstacles' in config:
            for obstacle in config['obstacles']:
                self.add_obstacle(obstacle)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid indices."""
        grid_x = int(np.floor(x / self.resolution))
        grid_y = int(np.floor(y / self.resolution))
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (center of cell)."""
        x = (grid_x + 0.5) * self.resolution
        y = (grid_y + 0.5) * self.resolution
        return x, y

    def add_obstacle(self, obstacle_def: Dict):
        """
        Add an obstacle to the map.

        Args:
            obstacle_def: Dictionary with obstacle definition:
                - type: "box" or "wall"
                - For box: center [x, y], size [width, height]
                - For wall: start [x, y], end [x, y], thickness
        """
        obstacle_type = obstacle_def.get('type')

        if obstacle_type == 'box':
            center = obstacle_def['center']
            size = obstacle_def['size']

            # Calculate bounds
            x_min = center[0] - size[0] / 2.0
            x_max = center[0] + size[0] / 2.0
            y_min = center[1] - size[1] / 2.0
            y_max = center[1] + size[1] / 2.0

            self._fill_rectangle(x_min, y_min, x_max, y_max)

        elif obstacle_type == 'wall':
            start = obstacle_def['start']
            end = obstacle_def['end']
            thickness = obstacle_def.get('thickness', 0.2)

            self._fill_wall(start[0], start[1], end[0], end[1], thickness)

    def _fill_rectangle(self, x_min: float, y_min: float, x_max: float, y_max: float):
        """Fill a rectangular region in the grid."""
        gx_min, gy_min = self.world_to_grid(x_min, y_min)
        gx_max, gy_max = self.world_to_grid(x_max, y_max)

        # Clamp to grid boundaries
        gx_min = max(0, gx_min)
        gy_min = max(0, gy_min)
        gx_max = min(self.width_cells - 1, gx_max)
        gy_max = min(self.height_cells - 1, gy_max)

        self.grid[gx_min:gx_max+1, gy_min:gy_max+1] = 1

    def _fill_wall(self, x1: float, y1: float, x2: float, y2: float, thickness: float):
        """Fill a wall (thick line) in the grid."""
        # Calculate perpendicular offset for thickness
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length < 1e-6:
            return

        # Normalized perpendicular vector
        perp_x = -dy / length * thickness / 2.0
        perp_y = dx / length * thickness / 2.0

        # Create rectangle around the line
        corners = [
            (x1 + perp_x, y1 + perp_y),
            (x1 - perp_x, y1 - perp_y),
            (x2 + perp_x, y2 + perp_y),
            (x2 - perp_x, y2 - perp_y)
        ]

        # Fill the region (simplified: use bounding box)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        self._fill_rectangle(min(xs), min(ys), max(xs), max(ys))

    def inflate_obstacles(self, radius: float):
        """
        Inflate obstacles by a safety margin.

        Args:
            radius: Inflation radius in meters (drone radius + safety margin)
        """
        if radius <= 0:
            return

        # Calculate inflation in grid cells
        inflation_cells = int(np.ceil(radius / self.resolution))

        # Create a copy of the original grid
        original_grid = self.grid.copy()

        # For each occupied cell, mark surrounding cells as occupied
        for i in range(self.width_cells):
            for j in range(self.height_cells):
                if original_grid[i, j] == 1:
                    # Inflate in a square pattern (simpler than circular)
                    i_min = max(0, i - inflation_cells)
                    i_max = min(self.width_cells - 1, i + inflation_cells)
                    j_min = max(0, j - inflation_cells)
                    j_max = min(self.height_cells - 1, j + inflation_cells)

                    self.grid[i_min:i_max+1, j_min:j_max+1] = 1

    def is_valid(self, x: float, y: float) -> bool:
        """
        Check if a world coordinate is valid (free space and within bounds).

        Args:
            x, y: World coordinates in meters

        Returns:
            True if the position is valid and collision-free
        """
        # Check bounds
        if x < 0 or x >= self.width_m or y < 0 or y >= self.height_m:
            return False

        # Convert to grid and check occupancy
        grid_x, grid_y = self.world_to_grid(x, y)

        # Additional bounds check for grid indices
        if grid_x < 0 or grid_x >= self.width_cells or grid_y < 0 or grid_y >= self.height_cells:
            return False

        return self.grid[grid_x, grid_y] == 0

    def is_collision(self, x: float, y: float) -> bool:
        """Check if a position is in collision (inverse of is_valid for grid check only)."""
        if x < 0 or x >= self.width_m or y < 0 or y >= self.height_m:
            return True

        grid_x, grid_y = self.world_to_grid(x, y)

        if grid_x < 0 or grid_x >= self.width_cells or grid_y < 0 or grid_y >= self.height_cells:
            return True

        return self.grid[grid_x, grid_y] == 1

    def has_line_of_sight(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """
        Check if there's a clear line of sight between two points using raycasting.

        Args:
            x1, y1: Start point in world coordinates
            x2, y2: End point in world coordinates

        Returns:
            True if line of sight is clear (no obstacles)
        """
        # Bresenham-like raycast
        gx1, gy1 = self.world_to_grid(x1, y1)
        gx2, gy2 = self.world_to_grid(x2, y2)

        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)

        x = gx1
        y = gy1

        x_inc = 1 if gx2 > gx1 else -1
        y_inc = 1 if gy2 > gy1 else -1

        # Check start and end points
        if not self._is_grid_valid(x, y) or not self._is_grid_valid(gx2, gy2):
            return False

        if dx > dy:
            error = dx / 2.0
            while x != gx2:
                if not self._is_grid_valid(x, y):
                    return False
                error -= dy
                if error < 0:
                    y += y_inc
                    error += dx
                x += x_inc
        else:
            error = dy / 2.0
            while y != gy2:
                if not self._is_grid_valid(x, y):
                    return False
                error -= dx
                if error < 0:
                    x += x_inc
                    error += dy
                y += y_inc

        return True

    def _is_grid_valid(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid coordinates are valid and free."""
        if grid_x < 0 or grid_x >= self.width_cells or grid_y < 0 or grid_y >= self.height_cells:
            return False
        return self.grid[grid_x, grid_y] == 0


class DynamicAStar:
    """
    Fast A* path planner for real-time obstacle avoidance.
    Optimized to return paths in < 50ms.
    """

    def __init__(self, grid_map: GridMap):
        """
        Initialize the planner.

        Args:
            grid_map: GridMap instance for collision checking
        """
        self.grid_map = grid_map
        self.replan_frequency = 2.0  # Hz
        self.last_plan_time = 0.0

    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a collision-free path from start to goal using A*.

        Args:
            start: (x, y) start position in world coordinates
            goal: (x, y) goal position in world coordinates

        Returns:
            List of (x, y) waypoints in world coordinates, or None if no path found
        """
        start_time = time.time()

        # Validate start and goal
        if not self.grid_map.is_valid(start[0], start[1]):
            print(f"Warning: Start position {start} is in collision or out of bounds")
            return None

        if not self.grid_map.is_valid(goal[0], goal[1]):
            print(f"Warning: Goal position {goal} is in collision or out of bounds")
            return None

        # Convert to grid coordinates
        start_grid = self.grid_map.world_to_grid(start[0], start[1])
        goal_grid = self.grid_map.world_to_grid(goal[0], goal[1])

        # Run A*
        path_grid = self._astar(start_grid, goal_grid)

        if path_grid is None:
            print(f"Warning: No path found from {start} to {goal}")
            return None

        # Convert back to world coordinates
        path_world = [self.grid_map.grid_to_world(gx, gy) for gx, gy in path_grid]

        # Smooth the path
        smoothed_path = self.smooth_path(path_world)

        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        print(f"Path planning took {elapsed:.1f}ms, waypoints: {len(path_grid)} -> {len(smoothed_path)}")

        return smoothed_path

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* implementation on grid coordinates.

        Returns:
            List of (grid_x, grid_y) coordinates, or None if no path found
        """
        # Priority queue: (f_score, counter, current_node)
        counter = 0
        open_set = [(0, counter, start)]
        counter += 1

        # Track visited nodes
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        # Set for fast membership checking
        open_set_hash = {start}

        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)

            # Goal reached
            if current == goal:
                return self._reconstruct_path(came_from, current)

            # Explore neighbors (8-connected grid)
            for neighbor in self._get_neighbors(current):
                # Calculate tentative g score
                # Diagonal moves cost sqrt(2), cardinal moves cost 1
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = 1.414 if (dx + dy) == 2 else 1.0

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        counter += 1
                        open_set_hash.add(neighbor)

        # No path found
        return None

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-connected)."""
        neighbors = []
        x, y = node

        # 8 directions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                # Check bounds and collision
                if (0 <= nx < self.grid_map.width_cells and
                    0 <= ny < self.grid_map.height_cells and
                    self.grid_map.grid[nx, ny] == 0):
                    neighbors.append((nx, ny))

        return neighbors

    def _heuristic(self, node: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from chain."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Smooth path by removing unnecessary waypoints using line-of-sight checks.

        Args:
            path: List of waypoints in world coordinates

        Returns:
            Smoothed path with fewer waypoints
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Try to skip as many waypoints as possible
            farthest_idx = current_idx + 1

            for i in range(current_idx + 2, len(path)):
                if self.grid_map.has_line_of_sight(
                    path[current_idx][0], path[current_idx][1],
                    path[i][0], path[i][1]
                ):
                    farthest_idx = i
                else:
                    break

            smoothed.append(path[farthest_idx])
            current_idx = farthest_idx

        return smoothed


class DynamicPlanner:
    """
    High-level planner that manages replanning and integrates target prediction.
    """

    def __init__(self, config: Dict):
        """
        Initialize the dynamic planner.

        Args:
            config: Configuration dictionary with map and navigation parameters
        """
        # Create grid map
        arena_config = config.get('arena_map', {})
        self.grid_map = GridMap(arena_config)

        # Apply safety inflation
        nav_params = config.get('nav_parameters', {})
        safety_margin = nav_params.get('safety_margin', 0.4)
        self.grid_map.inflate_obstacles(safety_margin)

        # Create A* planner
        self.astar = DynamicAStar(self.grid_map)

        # Replan parameters
        self.replan_rate = nav_params.get('replan_rate', 5.0)
        self.lookahead_time = nav_params.get('lookahead_time', 1.5)

        self.last_replan_time = 0.0
        self.current_path = None

    def get_path(self, start: Tuple[float, float], goal: Tuple[float, float],
                 force_replan: bool = False) -> Optional[List[Tuple[float, float]]]:
        """
        Get a path from start to goal, with optional replanning.

        Args:
            start: Current drone position
            goal: Target position
            force_replan: Force replanning even if not at replan interval

        Returns:
            List of waypoints or None if planning fails
        """
        current_time = time.time()

        # Check if we need to replan
        time_since_replan = current_time - self.last_replan_time
        should_replan = force_replan or (time_since_replan >= 1.0 / self.replan_rate)

        if should_replan or self.current_path is None:
            self.current_path = self.astar.plan_path(start, goal)
            self.last_replan_time = current_time

        return self.current_path

    def predict_intercept_point(self, target_pos: Tuple[float, float],
                                target_vel: Tuple[float, float]) -> Tuple[float, float]:
        """
        Predict where the target will be after lookahead_time.

        Args:
            target_pos: Current target position
            target_vel: Current target velocity

        Returns:
            Predicted future position
        """
        predicted_x = target_pos[0] + target_vel[0] * self.lookahead_time
        predicted_y = target_pos[1] + target_vel[1] * self.lookahead_time

        # Clamp to valid map bounds
        predicted_x = np.clip(predicted_x, 0, self.grid_map.width_m)
        predicted_y = np.clip(predicted_y, 0, self.grid_map.height_m)

        return (predicted_x, predicted_y)
