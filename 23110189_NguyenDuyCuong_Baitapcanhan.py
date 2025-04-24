import pygame
import time
import heapq
import random
from collections import deque
import math
from copy import deepcopy

# ... (Phần Khởi tạo Pygame, Màu sắc, Hằng số giữ nguyên) ...
pygame.init()

# Kích thước cửa sổ
WIDTH, HEIGHT = 900, 800
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Matrix UI")

# Màu sắc
WHITE = (255, 248, 252)
BLACK = (60, 50, 60)
PINK = (240, 100, 140)
PROCESS_PINK = (255, 120, 180)
BUTTON_COLOR = (230, 70, 140)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Kích thước ô vuông
CELL_SIZE = 60
MARGIN = 10
BORDER_RADIUS = 10
BUTTON_WIDTH, BUTTON_HEIGHT = 120, 40

# Danh sách thuật toán
algorithms = ["BFS", "DFS", "UCS", "IDS", "Greedy", "A*", "IDA*", "SHC", "S_AHC", "Stochastic", "Annealing", "And-Or Graph", "Genetic", "Sensorless", "Backtracking"]
selected_algorithm = None

# Biến toàn cục
global_elapsed_time = None
global_steps = None
global_path = [] # Thêm biến global để lưu đường đi cho nút Steps

# --- Các hàm vẽ (draw_matrix, draw_buttons, draw_action_buttons, draw_info_box) giữ nguyên ---
def draw_matrix(matrix, start_x, start_y, color=PINK):
    font = pygame.font.Font(None, 36)
    for row in range(3):
        for col in range(3):
            # Kiểm tra xem matrix có phải là list hợp lệ không trước khi truy cập
            if matrix and row < len(matrix) and col < len(matrix[row]):
                 value = matrix[row][col]
            else:
                 # Xử lý trường hợp matrix không hợp lệ (ví dụ: None hoặc list rỗng)
                 # Có thể vẽ ô trống hoặc bỏ qua
                 pygame.draw.rect(SCREEN, BLACK, (start_x + col * (CELL_SIZE + MARGIN), start_y + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE), 1, border_radius=BORDER_RADIUS)
                 continue # Bỏ qua phần còn lại của vòng lặp cho ô này


            rect_x = start_x + col * (CELL_SIZE + MARGIN)
            rect_y = start_y + row * (CELL_SIZE + MARGIN)
            pygame.draw.rect(SCREEN, color, (rect_x, rect_y, CELL_SIZE, CELL_SIZE), border_radius=BORDER_RADIUS)
            pygame.draw.rect(SCREEN, BLACK, (rect_x, rect_y, CELL_SIZE, CELL_SIZE), 2, border_radius=BORDER_RADIUS)

            if value != 0:
                text = font.render(str(value), True, BLACK)
                text_rect = text.get_rect(center=(rect_x + CELL_SIZE//2, rect_y + CELL_SIZE//2))
                SCREEN.blit(text, text_rect)

def draw_buttons():
    font = pygame.font.Font(None, 24)
    num_algorithms = len(algorithms)
    # Chia đều hơn nếu số lẻ
    half = (num_algorithms + 1) // 2

    for i, algo in enumerate(algorithms):
        col_index = i // half # 0 for left col, 1 for right col
        row_index = i % half

        if col_index == 0:
            # Cột bên trái
            button_x = 20
        else:
            # Cột bên phải
            button_x = WIDTH - BUTTON_WIDTH - 20

        button_y = 50 + row_index * (BUTTON_HEIGHT + 10)


        button_color = BUTTON_COLOR if selected_algorithm != algo else PROCESS_PINK
        pygame.draw.rect(SCREEN, button_color, (button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT), border_radius=5)
        pygame.draw.rect(SCREEN, BLACK, (button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT), 2, border_radius=5)
        text = font.render(algo, True, BUTTON_TEXT_COLOR)
        text_rect = text.get_rect(center=(button_x + BUTTON_WIDTH // 2, button_y + BUTTON_HEIGHT // 2))
        SCREEN.blit(text, text_rect)


def draw_action_buttons():
    """Vẽ các nút hành động như Solve, Random và Steps."""
    font = pygame.font.Font(None, 24)
    solve_button_x, solve_button_y = 250, 280  # Vị trí nút Solve
    random_button_x, random_button_y = 550, 280  # Vị trí nút Random
    steps_button_x, steps_button_y = 550, 500  # Vị trí nút Steps

    # Nút Solve
    pygame.draw.rect(SCREEN, BUTTON_COLOR, (solve_button_x, solve_button_y, BUTTON_WIDTH, BUTTON_HEIGHT), border_radius=5)
    pygame.draw.rect(SCREEN, BLACK, (solve_button_x, solve_button_y, BUTTON_WIDTH, BUTTON_HEIGHT), 2, border_radius=5)
    solve_text = font.render("Solve", True, BUTTON_TEXT_COLOR)
    solve_text_rect = solve_text.get_rect(center=(solve_button_x + BUTTON_WIDTH // 2, solve_button_y + BUTTON_HEIGHT // 2))
    SCREEN.blit(solve_text, solve_text_rect)

    # Nút Random
    pygame.draw.rect(SCREEN, BUTTON_COLOR, (random_button_x, random_button_y, BUTTON_WIDTH, BUTTON_HEIGHT), border_radius=5)
    pygame.draw.rect(SCREEN, BLACK, (random_button_x, random_button_y, BUTTON_WIDTH, BUTTON_HEIGHT), 2, border_radius=5)
    random_text = font.render("Random", True, BUTTON_TEXT_COLOR)
    random_text_rect = random_text.get_rect(center=(random_button_x + BUTTON_WIDTH // 2, random_button_y + BUTTON_HEIGHT // 2))
    SCREEN.blit(random_text, random_text_rect)

    # Nút Steps
    pygame.draw.rect(SCREEN, BUTTON_COLOR, (steps_button_x, steps_button_y, BUTTON_WIDTH, BUTTON_HEIGHT), border_radius=5)
    pygame.draw.rect(SCREEN, BLACK, (steps_button_x, steps_button_y, BUTTON_WIDTH, BUTTON_HEIGHT), 2, border_radius=5)
    steps_text = font.render("Steps", True, BUTTON_TEXT_COLOR)
    steps_text_rect = steps_text.get_rect(center=(steps_button_x + BUTTON_WIDTH // 2, steps_button_y + BUTTON_HEIGHT // 2))
    SCREEN.blit(steps_text, steps_text_rect)

def draw_info_box(elapsed_time=None, steps=None):
    """Vẽ khung thông tin với thời gian và số bước."""
    font = pygame.font.Font(None, 24)
    info_x, info_y = 500, 350
    info_width, info_height = 200, 120
    pygame.draw.rect(SCREEN, WHITE, (info_x, info_y, info_width, info_height), border_radius=BORDER_RADIUS) # Added border radius
    pygame.draw.rect(SCREEN, BLACK, (info_x, info_y, info_width, info_height), 2, border_radius=BORDER_RADIUS) # Added border radius

    title_text = font.render("Information", True, BLACK)
    title_rect = title_text.get_rect(center=(info_x + info_width // 2, info_y + 20))
    SCREEN.blit(title_text, title_rect)

    # Hiển thị thời gian
    time_str = f"Time: {elapsed_time:.4f}s" if elapsed_time is not None else "Time: --" # More precision
    time_text = font.render(time_str, True, BLACK)
    time_rect = time_text.get_rect(topleft=(info_x + 10, info_y + 40))
    SCREEN.blit(time_text, time_rect)

    # Hiển thị số bước
    steps_str = f"Steps: {steps}" if steps is not None else "Steps: --"
    steps_text = font.render(steps_str, True, BLACK)
    steps_rect = steps_text.get_rect(topleft=(info_x + 10, info_y + 70))
    SCREEN.blit(steps_text, steps_rect)


# --- Hàm trợ giúp (find_blank, get_neighbors) ---
def find_blank(state):
    """Tìm vị trí ô trống (số 0). Đảm bảo state hợp lệ."""
    if not state or len(state) != 3: # Kiểm tra cơ bản
        return None
    for i in range(3):
        if len(state[i]) != 3: return None # Kiểm tra hàng
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return None # Không tìm thấy ô trống (trạng thái không hợp lệ?)

def get_neighbors(state):
    """Tạo ra các trạng thái hàng xóm hợp lệ."""
    moves = []
    blank_pos = find_blank(state)
    if blank_pos is None:
        # print("Warning: find_blank failed in get_neighbors for state:", state) # Debug
        return [] # Trả về list rỗng nếu không tìm thấy ô trống
    x, y = blank_pos

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = deepcopy(state) # Sử dụng deepcopy để tránh lỗi tham chiếu
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            moves.append(new_state)
    return moves

def bfs_solve(start_state, goal_state):
    start_tuple = tuple(tuple(row) for row in start_state) # Chuyển sang tuple
    goal_tuple = tuple(tuple(row) for row in goal_state)   # Chuyển sang tuple

    queue = deque([(start_state, [start_state])]) # Lưu trữ đường đi trực tiếp
    visited = {start_tuple} # Set chứa các tuple trạng thái đã thăm

    while queue:
        state, path = queue.popleft()
        current_tuple = tuple(tuple(row) for row in state) # Chuyển state hiện tại sang tuple

        if current_tuple == goal_tuple:
            return path # Trả về danh sách các trạng thái

        neighbors = get_neighbors(state)
        if not neighbors: # Xử lý trường hợp get_neighbors trả về rỗng
            continue

        for neighbor in neighbors:
            neighbor_tuple = tuple(tuple(row) for row in neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
    return []

def dfs_solve(start_state, goal_state):
    start_tuple = tuple(tuple(row) for row in start_state)
    goal_tuple = tuple(tuple(row) for row in goal_state)

    stack = [(start_state, [start_state])] # (state, path_list)
    visited = {start_tuple} # Sử dụng set các tuple

    while stack:
        state, path = stack.pop()
        current_tuple = tuple(tuple(row) for row in state)

        if current_tuple == goal_tuple:
            return path
        neighbors = get_neighbors(state)
        if not neighbors: continue
        for neighbor in reversed(neighbors):
            neighbor_tuple = tuple(tuple(row) for row in neighbor)
            if neighbor_tuple not in visited:
                 visited.add(neighbor_tuple) # Đánh dấu đã thăm trước khi thêm vào stack
                 new_path = path + [neighbor]
                 stack.append((neighbor, new_path))
    return []

def ucs_solve(start_state, goal_state):
    start_tuple = tuple(tuple(row) for row in start_state)
    goal_tuple = tuple(tuple(row) for row in goal_state)

    # (cost, state_object, path_list)
    priority_queue = [(0, start_state, [start_state])]
    # visited lưu trữ chi phí tốt nhất để đến tuple đó: {state_tuple: cost}
    visited = {start_tuple: 0}

    while priority_queue:
        cost, state, path = heapq.heappop(priority_queue)
        current_tuple = tuple(tuple(row) for row in state)

        # Nếu chi phí hiện tại lớn hơn chi phí đã biết, bỏ qua
        if cost > visited.get(current_tuple, float('inf')):
            continue

        if current_tuple == goal_tuple:
            return path

        neighbors = get_neighbors(state)
        if not neighbors: continue

        for neighbor in neighbors:
            neighbor_tuple = tuple(tuple(row) for row in neighbor)
            new_cost = cost + 1 # Chi phí mỗi bước là 1

            if new_cost < visited.get(neighbor_tuple, float('inf')):
                visited[neighbor_tuple] = new_cost
                new_path = path + [neighbor]
                heapq.heappush(priority_queue, (new_cost, neighbor, new_path))
    return []

# --- Heuristic Function (Manhattan Distance is generally better) ---
def heuristic_misplaced(state, goal_state):
    """Hàm heuristic tính số ô sai vị trí."""
    goal_tuple = tuple(tuple(row) for row in goal_state) # Cache goal tuple?
    count = 0
    for r in range(3):
        for c in range(3):
            if state[r][c] != 0 and state[r][c] != goal_state[r][c]:
                 count += 1
    return count

def heuristic_manhattan(state, goal_state):
    """Hàm heuristic tính tổng khoảng cách Manhattan."""
    distance = 0
    goal_pos = {}
    for r in range(3):
        for c in range(3):
            goal_pos[goal_state[r][c]] = (r, c)

    for r in range(3):
        for c in range(3):
            val = state[r][c]
            if val != 0:
                try:
                    goal_r, goal_c = goal_pos[val]
                    distance += abs(r - goal_r) + abs(c - goal_c)
                except KeyError:
                    # print(f"Warning: Value {val} not found in goal state positions.") # Debug
                    # Handle potentially invalid states from GA?
                    # Return a large distance or skip? Let's return large.
                    return float('inf')
    return distance
def ids_solve(start_state, goal_state):
    # --- (Code IDS giữ nguyên, đảm bảo state sang tuple khi cần) ---
    def dls(state, goal, depth, path):
        state_tuple = tuple(tuple(row) for row in state)
        goal_tuple = tuple(tuple(row) for row in goal)

        if state_tuple == goal_tuple:
            return path
        if depth <= 0:
            return None

        # Check for cycles in the current path to avoid trivial loops
        path_tuples = {tuple(tuple(p_state)) for p_state in path}

        neighbors = get_neighbors(state)
        if not neighbors: return None

        for neighbor in neighbors:
             neighbor_tuple = tuple(tuple(row) for row in neighbor)
             if neighbor_tuple not in path_tuples: # Avoid immediate cycles in DLS path
                 result = dls(neighbor, goal, depth - 1, path + [neighbor])
                 if result:
                     return result
        return None

    for depth in range(100): # Giới hạn độ sâu tối đa
        print(f"IDS: Trying depth {depth}")
        result = dls(start_state, goal_state, depth, [start_state])
        if result:
            # The DLS function returns the path including the start state
            return result # Return the path list directly
    return []


def greedy_solve(start_state, goal_state):
    # --- (Code Greedy giữ nguyên, dùng Manhattan) ---
    start_tuple = tuple(tuple(row) for row in start_state)
    goal_tuple = tuple(tuple(row) for row in goal_state)

    # (heuristic_value, state_object, path_list)
    priority_queue = [(heuristic_manhattan(start_state, goal_state), start_state, [start_state])]
    visited = {start_tuple} # Set of visited state tuples

    while priority_queue:
        _, state, path = heapq.heappop(priority_queue)
        current_tuple = tuple(tuple(row) for row in state)

        if current_tuple == goal_tuple:
            return path

        neighbors = get_neighbors(state)
        if not neighbors: continue

        for neighbor in neighbors:
            neighbor_tuple = tuple(tuple(row) for row in neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                h_value = heuristic_manhattan(neighbor, goal_state)
                new_path = path + [neighbor]
                heapq.heappush(priority_queue, (h_value, neighbor, new_path))
    return []

def a_star_solve(start_state, goal_state):
    # --- (Code A* giữ nguyên, dùng Manhattan) ---
    start_tuple = tuple(tuple(row) for row in start_state)
    goal_tuple = tuple(tuple(row) for row in goal_state)

    # (f_cost, g_cost, state_object, path_list)
    priority_queue = [(heuristic_manhattan(start_state, goal_state), 0, start_state, [start_state])]
    # visited stores the *minimum g_cost* found so far for a state tuple
    visited = {start_tuple: 0}

    while priority_queue:
        f, g, state, path = heapq.heappop(priority_queue)
        current_tuple = tuple(tuple(row) for row in state)

        # Optimization: If we found a shorter path already processed, skip
        if g > visited.get(current_tuple, float('inf')):
            continue

        if current_tuple == goal_tuple:
            return path

        neighbors = get_neighbors(state)
        if not neighbors: continue

        for neighbor in neighbors:
            neighbor_tuple = tuple(tuple(row) for row in neighbor)
            new_g = g + 1
            if new_g < visited.get(neighbor_tuple, float('inf')):
                visited[neighbor_tuple] = new_g
                h = heuristic_manhattan(neighbor, goal_state)
                new_f = new_g + h
                new_path = path + [neighbor]
                heapq.heappush(priority_queue, (new_f, new_g, neighbor, new_path))

    return []

def ida_star_solve(start_state, goal_state):
    # --- (Code IDA* giữ nguyên, dùng Manhattan) ---
    goal_tuple = tuple(tuple(row) for row in goal_state)

    def search(path, g, threshold):
        state = path[-1]
        h = heuristic_manhattan(state, goal_state)
        f = g + h

        if f > threshold:
            return f, None

        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple == goal_tuple:
            return f, path

        min_threshold = float('inf')
        neighbors = get_neighbors(state)
        if not neighbors: return float('inf'), None # Dead end

        for neighbor in neighbors:
             # Avoid immediate backtracking
             if len(path) < 2 or neighbor != path[-2]:
                 new_path = path + [neighbor]
                 t, result = search(new_path, g + 1, threshold)
                 if result:
                     return t, result
                 min_threshold = min(min_threshold, t)

        return min_threshold, None

    threshold = heuristic_manhattan(start_state, goal_state)
    path = [start_state]

    while True:
        print(f"IDA*: Trying threshold {threshold}")
        t, result = search(path, 0, threshold)
        if result:
            return result
        if t == float('inf'):
            return [] # No solution found
        threshold = t

def shc_solve(start_state, goal_state): # Simple Hill Climbing
    # --- (Code SHC giữ nguyên, dùng Manhattan) ---
    current_state = start_state
    path = [current_state]
    visited_tuples = {tuple(tuple(row) for row in current_state)} # Avoid cycles

    while tuple(tuple(row) for row in current_state) != tuple(tuple(row) for row in goal_state):
        current_h = heuristic_manhattan(current_state, goal_state)
        neighbors = get_neighbors(current_state)
        if not neighbors: break # No neighbors

        best_neighbor = None
        found_better = False

        # Sort neighbors by heuristic to find the best one(s) first
        sorted_neighbors = sorted(neighbors, key=lambda n: heuristic_manhattan(n, goal_state))

        for neighbor in sorted_neighbors:
            neighbor_h = heuristic_manhattan(neighbor, goal_state)
            neighbor_tuple = tuple(tuple(row) for row in neighbor)

            # Take the first strictly better neighbor found that hasn't been visited
            if neighbor_h < current_h and neighbor_tuple not in visited_tuples:
                best_neighbor = neighbor
                found_better = True
                break # Found a better one, move

        if not found_better:
            break # Local minimum or cycle detected

        current_state = best_neighbor
        path.append(current_state)
        visited_tuples.add(tuple(tuple(row) for row in current_state))

    # Return path only if goal is actually reached
    return path if tuple(tuple(row) for row in path[-1]) == tuple(tuple(row) for row in goal_state) else []


def s_ahc_solve(start_state, goal_state): # Steepest Ascent Hill Climbing
    # --- (Code S_AHC giữ nguyên, dùng Manhattan) ---
    current_state = start_state
    path = [current_state]
    visited_tuples = {tuple(tuple(row) for row in current_state)}

    while tuple(tuple(row) for row in current_state) != tuple(tuple(row) for row in goal_state):
        current_h = heuristic_manhattan(current_state, goal_state)
        neighbors = get_neighbors(current_state)
        if not neighbors: break

        best_neighbor = None
        min_h = current_h

        # Find the absolute best among all valid neighbors
        candidates = []
        for neighbor in neighbors:
             if tuple(tuple(row) for row in neighbor) not in visited_tuples:
                  candidates.append(neighbor)

        if not candidates: break # No unvisited neighbors

        # Find the best one among candidates
        temp_best = min(candidates, key=lambda n: heuristic_manhattan(n, goal_state))
        temp_min_h = heuristic_manhattan(temp_best, goal_state)

        # Must be strictly better to move
        if temp_min_h < current_h:
             best_neighbor = temp_best
        else:
             break # Local minimum or plateau

        current_state = best_neighbor
        path.append(current_state)
        visited_tuples.add(tuple(tuple(row) for row in current_state))

    return path if tuple(tuple(row) for row in path[-1]) == tuple(tuple(row) for row in goal_state) else []

def stochastic_solve(start_state, goal_state):
     # --- (Code Stochastic giữ nguyên, dùng Manhattan) ---
    current_state = start_state
    path = [current_state]
    visited_tuples = {tuple(tuple(row) for row in current_state)}
    max_steps = 5000 # Limit steps
    steps_taken = 0

    while tuple(tuple(row) for row in current_state) != tuple(tuple(row) for row in goal_state) and steps_taken < max_steps:
        current_h = heuristic_manhattan(current_state, goal_state)
        neighbors = get_neighbors(current_state)
        if not neighbors: break

        # Filter for *better* neighbors only
        uphill_neighbors = []
        for neighbor in neighbors:
             neighbor_tuple = tuple(tuple(row) for row in neighbor)
             if heuristic_manhattan(neighbor, goal_state) < current_h and neighbor_tuple not in visited_tuples:
                 uphill_neighbors.append(neighbor)

        if not uphill_neighbors:
             break # Local minimum

        # Choose randomly from the better neighbors
        next_state = random.choice(uphill_neighbors)
        current_state = next_state
        path.append(current_state)
        visited_tuples.add(tuple(tuple(row) for row in current_state))
        steps_taken += 1

    return path if tuple(tuple(row) for row in path[-1]) == tuple(tuple(row) for row in goal_state) else []


def Simulated_Annealing(start_state, goal_state):
     # --- (Code Annealing giữ nguyên, dùng Manhattan) ---
    initial_temp = 100.0 # Start reasonably high
    cooling_rate = 0.995 # Cool slightly slower maybe
    min_temp = 0.1
    max_iterations_at_temp = 100 # Iterations before cooling

    current_state = start_state
    current_h = heuristic_manhattan(current_state, goal_state)
    # Path tracking is tricky for SA, as it wanders. We track the best *seen* state.
    best_state = current_state
    best_h = current_h
    # We *can* store the actual path taken, but it won't be optimal
    path_taken = [current_state]

    current_temp = initial_temp

    while current_temp > min_temp:
        if best_h == 0: break # Found the goal state as best

        for _ in range(max_iterations_at_temp):
            if heuristic_manhattan(current_state, goal_state) == 0: # Check current state too
                 if current_h < best_h: # Update best if current is goal
                     best_state = current_state
                     best_h = current_h
                 break # Exit inner loop if current is goal

            neighbors = get_neighbors(current_state)
            if not neighbors: continue # Skip if no neighbors

            next_state = random.choice(neighbors)
            next_h = heuristic_manhattan(next_state, goal_state)
            delta_e = current_h - next_h # Higher delta means next_state is better

            if delta_e > 0 or random.random() < math.exp(delta_e / current_temp):
                 current_state = next_state
                 current_h = next_h
                 path_taken.append(current_state) # Track the wandering path
                 # Update best state found so far
                 if current_h < best_h:
                     best_state = current_state
                     best_h = current_h

        if heuristic_manhattan(current_state, goal_state) == 0: break # Exit outer loop if goal found
        current_temp *= cooling_rate # Cool down

    print(f"SimAnneal finished. Best heuristic found: {best_h}")
    # If the best state found is the goal, we need to reconstruct a path.
    # Since SA doesn't guarantee optimality or store parents,
    # the simplest is to return the *wandering path* if its end is the goal,
    # or try to run A* from start to the best_state found if it's the goal.
    # For simplicity here, return the path_taken *only if* the final state is the goal.
    if tuple(tuple(row) for row in path_taken[-1]) == tuple(tuple(row) for row in goal_state):
         print("Goal reached at the end of SA path.")
         return path_taken
    elif best_h == 0:
         print("Goal was found as best state, but path reconstruction not implemented here. Returning empty.")
         # To return a path: run A*(start_state, best_state) here.
         return [] # Placeholder
    else:
         print("Goal not reached.")
         return []


def and_or_graph_search(start_state, goal_state):
    # --- (Code And-Or Graph giữ nguyên) ---
    # Note: This implementation is complex and might not be standard for 8-puzzle.
    # It likely behaves like DFS or BFS in practice here.
    print("Warning: And-Or Graph Search implementation might be basic.")
    def or_search(state, goal_state, path_tuples):
        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple == tuple(tuple(row) for row in goal_state):
            return state

        if state_tuple in path_tuples:
            return 'failure'

        neighbors = get_neighbors(state)
        if not neighbors: return 'failure'

        # For 8-puzzle, OR means trying any valid move.
        # AND doesn't really apply unless subproblems were defined differently.
        # This structure resembles standard search more than true AND-OR decomposition.
        plan = {} # Store potential plan steps

        for neighbor in neighbors:
             # Simulate the 'AND' part - for 8-puzzle, the 'AND' is just reaching the next state.
             # Then call OR search recursively.
             result = or_search(neighbor, goal_state, path_tuples | {state_tuple})
             if result != 'failure':
                  # Found a path via this neighbor
                  # Return a structure indicating the choice made
                  return {'OR': (state, result)} # state -> chose neighbor -> result path

        return 'failure'

    # Initial call
    plan_tree = or_search(start_state, goal_state, set())

    if plan_tree == 'failure':
         return []
    else:
         # Extract path from the plan tree structure
         path = []
         curr = plan_tree
         while isinstance(curr, dict):
             if 'OR' in curr:
                 state_node, next_node = curr['OR']
                 path.append(state_node)
                 curr = next_node
             # Add handling for 'AND' if it were used differently
             else: break # Unknown structure
         if curr != 'failure': # Append the final goal state
             path.append(curr)
         return path


def sensorless_search(start_state, goal_state):
    """
    Sensorless search cho 8-puzzle với trạng thái bắt đầu đã biết.
    Hoạt động tương đương BFS trong trường hợp này.
    Trả về đường đi (list các states) nếu thành công.
    """
    print("Running Sensorless Search (equivalent to BFS for known start state)...")
    # Chỉ cần gọi BFS vì trạng thái bắt đầu đã biết và hành động xác định
    return bfs_solve(start_state, goal_state)

def genetic_search(start_state, goal_state, population_size=50, generations=100, mutation_rate=0.2):
    """
    Genetic Algorithm (Đơn giản hóa) để giải bài toán 8-puzzle.
    Trả về trạng thái cuối cùng tốt nhất tìm được hoặc [] nếu thất bại.
    Lưu ý: Phiên bản này rất cơ bản và có thể không hiệu quả.
    """
    print(f"Running Genetic Algorithm (Pop:{population_size}, Gen:{generations}, Mut:{mutation_rate})...")
    start_time_ga = time.time()

    # --- Fitness Function (Sử dụng Manhattan) ---
    def fitness(state, goal):
        # Trả về giá trị fitness cao hơn cho trạng thái tốt hơn (gần goal hơn)
        # Dùng -Manhattan hoặc 1 / (1 + Manhattan)
        mh = heuristic_manhattan(state, goal)
        if mh == float('inf'): return -float('inf') # Penalize invalid states heavily
        return -mh # Cao hơn là tốt hơn

    # --- Mutation ---
    def mutate(state):
        neighbors = get_neighbors(state)
        if not neighbors:
            return state # Trả về trạng thái cũ nếu không có nước đi
        return random.choice(neighbors)

    # --- Khởi tạo Quần thể (với một chút đa dạng) ---
    population = [start_state]
    visited_init_tuples = {tuple(tuple(row) for row in start_state)}
    current_gen_states = [start_state]

    # Tạo các cá thể ban đầu bằng cách thực hiện các bước đi ngẫu nhiên
    # Giữ cho quần thể ban đầu không quá xa trạng thái bắt đầu
    tries = 0
    max_tries = population_size * 5 # Giới hạn số lần thử để tránh vòng lặp vô hạn
    while len(population) < population_size and tries < max_tries:
        tries += 1
        # Chọn ngẫu nhiên một trạng thái từ các trạng thái đã tạo gần đây
        parent_state = random.choice(current_gen_states)
        neighbors = get_neighbors(parent_state)
        if neighbors:
            child_state = random.choice(neighbors)
            child_tuple = tuple(tuple(row) for row in child_state)
            if child_tuple not in visited_init_tuples:
                 population.append(child_state)
                 visited_init_tuples.add(child_tuple)
                 current_gen_states.append(child_state)
                 # Giữ kích thước của current_gen_states hợp lý để không bị lệch quá xa
                 if len(current_gen_states) > population_size // 2:
                     current_gen_states.pop(0)
        # Nếu không thêm được cá thể mới, lặp lại từ đầu
        if not neighbors or child_tuple in visited_init_tuples:
            current_gen_states = [start_state] # Reset lại để thử từ đầu


    # Đảm bảo đủ kích thước quần thể, thêm trạng thái bắt đầu nếu cần
    while len(population) < population_size:
         population.append(deepcopy(start_state))

    print(f"GA Initial population size: {len(population)}")


    # --- Vòng lặp Thế hệ ---
    best_state_overall = start_state
    best_fitness_overall = fitness(start_state, goal_state)

    for generation in range(generations):
        # Đánh giá fitness cho toàn bộ quần thể
        fitness_scores = [(fitness(state, goal_state), state) for state in population]

        # Sắp xếp theo fitness (cao hơn là tốt hơn)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)

        # Kiểm tra lời giải
        current_best_fitness, current_best_state = fitness_scores[0]
        if current_best_fitness > best_fitness_overall:
             best_fitness_overall = current_best_fitness
             best_state_overall = current_best_state

        if current_best_fitness == 0:  # Tìm thấy lời giải (Manhattan = 0)
            print(f"GA Solution found in generation {generation + 1}!")
            print(f"Total GA Time: {time.time() - start_time_ga:.4f}s")
            # Trả về trạng thái đích tìm được
            # Hàm gọi sẽ tạo path = [start, solution]
            return current_best_state

        # Chọn lọc (Giữ lại nửa tốt nhất - Truncation Selection)
        # Có thể dùng các phương pháp chọn lọc khác (Roulette, Tournament)
        num_survivors = population_size // 2
        survivors = [state for fit, state in fitness_scores[:num_survivors]]

        # Tạo thế hệ mới từ survivors thông qua đột biến (và crossover bị loại bỏ)
        new_population = deepcopy(survivors) # Giữ lại survivors

        while len(new_population) < population_size:
             # Chọn ngẫu nhiên một survivor làm cha mẹ
             parent = random.choice(survivors)
             # Tạo con chỉ bằng cách đột biến cha mẹ (Do crossover bị lỗi)
             child = mutate(parent)
             # Có thể thêm một bước đột biến nữa với tỷ lệ mutation_rate
             if random.random() < mutation_rate: # Đột biến thêm lần nữa
                 child = mutate(child)

             new_population.append(child)

        population = new_population # Cập nhật quần thể

        if (generation + 1) % 10 == 0: # In log định kỳ
             print(f"GA Gen {generation + 1}: Best Fitness = {current_best_fitness:.2f}")


    # --- Kết thúc GA ---
    print(f"GA finished. Max generations reached. Best fitness found: {best_fitness_overall:.2f}")
    print(f"Total GA Time: {time.time() - start_time_ga:.4f}s")
    # Trả về trạng thái tốt nhất tìm được, ngay cả khi không phải goal
    return best_state_overall


def backtracking_search(start_state, goal_state, max_depth=50):
    """Thuật toán Backtracking Search."""
    path = []  # Lưu trữ đường đi
    visited = set()  # Tập các trạng thái đã thăm để tránh lặp

    def backtrack(state, depth):
        # Nếu vượt quá độ sâu tối đa, quay lui
        if depth > max_depth:
            return False

        # Thêm trạng thái hiện tại vào đường đi
        path.append(state)

        # Nếu đạt trạng thái mục tiêu, trả về True
        if tuple(tuple(row) for row in state) == tuple(tuple(row) for row in goal_state):
            return True

        # Đánh dấu trạng thái đã thăm
        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple in visited:
            path.pop()
            return False
        visited.add(state_tuple)

        # Lấy tất cả các trạng thái có thể di chuyển
        for next_state in get_neighbors(state):
            if backtrack(next_state, depth + 1):
                return True

        # Nếu không tìm thấy lời giải, loại bỏ trạng thái hiện tại khỏi đường đi
        path.pop()
        return False

    # Gọi hàm backtrack từ trạng thái ban đầu
    if backtrack(start_state, 0):
        return path
    else:
        return []







# --- Ma trận trạng thái ban đầu và đích ---
original_state = [
     [1, 2, 3],
    [4, 5, 6],
    [0, 7, 8]
]

target_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Khởi tạo process_state giống original_state ban đầu
process_state = deepcopy(original_state)

# --- Hàm cập nhật và vẽ màn hình ---
def update_process_state(path):
    """Cập nhật trạng thái process_state theo từng bước trong path."""
    global process_state
    if not path or not isinstance(path, list):
         print("Invalid path for update_process_state")
         return
    for i, state in enumerate(path):
         # Kiểm tra state có hợp lệ không
         if not state or not isinstance(state, list) or len(state) != 3:
              print(f"Invalid state in path at index {i}: {state}")
              continue # Bỏ qua trạng thái không hợp lệ

         process_state = state
         draw_screen(color=PROCESS_PINK)
         pygame.display.flip() # Cần flip để thấy cập nhật
         pygame.time.delay(300) # Giảm delay một chút
         pygame.event.pump() # Xử lý event để tránh treo
    # Giữ lại trạng thái cuối cùng và đổi màu lại
    draw_screen(color=PINK)
    pygame.display.flip()


def draw_screen(color=PINK):
    """Vẽ toàn bộ màn hình."""
    SCREEN.fill(WHITE)
    draw_buttons()
    draw_action_buttons()
    draw_matrix(original_state, 220, 50)
    draw_matrix(target_state, 500, 50)
    # Truyền màu chính xác cho process_state
    draw_matrix(process_state, 220, 350, color=color if color == PROCESS_PINK else PINK)
    draw_info_box(global_elapsed_time, global_steps)
    # Không cần flip ở đây, flip sẽ ở update_process_state hoặc vòng lặp chính
    # pygame.display.flip()


def print_steps_to_terminal(path):
    """In ra các ma trận trạng thái trong terminal."""
    global global_path # Sử dụng biến global đã lưu
    if not global_path:
        print("No path available to print steps.")
        return

    print("\n--- Path States ---")
    for i, state in enumerate(global_path):
         if not state or not isinstance(state, list): # Kiểm tra state trước khi in
              print(f"Step {i}: Invalid state data")
              continue
         print(f"Step {i}:")
         for row in state:
            # Kiểm tra row trước khi in
            if isinstance(row, list):
                 print(f"  {row}")
            else:
                 print(f"  Invalid row data: {row}")
         print("-" * 15)
    print(f"Total steps in path: {len(global_path) - 1}")
    print("-------------------\n")


def randomize_matrix():
    """Tạo ma trận ngẫu nhiên có thể giải được."""
    global original_state, process_state, global_elapsed_time, global_steps, global_path
    # Tạo trạng thái giải được bằng cách đi ngược từ goal
    temp_state = deepcopy(target_state)
    visited_random = {tuple(tuple(row) for row in temp_state)}
    for _ in range(100): # Số bước đi ngẫu nhiên
        neighbors = get_neighbors(temp_state)
        possible_next = [n for n in neighbors if tuple(tuple(row) for row in n) not in visited_random]
        if not possible_next: break # Không có nước đi mới
        next_s = random.choice(possible_next)
        visited_random.add(tuple(tuple(row) for row in next_s))
        temp_state = next_s

    # Đảm bảo trạng thái tạo ra khác trạng thái đích
    if temp_state == target_state:
         # Nếu vẫn là goal, thử lại với nhiều bước hơn hoặc chỉ đảo 2 ô cuối
         neighbors = get_neighbors(temp_state)
         if neighbors: temp_state = random.choice(neighbors)


    original_state = temp_state
    process_state = deepcopy(original_state) # Reset process state
    # Reset info và path
    global_elapsed_time = None
    global_steps = None
    global_path = []
    print("Generated new random solvable start state.")


# --- Vòng lặp chính ---
running = True
draw_screen() # Vẽ màn hình ban đầu
pygame.display.flip() # Hiển thị lần đầu

while running:
    for event in pygame.event.get():
        redraw_needed = False # Cờ để chỉ vẽ lại khi cần

        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            redraw_needed = True # Click chuột -> cần vẽ lại

            # --- Xử lý chọn thuật toán ---
            num_algorithms = len(algorithms)
            half = (num_algorithms + 1) // 2
            button_clicked = False
            for i, algo in enumerate(algorithms):
                col_index = i // half
                row_index = i % half
                button_x = 20 if col_index == 0 else WIDTH - BUTTON_WIDTH - 20
                button_y = 50 + row_index * (BUTTON_HEIGHT + 10)

                if button_x <= x <= button_x + BUTTON_WIDTH and button_y <= y <= button_y + BUTTON_HEIGHT:
                    if selected_algorithm != algo:
                         selected_algorithm = algo
                         print(f"Selected Algorithm: {selected_algorithm}")
                    button_clicked = True
                    break # Chọn một cái thôi

            # --- Xử lý nút "Solve" ---
            solve_button_x, solve_button_y = 250, 280
            if not button_clicked and solve_button_x <= x <= solve_button_x + BUTTON_WIDTH and solve_button_y <= y <= solve_button_y + BUTTON_HEIGHT:
                button_clicked = True
                if selected_algorithm:
                    print(f"Solving using {selected_algorithm}...")
                    process_state = deepcopy(original_state) # Reset trước khi giải
                    draw_screen(color=PROCESS_PINK) # Hiển thị đang xử lý
                    pygame.display.flip() # Cập nhật màn hình
                    pygame.event.pump()   # Xử lý event nhanh

                    start_time = time.time()
                    path_result = [] # Đặt lại path_result

                    # --- Gọi thuật toán ---
                    try: # Bọc trong try-except để bắt lỗi dễ hơn
                        if selected_algorithm == "BFS": path_result = bfs_solve(original_state, target_state)
                        elif selected_algorithm == "DFS": path_result = dfs_solve(original_state, target_state)
                        elif selected_algorithm == "UCS": path_result = ucs_solve(original_state, target_state)
                        elif selected_algorithm == "IDS": path_result = ids_solve(original_state, target_state)
                        elif selected_algorithm == "Greedy": path_result = greedy_solve(original_state, target_state)
                        elif selected_algorithm == "A*": path_result = a_star_solve(original_state, target_state)
                        elif selected_algorithm == "IDA*": path_result = ida_star_solve(original_state, target_state)
                        elif selected_algorithm == "SHC": path_result = shc_solve(original_state, target_state)
                        elif selected_algorithm == "S_AHC": path_result = s_ahc_solve(original_state, target_state)
                        elif selected_algorithm == "Stochastic": path_result = stochastic_solve(original_state, target_state)
                        elif selected_algorithm == "Annealing": path_result = Simulated_Annealing(original_state, target_state)
                        elif selected_algorithm == "And-Or Graph": path_result = and_or_graph_search(original_state, target_state)
                        elif selected_algorithm == "Sensorless":
                            path_result = sensorless_search(original_state, target_state) # Gọi hàm đã sửa
                        elif selected_algorithm == "Genetic":
                            # GA trả về trạng thái cuối cùng, không phải path
                            solution_state = genetic_search(original_state, target_state)
                            if solution_state and tuple(tuple(row) for row in solution_state) == tuple(tuple(row) for row in target_state):
                                print("Genetic Algorithm found the goal state.")
                                # Tạo path giả lập để hiển thị start -> end
                                # Cần A* hoặc BFS để tìm path thật từ start -> solution_state nếu muốn path đúng
                                # path_result = a_star_solve(original_state, solution_state) # Tùy chọn: Tìm path thật
                                path_result = [original_state, solution_state] # Đơn giản: chỉ start và end
                            elif solution_state:
                                print("Genetic Algorithm finished, showing best state found.")
                                path_result = [original_state, solution_state] # Hiển thị trạng thái tốt nhất
                            else:
                                print("Genetic Algorithm did not return a valid state.")
                                path_result = [] # Không có giải pháp
                        elif selected_algorithm == "Backtracking":
                            path_result = backtracking_search(original_state, target_state)
                    except Exception as e:
                         print(f"!!!!!!!!!!!!!! Error during {selected_algorithm} execution !!!!!!!!!!!!!!")
                         import traceback
                         traceback.print_exc() 
                         path_result = [] 

                    elapsed_time = time.time() - start_time

                    if path_result and isinstance(path_result, list) and len(path_result) > 0:
                         # Kiểm tra phần tử cuối cùng của path_result có phải goal không
                         is_goal_found = tuple(tuple(row) for row in path_result[-1]) == tuple(tuple(row) for row in target_state)
                         print(f"Solution found by {selected_algorithm}: {'Yes' if is_goal_found else 'No (showing path/state found)'}")
                         print(f"Time elapsed: {elapsed_time:.4f} seconds.")
                         steps = len(path_result) - 1
                         global_path = path_result # Lưu lại đường đi để xem steps
                         update_process_state(path_result) # Animate
                    else:
                         print(f"No solution path found by {selected_algorithm}.")
                         steps = None
                         global_path = [] # Xóa path cũ
                         process_state = deepcopy(original_state) # Reset lại process state nếu không có giải pháp

                    # Cập nhật global info
                    global_elapsed_time = elapsed_time
                    global_steps = steps

                else:
                    print("Please select an algorithm first!")

            # --- Xử lý nút "Random" ---
            random_button_x, random_button_y = 550, 280
            if not button_clicked and random_button_x <= x <= random_button_x + BUTTON_WIDTH and random_button_y <= y <= random_button_y + BUTTON_HEIGHT:
                button_clicked = True
                randomize_matrix() # Tạo ma trận mới và reset info

            # --- Xử lý nút "Steps" ---
            steps_button_x, steps_button_y = 550, 500
            if not button_clicked and steps_button_x <= x <= steps_button_x + BUTTON_WIDTH and steps_button_y <= y <= steps_button_y + BUTTON_HEIGHT:
                button_clicked = True
                print_steps_to_terminal(global_path) # In path đã lưu

            # Vẽ lại màn hình nếu có nút được nhấn
            if redraw_needed:
                draw_screen() # Vẽ lại trạng thái hiện tại
                pygame.display.flip()


pygame.quit()