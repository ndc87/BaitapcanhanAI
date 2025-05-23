import pygame
import time
import heapq
import random  # Thêm thư viện random để tạo ma trận ngẫu nhiên
from collections import deque
import math  # Thêm import math nếu chưa có
import tkinter as tk  # Import Tkinter for GUI selection
from tkinter import ttk

# Khởi tạo pygame
pygame.init()

# Kích thước cửa sổ
WIDTH, HEIGHT = 900, 800
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Matrix UI")

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 102, 204)
BUTTON_COLOR = (255, 105, 180)  # Màu hồng
PROCESS_BLUE = (255, 182, 193)  # Màu hồng nhạt khi được chọn
BUTTON_TEXT_COLOR = (255, 255, 255)
MATRIX_COLOR = (173, 216, 230)  # Màu xanh nhạt
TARGET_MATRIX_COLOR = (144, 238, 144)  # Màu xanh lá nhạt
PROCESS_MATRIX_COLOR = (240, 230, 140)  # Màu vàng nhạt

# Kích thước ô vuông
CELL_SIZE = 60
MARGIN = 10
BORDER_RADIUS = 10
BUTTON_WIDTH, BUTTON_HEIGHT = 120, 40

# Danh sách thuật toán
algorithms = ["BFS", "DFS", "UCS", "IDS", "Greedy", "A*", "IDA*", "SHC", "S_AHC", "Stochastic", "SA", "BeamSearch", "AND-OR", "Genetic", "Sensorless", "Backtracking","AC3"]
selected_algorithm = None

# Biến toàn cục để lưu thời gian và số bước
global_elapsed_time = None
global_steps = None

# Biến toàn cục để theo dõi vị trí cuộn
scroll_offset = 0  # Giá trị cuộn ban đầu
SCROLL_STEP = 20  # Bước cuộn mỗi lần

def draw_matrix(matrix, start_x, start_y, color=BLUE):
    font = pygame.font.Font(None, 36)
    for row in range(3):
        for col in range(3):
            # Kiểm tra xem matrix có phải là list hợp lệ không trước khi truy cập
            if matrix and isinstance(matrix, list) and row < len(matrix) and isinstance(matrix[row], list) and col < len(matrix[row]):
                value = matrix[row][col]
            else:
                # Xử lý trường hợp matrix không hợp lệ (ví dụ: None hoặc list rỗng)
                # Có thể vẽ ô trống hoặc bỏ qua
                pygame.draw.rect(SCREEN, BLACK, (start_x + col * (CELL_SIZE + MARGIN), start_y + row * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE), 1, border_radius=BORDER_RADIUS)
                continue  # Bỏ qua phần còn lại của vòng lặp cho ô này

            rect_x = start_x + col * (CELL_SIZE + MARGIN)
            rect_y = start_y + row * (CELL_SIZE + MARGIN)
            pygame.draw.rect(SCREEN, color, (rect_x, rect_y, CELL_SIZE, CELL_SIZE), border_radius=BORDER_RADIUS)
            pygame.draw.rect(SCREEN, BLACK, (rect_x, rect_y, CELL_SIZE, CELL_SIZE), 2, border_radius=BORDER_RADIUS)

            if value != 0:
                text = font.render(str(value), True, BLACK)
                text_rect = text.get_rect(center=(rect_x + CELL_SIZE // 2, rect_y + CELL_SIZE // 2))
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


        button_color = BUTTON_COLOR if selected_algorithm != algo else PROCESS_BLUE
        pygame.draw.rect(SCREEN, button_color, (button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT), border_radius=5)
        pygame.draw.rect(SCREEN, BLACK, (button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT), 2, border_radius=5)
        text = font.render(algo, True, BUTTON_TEXT_COLOR)
        text_rect = text.get_rect(center=(button_x + BUTTON_WIDTH // 2, button_y + BUTTON_HEIGHT // 2))
        SCREEN.blit(text, text_rect)


def handle_scroll(event):
    """Xử lý sự kiện cuộn chuột."""
    global scroll_offset
    visible_area_height = HEIGHT - 100  # Chiều cao vùng hiển thị
    max_scroll = max(0, len(algorithms) * (BUTTON_HEIGHT + 10) - visible_area_height)  # Giới hạn cuộn

    if event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 4:  # Cuộn lên
            scroll_offset = max(0, scroll_offset - SCROLL_STEP)
        elif event.button == 5:  # Cuộn xuống
            scroll_offset = min(max_scroll, scroll_offset + SCROLL_STEP)

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


# Thuật toán BFS
def bfs_solve(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    while queue:
        state, path = queue.popleft()
        if state == goal_state:
            return path + [state]  # Trả về đường đi bao gồm trạng thái cuối cùng
        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for neighbor in get_neighbors(state):
            queue.append((neighbor, path + [state]))
    return []  # Trả về danh sách rỗng nếu không tìm thấy đường đi

def get_neighbors(state):
    x, y = find_blank(state)
    moves = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            moves.append(new_state)
    return moves

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def dfs_solve(start_state, goal_state):
    stack = [(start_state, [])]
    visited = set()
    while stack:
        state, path = stack.pop()
        if state == goal_state:
            return path + [state]  # Trả về đường đi bao gồm trạng thái cuối cùng
        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for neighbor in get_neighbors(state):
            stack.append((neighbor, path + [state]))
    return []  # Trả về danh sách rỗng nếu không tìm thấy đường đi

def ucs_solve(start_state, goal_state):
    priority_queue = [(0, start_state, [])]  # (cost, state, path)
    visited = set()
    while priority_queue:
        cost, state, path = heapq.heappop(priority_queue)
        if state == goal_state:
            return path + [state]  # Trả về đường đi bao gồm trạng thái cuối cùng
        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for neighbor in get_neighbors(state):
            heapq.heappush(priority_queue, (cost + 1, neighbor, path + [state]))
    return []  # Trả về danh sách rỗng nếu không tìm thấy đường đi

def ids_solve(start_state, goal_state):
    def dls(state, goal, depth, path, visited):
        """Hàm tìm kiếm theo chiều sâu với giới hạn (Depth-Limited Search)."""
        if depth == 0 and state == goal:
            return path + [state]
        if depth > 0:
            state_tuple = tuple(tuple(row) for row in state)
            if state_tuple in visited:
                return None
            visited.add(state_tuple)
            for neighbor in get_neighbors(state):
                result = dls(neighbor, goal, depth - 1, path + [state], visited)
                if result:
                    return result
        return None

    # Lặp qua các giới hạn độ sâu
    for depth in range(1, 100):  # Giới hạn độ sâu tối đa là 100
        visited = set()
        result = dls(start_state, goal_state, depth, [], visited)
        if result:
            return result
    return []  # Trả về danh sách rỗng nếu không tìm thấy đường đi

def greedy_solve(start_state, goal_state):
    def heuristic(state):
        """Hàm heuristic tính số ô sai vị trí."""
        return sum(
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j]
        )

    priority_queue = [(heuristic(start_state), start_state, [])]  # (heuristic, state, path)
    visited = set()
    while priority_queue:
        _, state, path = heapq.heappop(priority_queue)
        if state == goal_state:
            return path + [state]
        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for neighbor in get_neighbors(state):
            heapq.heappush(priority_queue, (heuristic(neighbor), neighbor, path + [state]))
    return []  # Trả về danh sách rỗng nếu không tìm thấy đường đi

def a_star_solve(start_state, goal_state):
    def heuristic(state):
        """Hàm heuristic tính số ô sai vị trí."""
        return sum(
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j]
        )

    priority_queue = [(0 + heuristic(start_state), 0, start_state, [])]  # (f, g, state, path)
    visited = set()
    while priority_queue:
        f, g, state, path = heapq.heappop(priority_queue)
        if state == goal_state:
            return path + [state]
        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for neighbor in get_neighbors(state):
            heapq.heappush(priority_queue, (g + 1 + heuristic(neighbor), g + 1, neighbor, path + [state]))
    return []  # Trả về danh sách rỗng nếu không tìm thấy đường đi

def ida_star_solve(start_state, goal_state):
    def heuristic(state):
        """Hàm heuristic tính số ô sai vị trí."""
        return sum(
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j]
        )

    def search(state, path, g, threshold):
        f = g + heuristic(state)
        if f > threshold:
            return f, None
        if state == goal_state:
            return f, path + [state]
        min_threshold = float('inf')
        state_tuple = tuple(tuple(row) for row in state)
        visited.add(state_tuple)
        for neighbor in get_neighbors(state):
            if tuple(tuple(row) for row in neighbor) not in visited:
                t, result = search(neighbor, path + [state], g + 1, threshold)
                if result:
                    return t, result
                min_threshold = min(min_threshold, t)
        visited.remove(state_tuple)
        return min_threshold, None

    threshold = heuristic(start_state)
    while True:
        visited = set()
        t, result = search(start_state, [], 0, threshold)
        if result:
            return result
        if t == float('inf'):
            return []  # Không tìm thấy đường đi
        threshold = t

def shc_solve(start_state, goal_state):
    def heuristic(state):
        """Hàm heuristic tính số ô sai vị trí."""
        return sum(
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j]
        )

    current_state = start_state
    path = [current_state]
    while current_state != goal_state:
        neighbors = get_neighbors(current_state)
        next_state = min(neighbors, key=heuristic, default=None)
        if next_state is None or heuristic(next_state) >= heuristic(current_state):
            break  # Không thể cải thiện thêm
        current_state = next_state
        path.append(current_state)
    return path if current_state == goal_state else []  # Trả về đường đi nếu tìm thấy

def s_ahc_solve(start_state, goal_state):
    def heuristic(state):
        """Hàm heuristic tính số ô sai vị trí."""
        return sum(
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and  state[i][j] != goal_state[i][j]
        )

    current_state = start_state
    path = [current_state]
    while current_state != goal_state:
        neighbors = get_neighbors(current_state)
        next_state = min(neighbors, key=heuristic, default=None)  # Chọn trạng thái có heuristic tốt nhất
        if next_state is None or heuristic(next_state) >= heuristic(current_state):
            break  # Không thể cải thiện thêm
        current_state = next_state
        path.append(current_state)
    return path if current_state == goal_state else []  # Trả về đường đi nếu tìm thấy

def stochastic_solve(start_state, goal_state):
    def heuristic(state):
        """Hàm heuristic tính số ô sai vị trí."""
        return sum(
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j])

    current_state = start_state
    path = [current_state]
    while current_state != goal_state:
        neighbors = get_neighbors(current_state)
        if not neighbors:
            break  # Không có hàng xóm để di chuyển
        # Chọn ngẫu nhiên một trạng thái hàng xóm
        next_state = random.choice(neighbors)
        if heuristic(next_state) < heuristic(current_state):
            current_state = next_state
            path.append(current_state)
    return path if current_state == goal_state else []  # Trả về đường đi nếu tìm thấy


def simulated_annealing_solve(initial_state, goal_state):
    def heuristic(state):
        """Hàm heuristic tính tổng khoảng cách Manhattan."""
        return sum(
            abs(i - goal_i) + abs(j - goal_j)
            for i in range(3) for j in range(3)
            if state[i][j] != 0
            for goal_i in range(3) for goal_j in range(3)
            if goal_state[goal_i][goal_j] == state[i][j]
        )
    
    def probability(delta_e, temperature):
        """Tính xác suất chấp nhận trạng thái kém hơn."""
        return math.exp(-delta_e / temperature)

    current_state = initial_state
    current_heuristic = heuristic(current_state)
    temperature = 500  # Giảm nhiệt độ ban đầu
    cooling_rate = 0.99  # Tăng tốc độ giảm nhiệt độ
    min_temperature = 0.1  # Nhiệt độ tối thiểu
    path = [current_state]  # Lưu đường đi
    steps = 0
    start_time = time.time()

    while temperature > min_temperature:
        steps += 1

        neighbors = get_neighbors(current_state)
        if not neighbors:
            break  # Không có hàng xóm để di chuyển

        # Chọn trạng thái hàng xóm tốt hơn hoặc ngẫu nhiên
        next_state = min(neighbors, key=heuristic)
        next_heuristic = heuristic(next_state)
        delta_e = next_heuristic - current_heuristic

        # Chấp nhận trạng thái mới nếu tốt hơn hoặc theo xác suất
        if delta_e < 0 or random.random() < probability(delta_e, temperature):
            current_state = next_state
            current_heuristic = next_heuristic
            path.append(current_state)

        # Kiểm tra trạng thái đích
        if current_state == goal_state:
            end_time = time.time()
            return path, {
                "steps_checked": steps,
                "path_length": len(path),
                "time": end_time - start_time
            }

        # Giảm nhiệt độ
        temperature *= cooling_rate

    # Nếu không tìm thấy lời giải
    end_time = time.time()
    return path, {
        "error": "Không tìm thấy lời giải!",
        "steps_checked": steps,
        "path_length": len(path),
        "time": end_time - start_time
    }


def beam_search_solve(initial_state, goal_state, beam_width=2):
    
    def heuristic(state, goal_state):
        """Hàm heuristic tính tổng khoảng cách Manhattan."""
        return sum(
            abs(i - goal_i) + abs(j - goal_j)
            for i in range(3) for j in range(3)
            if state[i][j] != 0
            for goal_i in range(3) for goal_j in range(3)
            if goal_state[goal_i][goal_j] == state[i][j]
        )

    start_time = time.time()
    steps = 0
    visited = set()

    # Khởi tạo danh sách các trạng thái hiện tại (beam)
    beam = [(initial_state, [], heuristic(initial_state, goal_state))]  # (trạng thái, đường đi, heuristic)

    while beam:
        steps += 1
        next_beam = []

        for current_state, path, current_heuristic in beam:
            # Kiểm tra nếu đã đạt trạng thái đích
            if current_state == goal_state:
                end_time = time.time()
                result_info = {
                    "steps_checked": steps,
                    "path_length": len(path) + 1,
                    "time": end_time - start_time,
                    "states_visited": len(visited)
                }
                return path + [current_state], result_info

            # Đánh dấu trạng thái hiện tại là đã thăm
            visited.add(tuple(tuple(row) for row in current_state))

            # Lấy tất cả các trạng thái lân cận
            for neighbor in get_neighbors(current_state):
                neighbor_tuple = tuple(tuple(row) for row in neighbor)
                if neighbor_tuple not in visited:
                    new_path = path + [current_state]
                    neighbor_heuristic = heuristic(neighbor, goal_state)
                    next_beam.append((neighbor, new_path, neighbor_heuristic))

        # Sắp xếp các trạng thái lân cận theo heuristic và chỉ giữ lại `beam_width` trạng thái tốt nhất
        next_beam.sort(key=lambda x: x[2])  # Sắp xếp theo giá trị heuristic
        beam = next_beam[:beam_width]

    # Nếu không tìm thấy lời giải
    end_time = time.time()
    return None, {
        "error": "Không tìm thấy lời giải!",
        "steps_checked": steps,
        "time": end_time - start_time,
        "states_visited": len(visited)
    }



def or_and_search(state, goal_state, get_successors, node_type='OR', visited=None):
    if visited is None:
        visited = set()

    state_tuple = tuple(tuple(row) for row in state)
    if state == goal_state:
        return [state]
    if state_tuple in visited:
        return None

    visited.add(state_tuple)

    if node_type == 'OR':
        for next_state in get_successors(state):
            result = or_and_search(next_state, goal_state, get_successors, node_type='AND', visited=visited.copy())
            if result:
                return [state] + result
    else:  # AND node
        all_results = [state]
        for next_state in get_successors(state):
            result = or_and_search(next_state, goal_state, get_successors, node_type='OR', visited=visited.copy())
            if not result:
                return None
            all_results += result
        return all_results

    return None

def genetic_algorithm_solve(start_state, goal_state, population_size=100, generations=100, mutation_rate=0.1):
    def fitness(state):
        """Hàm fitness tính số ô sai vị trí."""
        return sum(
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j]
        )

    def initialize_population():
        """Khởi tạo quần thể ngẫu nhiên."""
        population = []
        for _ in range(population_size):
            flattened = [i for row in goal_state for i in row]
            random.shuffle(flattened)
            individual = [flattened[i:i + 3] for i in range(0, len(flattened), 3)]
            population.append(individual)
        return population

    def select_parents(population):
        """Chọn lọc cha mẹ dựa trên fitness."""
        population.sort(key=fitness)
        return population[:2]  # Chọn 2 cá thể tốt nhất

    def crossover(parent1, parent2):
        """Lai ghép hai cá thể."""
        child = [row[:] for row in parent1]
        for i in range(3):
            for j in range(3):
                if random.random() > 0.5:
                    child[i][j] = parent2[i][j]
        return child

    def mutate(individual):
        """Đột biến một cá thể."""
        if random.random() < mutation_rate:
            x1, y1 = random.randint(0, 2), random.randint(0, 2)
            x2, y2 = random.randint(0, 2), random.randint(0, 2)
            individual[x1][y1], individual[x2][y2] = individual[x2][y2], individual[x1][y1]

    # Khởi tạo quần thể
    population = initialize_population()

    for generation in range(generations):
        # Nếu tìm thấy trạng thái đích, trả về đường đi
        for individual in population:
            if individual == goal_state:
                return [start_state, individual]

        # Chọn lọc và tạo thế hệ mới
        parents = select_parents(population)
        new_population = []
        for _ in range(population_size // 2):
            child1 = crossover(parents[0], parents[1])
            child2 = crossover(parents[1], parents[0])
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = new_population

    return []  # Trả về danh sách rỗng nếu không tìm thấy đường đi
def sensorless_solve(start_state, goal_state):
    """
    Thuật toán Sensorless tìm kiếm trạng thái đích mà không biết vị trí chính xác của các ô.
    """
    def heuristic(state):
        """Hàm heuristic tính số ô sai vị trí."""
        return sum(
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j]
        )

    # Khởi tạo tập hợp các trạng thái có thể từ vị trí hiện tại và các lân cận
    possible_states = set()
    for neighbor in get_neighbors(start_state):
        possible_states.add(tuple(tuple(row) for row in neighbor))
    possible_states.add(tuple(tuple(row) for row in start_state))  # Thêm trạng thái hiện tại
    path = []
    visited = set()  # Thêm bộ nhớ để tránh lặp lại các trạng thái



    while possible_states:
        # Chọn trạng thái có heuristic thấp nhất
        current_state = min(possible_states, key=heuristic)
        path.append(current_state)
        possible_states.remove(current_state)

        # Kiểm tra xem trạng thái hiện tại đã đạt đích chưa
        if current_state == tuple(tuple(row) for row in goal_state):
            return [list(map(list, state)) for state in path]  # Trả về đường đi

        # Sinh các trạng thái hàng xóm từ trạng thái hiện tại
        neighbors = set()
        for state in get_neighbors([list(row) for row in current_state]):
            state_tuple = tuple(tuple(row) for row in state)
            if state_tuple not in visited:
                neighbors.add(state_tuple)
                visited.add(state_tuple)
        
        # Cập nhật lại các trạng thái có thể
        possible_states = neighbors

    return []  #


# def backtracking(algorithm, goal_state):
   
#     def is_valid_state(state):
#         values = {state[i][j] for i in range(3) for j in range(3)}
#         return values == set(range(9))

#     def backtrack(state, used, i=0, j=0):
#         if i == 3:
#             if is_valid_state(state):
#                 for row in state:
#                     print(row)
#                 path = None
#                 if algorithm == "BFS":
#                     path = bfs_solve(state, goal_state)
#                 elif algorithm == "DFS":
#                     path = dfs_solve(state, goal_state)
#                 elif algorithm == "UCS":
#                     path = ucs_solve(state, goal_state)
#                 elif algorithm == "IDS":
#                     path = ids_solve(state, goal_state)
#                 elif algorithm == "Greedy":
#                     path = greedy_solve(state, goal_state)
#                 elif algorithm == "A*":
#                     path = a_star_solve(state, goal_state)
#                 elif algorithm == "IDA*":
#                     path = ida_star_solve(state, goal_state)
#                 elif algorithm == "SHC":
#                     path = shc_solve(state, goal_state)
#                 elif algorithm == "S_AHC":
#                     path = s_ahc_solve(state, goal_state)
#                 elif algorithm == "Stochastic":
#                     path = stochastic_solve(state, goal_state)
#                 elif algorithm == "SA":
#                     path = simulated_annealing_solve(state, goal_state)
#                 elif algorithm == "BeamSearch":
#                     path = beam_search_solve(state, goal_state)
#                 elif algorithm == "AND-OR":
#                     path = or_and_search(state, goal_state, get_neighbors)
#                 elif algorithm == "Genetic":
#                     path = genetic_algorithm_solve(state, goal_state)
#                 elif algorithm == "Sensorless":
#                     path = sensorless_solve(state, goal_state)

#                 if path:
#                     messagebox.showinfo("Success", f"Found a valid start state for algorithm: {algorithm}")
#                     return state
#             return None

#         ni, nj = (i, j + 1) if j < 2 else (i + 1, 0)

#         for num in range(9):
#             if num not in used:
#                 state[i][j] = num
#                 used.add(num)
#                 result = backtrack(state, used, ni, nj)
#                 if result:
#                     return result
#                 used.remove(num)
#                 state[i][j] = 0  # Quay lui

#         return None

#     ori = [[0] * 3 for _ in range(3)]
#     return backtrack(ori, set())


def backtracking(algorithm, goal_state):
    """
    Sử dụng thuật toán Backtracking để tạo ra trạng thái ban đầu hợp lệ và giải bài toán.
    """
    def is_valid_state(state):
        """Kiểm tra xem trạng thái có hợp lệ không."""
        values = {state[i][j] for i in range(3) for j in range(3)}
        return values == set(range(9))

    def backtrack(state, used, i=0, j=0):
        """Thuật toán Backtracking để tạo trạng thái hợp lệ."""
        if i == 3:
            if is_valid_state(state):
                for row in state:
                    print(row)
                path = None
                if algorithm == "BFS":
                    path = bfs_solve(state, goal_state)
                elif algorithm == "DFS":
                    path = dfs_solve(state, goal_state)
                elif algorithm == "UCS":
                    path = ucs_solve(state, goal_state)
                elif algorithm == "IDS":
                    path = ids_solve(state, goal_state)
                elif algorithm == "Greedy":
                    path = greedy_solve(state, goal_state)
                elif algorithm == "A*":
                    path = a_star_solve(state, goal_state)
                elif algorithm == "IDA*":
                    path = ida_star_solve(state, goal_state)
                elif algorithm == "SHC":
                    path = shc_solve(state, goal_state)
                elif algorithm == "S_AHC":
                    path = s_ahc_solve(state, goal_state)
                elif algorithm == "Stochastic":
                    path = stochastic_solve(state, goal_state)
                elif algorithm == "SA":
                    path, _ = simulated_annealing_solve(state, goal_state)
                elif algorithm == "BeamSearch":
                    path, _ = beam_search_solve(state, goal_state)
                elif algorithm == "AND-OR":
                    path = or_and_search(state, goal_state, get_neighbors)
                elif algorithm == "Genetic":
                    path = genetic_algorithm_solve(state, goal_state)
                elif algorithm == "Sensorless":
                    path = sensorless_solve(state, goal_state)

                if path:
                    print("Backtracking tìm thấy đường đi:")
                    for step in path:
                        print(step)
                    return state
            return None

        ni, nj = (i, j + 1) if j < 2 else (i + 1, 0)

        for num in range(9):
            if num not in used:
                state[i][j] = num
                used.add(num)
                result = backtrack(state, used, ni, nj)
                if result:
                    return result
                used.remove(num)
                state[i][j] = 0  # Quay lui

        return None

    initial_state = [[0] * 3 for _ in range(3)]
    return backtrack(initial_state, set())

def is_solvable(state):
    flat = sum(state, [])
    inv_count = 0
    for i in range(len(flat)):
        for j in range(i+1, len(flat)):
            if flat[i] != 0 and flat[j] != 0 and flat[i] > flat[j]:
                inv_count += 1
    return inv_count % 2 == 0


def ac3_generate_and_solve(algorithm, goal_state):
    """
    Sử dụng thuật toán AC3 để tạo ra trạng thái ban đầu hợp lệ và giải bài toán.
    """
    print("AC-3")
    variables = [(i, j) for i in range(3) for j in range(3)]

    def constraints(xi, vi, xj, vj):
        """Ràng buộc: Các giá trị trong ma trận phải khác nhau."""
        return vi != vj

    def revise(xi, xj, domains):
        """Loại bỏ các giá trị không hợp lệ khỏi miền giá trị."""
        revised = False
        for vi in domains[xi][:]:
            if all(not constraints(xi, vi, xj, vj) for vj in domains[xj]):
                domains[xi].remove(vi)
                revised = True
        return revised

    def ac3(domains):
        """Thuật toán AC3 để duy trì tính nhất quán của các ràng buộc."""
        arcs = deque([(xi, xj) for xi in variables for xj in variables if xi != xj])
        while arcs:
            xi, xj = arcs.popleft()
            if revise(xi, xj, domains):
                if not domains[xi]:
                    return False
                for xk in variables:
                    if xk != xi and xk != xj:
                        arcs.append((xk, xi))
        return True

    def generate_valid_assignment():
        """Tạo một trạng thái hợp lệ bằng AC3."""
        while True:
            # Tạo miền giá trị ngẫu nhiên
            all_values = list(range(9))
            random.shuffle(all_values)
            domains = {var: [val] for var, val in zip(variables, all_values)}

            if not ac3(domains):
                continue

            # Gán giá trị duy nhất cho mỗi biến
            assignment = [[0] * 3 for _ in range(3)]
            used = set()
            for (i, j) in variables:
                for val in domains[(i, j)]:
                    if val not in used:
                        assignment[i][j] = val
                        used.add(val)
                        break

            if len(used) == 9 and is_solvable(assignment):
                return assignment

    assignment = generate_valid_assignment()

    for row in assignment:
        print(row)

    # Giải bài toán với trạng thái ban đầu được tạo bởi AC3
    path = None
    if algorithm == "BFS":
        path = bfs_solve(assignment, goal_state)
    elif algorithm == "DFS":
        path = dfs_solve(assignment, goal_state)
    elif algorithm == "UCS":
        path = ucs_solve(assignment, goal_state)
    elif algorithm == "IDS":
        path = ids_solve(assignment, goal_state)
    elif algorithm == "Greedy":
        path = greedy_solve(assignment, goal_state)
    elif algorithm == "A*":
        path = a_star_solve(assignment, goal_state)
    elif algorithm == "IDA*":
        path = ida_star_solve(assignment, goal_state)
    elif algorithm == "SHC":
        path = shc_solve(assignment, goal_state)
    elif algorithm == "S_AHC":
        path = s_ahc_solve(assignment, goal_state)
    elif algorithm == "Stochastic":
        path = stochastic_solve(assignment, goal_state)
    elif algorithm == "SA":
        path, _ = simulated_annealing_solve(assignment, goal_state)
    elif algorithm == "BeamSearch":
        path, _ = beam_search_solve(assignment, goal_state)
    elif algorithm == "AND-OR":
        path = or_and_search(assignment, goal_state, get_neighbors)
    elif algorithm == "Genetic":
        path = genetic_algorithm_solve(assignment, goal_state)
    elif algorithm == "Sensorless":
        path = sensorless_solve(assignment, goal_state)

    if path:
        print("AC3 tìm thấy đường đi:")
        for step in path:
            print(step)
        return assignment

    return None


def show_algorithm_selector():
    """Hiển thị cửa sổ Tkinter để chọn thuật toán và căn giữa màn hình."""
    def on_select():
        global selected_algorithm
        selected_algorithm = algo_var.get()
        root.destroy()

    root = tk.Tk()
    root.title("Select Algorithm")

    # Lọc các thuật toán, loại bỏ "Backtracking" và "AC-3"
    filtered_algorithms = [algo for algo in algorithms if algo not in ["Backtracking", "AC-3"]]
    tk.Label(root, text="Choose an algorithm to solve:").pack(pady=10)
    algo_var = tk.StringVar(value=filtered_algorithms[0] if filtered_algorithms else "")
    algo_dropdown = ttk.Combobox(root, textvariable=algo_var, values=filtered_algorithms, state="readonly")
    algo_dropdown.pack(pady=10)
    tk.Button(root, text="OK", command=on_select).pack(pady=10)

    # Cập nhật kích thước giao diện và căn giữa
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - w) // 2
    y = (screen_h - h) // 2
    root.geometry(f'{w}x{h}+{x}+{y}')

    root.mainloop()
# Ma trận trạng thái ban đầu và trạng thái đích
original_state = [
     [2, 6, 5],
    [0, 8, 7],
    [4,3,1]
]

target_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

process_state = original_state

def update_process_state(path):
    """Cập nhật trạng thái process_state theo từng bước trong path."""
    global process_state
    for state in path:
        process_state = state  # Cập nhật trạng thái hiện tại
        draw_screen(color=PROCESS_BLUE)  # Vẽ lại màn hình với màu xanh dương nhạt
        pygame.time.delay(500)  # Tạm dừng 500ms để hiển thị từng bước

def draw_screen(color=BLUE):
    """Vẽ toàn bộ màn hình."""
    SCREEN.fill(WHITE)
    draw_buttons()
    draw_action_buttons()
    draw_matrix(original_state, 220, 50, color=MATRIX_COLOR)  # Ma trận trạng thái ban đầu
    draw_matrix(target_state, 500, 50, color=TARGET_MATRIX_COLOR)  # Ma trận trạng thái đích
    draw_matrix(process_state, 220, 350, color=PROCESS_MATRIX_COLOR)  # Ma trận trạng thái đang xử lý
    draw_info_box(global_elapsed_time, global_steps)  # Vẽ khung thông tin
    pygame.display.flip()

def print_steps_to_terminal(path):
    """In ra các ma trận trạng thái trong terminal."""
    print("Các trạng thái trong đường đi:")
    for i, state in enumerate(path):
        print(f"Step {i}:")
        for row in state:
            print(row)
        print()  # Dòng trống giữa các bước

def randomize_matrix():
    """Tạo ma trận ngẫu nhiên."""
    global original_state, process_state
    flattened = [i for row in target_state for i in row]  # Lấy tất cả các phần tử từ ma trận đích
    random.shuffle(flattened)  # Trộn ngẫu nhiên các phần tử
    original_state = [flattened[i:i + 3] for i in range(0, len(flattened), 3)]  # Chuyển về ma trận 3x3
    process_state = [row[:] for row in original_state]  # Đồng bộ process_state với original_state

def draw_info_box(elapsed_time, steps):
    """Vẽ hộp thông tin chứa thời gian và số bước."""
    font = pygame.font.Font(None, 30)
    time_text = font.render(f"Time: {elapsed_time:.2f}s", True, (0, 0, 0))
    steps_text = font.render(f"Steps: {steps}", True, (0, 0, 0))
    SCREEN.blit(time_text, (600, 400))
    SCREEN.blit(steps_text, (600, 450))

# Hàm tiện ích để thực thi thuật toán
def execute_algorithm(solve_func, start_state, target_state):
    start_time = time.time()
    path = solve_func(start_state, target_state)
    elapsed_time = time.time() - start_time
    steps = len(path) - 1 if path else 0
    if path:
        update_process_state(path)
    else:
        print("Không tìm thấy đường đi!")
    return path, elapsed_time, steps

# Hàm tiện ích để cập nhật giao diện
def update_screen(state, target_state, elapsed_time, steps):
    SCREEN.fill(WHITE)
    draw_buttons()
    draw_action_buttons()
    draw_matrix(state, 220, 50)
    draw_matrix(target_state, 500, 50)
    draw_matrix(state, 220, 350)
    draw_info_box(elapsed_time, steps)
    pygame.display.flip()

# Hàm hiển thị bộ chọn thuật toán
def show_algorithm_selector():
    """Hiển thị cửa sổ Tkinter để chọn thuật toán và căn giữa màn hình."""
    def on_select():
        global selected_algorithm
        selected_algorithm = algo_var.get()
        root.destroy()

    root = tk.Tk()
    root.title("Select Algorithm")

    # Lọc các thuật toán, loại bỏ "Backtracking" và "AC-3"
    filtered_algorithms = [algo for algo in algorithms if algo not in ["Backtracking", "AC-3"]]
    tk.Label(root, text="Choose an algorithm to solve:").pack(pady=10)
    algo_var = tk.StringVar(value=filtered_algorithms[0] if filtered_algorithms else "")
    algo_dropdown = ttk.Combobox(root, textvariable=algo_var, values=filtered_algorithms, state="readonly")
    algo_dropdown.pack(pady=10)
    tk.Button(root, text="OK", command=on_select).pack(pady=10)

    # Cập nhật kích thước giao diện và căn giữa
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - w) // 2
    y = (screen_h - h) // 2
    root.geometry(f'{w}x{h}+{x}+{y}')

    root.mainloop()

# Hàm hiển thị từng bước của đường đi
def show_path_steps(path):
    if path:
        for state in path:
            update_screen(state, target_state, global_elapsed_time, global_steps)
            pygame.time.wait(500)  # Đợi 500ms giữa các bước
    else:
        print("Không có đường đi để hiển thị!")

# Vòng lặp chính
running = True
path = None
selected_algorithm = None
global_elapsed_time = 0
global_steps = 0
draw_screen()

# Từ điển ánh xạ thuật toán
algorithm_map = {
    "BFS": bfs_solve,
    "DFS": dfs_solve,
    "UCS": ucs_solve,
    "IDS": ids_solve,
    "Greedy": greedy_solve,
    "A*": a_star_solve,
    "IDA*": ida_star_solve,
    "SHC": shc_solve,
    "S_AHC": s_ahc_solve,
    "Stochastic": stochastic_solve,
    "SA": simulated_annealing_solve,
    "BeamSearch": lambda s, t: beam_search_solve(s, t, beam_width=2),
    "AND-OR": lambda s, t: or_and_search(s, t, get_neighbors),
    "Genetic": genetic_algorithm_solve,
    "Sensorless": sensorless_solve
}

while running:
    for event in pygame.event.get():
        handle_scroll(event)
        draw_screen()
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos

            # Xử lý chọn thuật toán
            num_algorithms = len(algorithms)
            half = (num_algorithms + 1) // 2
            for i, algo in enumerate(algorithms):
                col_index = i // half
                row_index = i % half
                button_x = 20 if col_index == 0 else WIDTH - BUTTON_WIDTH - 20
                button_y = 50 + row_index * (BUTTON_HEIGHT + 10) - scroll_offset
                if button_x <= x <= button_x + BUTTON_WIDTH and button_y <= y <= button_y + BUTTON_HEIGHT:
                    selected_algorithm = algo

            # Xử lý nút "Solve"
            solve_button_x, solve_button_y = 250, 280
            if solve_button_x <= x <= solve_button_x + BUTTON_WIDTH and solve_button_y <= y <= solve_button_y + BUTTON_HEIGHT:
                if selected_algorithm in ["AC-3", "Backtracking"]:
                    # Khởi tạo trạng thái ban đầu
                    state = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    update_screen(state, target_state, global_elapsed_time, global_steps)
                    show_algorithm_selector()
                    tmp_state = original_state
                    
                    # Gọi hàm tạo trạng thái ban đầu
                    original_state = (ac3_generate_and_solve if selected_algorithm == "AC-3" else backtracking)(selected_algorithm, target_state)
                    
                    if original_state:
                        print("Trạng thái ban đầu tìm thấy:")
                        for row in original_state:
                            print(row)
                        update_screen(original_state, target_state, global_elapsed_time, global_steps)
                        # Tiếp tục giải từ trạng thái ban đầu đến trạng thái đích
                        path, elapsed_time, steps = execute_algorithm(algorithm_map["BFS"], original_state, target_state)
                        global_elapsed_time, global_steps = elapsed_time, steps
                        show_path_steps(path)
                    else:
                        print("Không tìm thấy trạng thái ban đầu hợp lệ!")
                        original_state = tmp_state
                        path = None
                elif selected_algorithm in algorithm_map:
                    path, elapsed_time, steps = execute_algorithm(algorithm_map[selected_algorithm], original_state, target_state)
                    global_elapsed_time, global_steps = elapsed_time, steps
                    show_path_steps(path)
                else:
                    print(f"Thuật toán {selected_algorithm} không được hỗ trợ!")
                    path = None

            # Xử lý nút "Steps"
            steps_button_x, steps_button_y = 550, 500
            if steps_button_x <= x <= steps_button_x + BUTTON_WIDTH and steps_button_y <= y <= steps_button_y + BUTTON_HEIGHT:
                if path:
                    print_steps_to_terminal(path)
                else:
                    print("Không có đường đi để hiển thị các bước!")

            # Xử lý nút "Random"
            random_button_x, random_button_y = 550, 280
            if random_button_x <= x <= random_button_x + BUTTON_WIDTH and random_button_y <= y <= random_button_y + BUTTON_HEIGHT:
                randomize_matrix()
                draw_screen()

        handle_scroll(event)

pygame.quit()