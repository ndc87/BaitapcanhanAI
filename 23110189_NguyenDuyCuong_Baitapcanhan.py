import pygame
import time
import heapq
import random  
from collections import deque
import math
# Khởi tạo pygame
pygame.init()

# Kích thước cửa sổ
WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Matrix UI")

# Màu sắc
WHITE = (255, 248, 252)               # Trắng ngà hồng rất nhẹ, làm màu nền
BLACK = (60, 50, 60)                  # Xám đen mềm, không quá gắt
PINK = (240, 100, 140)                # Hồng đậm, dùng cho điểm nhấn
PROCESS_PINK = (255, 120, 180)        # Hồng sáng dùng cho nút đang được chọn
BUTTON_COLOR = (230, 70, 140)         # Hồng rực rỡ cho nút
BUTTON_TEXT_COLOR = (255, 255, 255)   # Giữ nguyên màu trắng cho chữ dễ đọc

# Kích thước ô vuông
CELL_SIZE = 60
MARGIN = 10
BORDER_RADIUS = 10
BUTTON_WIDTH, BUTTON_HEIGHT = 120, 40

# Danh sách thuật toán
algorithms = ["BFS", "DFS", "UCS", "IDS", "Greedy", "A*","IDA*","SHC","S_AHC","Stochastic","Annealing"]
selected_algorithm = None

# Biến toàn cục để lưu thời gian và số bước
global_elapsed_time = None
global_steps = None

def draw_matrix(matrix, start_x, start_y, color=PINK):
    font = pygame.font.Font(None, 36)
    for row in range(3):
        for col in range(3):
            value = matrix[row][col]
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
    for i, algo in enumerate(algorithms):
        button_x = 20
        button_y = 50 + i * (BUTTON_HEIGHT + 10)
        button_color = BUTTON_COLOR if selected_algorithm != algo else PROCESS_PINK
        pygame.draw.rect(SCREEN, button_color, (button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT), border_radius=5)
        pygame.draw.rect(SCREEN, BLACK, (button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT), 2, border_radius=5)
        text = font.render(algo, True, BUTTON_TEXT_COLOR)
        text_rect = text.get_rect(center=(button_x + BUTTON_WIDTH//2, button_y + BUTTON_HEIGHT//2))
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
    pygame.draw.rect(SCREEN, WHITE, (info_x, info_y, info_width, info_height))
    pygame.draw.rect(SCREEN, BLACK, (info_x, info_y, info_width, info_height), 2)
    
    title_text = font.render("Information", True, BLACK)
    title_rect = title_text.get_rect(center=(info_x + info_width // 2, info_y + 20))
    SCREEN.blit(title_text, title_rect)
    
    # Hiển thị thời gian
    if elapsed_time is not None:
        time_text = font.render(f"Time: {elapsed_time:.2f}s", True, BLACK)
    else:
        time_text = font.render("Time: --", True, BLACK)
    time_rect = time_text.get_rect(topleft=(info_x + 10, info_y + 40))
    SCREEN.blit(time_text, time_rect)
    
    # Hiển thị số bước
    if steps is not None:
        steps_text = font.render(f"Steps: {steps}", True, BLACK)
    else:
        steps_text = font.render("Steps: --", True, BLACK)
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
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j]
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
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j]
        )

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


def Simulated_Annealing(start_state, goal_state):
    def heuristic(state):
        """Hàm heuristic tính số ô sai vị trí."""
        return sum(
            1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j]
        )

    def get_neighbors(state):
        """Tìm các trạng thái hàng xóm của trạng thái hiện tại (các trạng thái có thể đạt được từ trạng thái hiện tại)."""
        neighbors = []
        zero_pos = next((i, j) for i in range(3) for j in range(3) if state[i][j] == 0)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Di chuyển lên, xuống, trái, phải
        
        for di, dj in directions:
            ni, nj = zero_pos[0] + di, zero_pos[1] + dj
            if 0 <= ni < 3 and 0 <= nj < 3:
                new_state = [row[:] for row in state]  # Sao chép trạng thái
                new_state[zero_pos[0]][zero_pos[1]], new_state[ni][nj] = new_state[ni][nj], new_state[zero_pos[0]][zero_pos[1]]
                neighbors.append(new_state)
        
        return neighbors

    # Thiết lập giá trị ban đầu cho nhiệt độ và số lần lặp
    initial_temp = 1000
    cooling_rate = 0.99
    max_iterations = 10000

    current_state = start_state
    current_temp = initial_temp
    path = [current_state]
    iterations = 0

    while iterations < max_iterations:
        # Tính toán heuristic cho trạng thái hiện tại
        current_heuristic = heuristic(current_state)
        
        # Nếu trạng thái hiện tại đã đạt được mục tiêu, thoát khỏi vòng lặp
        if current_state == goal_state:
            return path
        
        neighbors = get_neighbors(current_state)
        if not neighbors:
            break  # Không có hàng xóm để di chuyển
        
        # Chọn ngẫu nhiên một trạng thái hàng xóm
        next_state = random.choice(neighbors)
        next_heuristic = heuristic(next_state)
        
        # Tính toán sự chênh lệch heuristic
        delta_e = current_heuristic - next_heuristic
        
        # Nếu trạng thái mới tốt hơn, chấp nhận ngay
        # Nếu không tốt hơn, chấp nhận với một xác suất
        if delta_e > 0 or random.random() < math.exp(delta_e / current_temp):
            current_state = next_state
            path.append(current_state)
        
        # Giảm nhiệt độ dần dần
        current_temp *= cooling_rate
        iterations += 1
    
    return path if current_state == goal_state else []
# Ma trận trạng thái ban đầu và trạng thái đích
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

process_state = [
    [1, 2, 3],
    [4, 0, 6],
    [7, 5, 8]
]

def update_process_state(path):
    """Cập nhật trạng thái process_state theo từng bước trong path."""
    global process_state
    for state in path:
        process_state = state  # Cập nhật trạng thái hiện tại
        draw_screen(color=PROCESS_PINK)  # Vẽ lại màn hình với màu xanh dương nhạt
        pygame.time.delay(500)  # Tạm dừng 500ms để hiển thị từng bước

def draw_screen(color=PINK):
    """Vẽ toàn bộ màn hình."""
    SCREEN.fill(WHITE)
    draw_buttons()
    draw_action_buttons()
    draw_matrix(original_state, 220, 50)
    draw_matrix(target_state, 500, 50)
    draw_matrix(process_state, 220, 350, color=color)  # Vẽ ma trận process_state với màu được chỉ định
    draw_info_box(global_elapsed_time, global_steps)  # Truyền thời gian và số bước vào khung thông tin
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
    global original_state
    flattened = [i for row in target_state for i in row]  # Lấy tất cả các phần tử từ ma trận đích
    random.shuffle(flattened)  # Trộn ngẫu nhiên các phần tử
    original_state = [flattened[i:i + 3] for i in range(0, len(flattened), 3)]  # Chuyển về ma trận 3x3

# Vòng lặp chính
running = True
draw_screen()
while running:
    for event in pygame.event.get():
        draw_screen()
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            
            # Xử lý chọn thuật toán
            for i, algo in enumerate(algorithms):
                button_x = 20
                button_y = 50 + i * (BUTTON_HEIGHT + 10)
                if button_x <= x <= button_x + BUTTON_WIDTH and button_y <= y <= button_y + BUTTON_HEIGHT:
                    selected_algorithm = algo
            
            # Xử lý nút "Solve"
            solve_button_x, solve_button_y = 250, 280
            if solve_button_x <= x <= solve_button_x + BUTTON_WIDTH and solve_button_y <= y <= solve_button_y + BUTTON_HEIGHT:
                if selected_algorithm == "BFS":
                    start_time = time.time()  # Bắt đầu đo thời gian
                    path = bfs_solve(original_state, target_state)
                    elapsed_time = time.time() - start_time  # Tính thời gian thực thi
                    steps = len(path) - 1 if path else 0  # Tính số bước
                    if path:
                        update_process_state(path)  # Cập nhật trạng thái theo từng bước
                    else:
                        print("Không tìm thấy đường đi!")
                    
                    # Cập nhật giá trị toàn cục
                    global_elapsed_time = elapsed_time
                    global_steps = steps

                elif selected_algorithm == "DFS":
                    start_time = time.time()  # Bắt đầu đo thời gian
                    path = dfs_solve(original_state, target_state)
                    elapsed_time = time.time() - start_time  # Tính thời gian thực thi
                    steps = len(path) - 1 if path else 0  # Tính số bước
                    if path:
                        update_process_state(path)  # Cập nhật trạng thái theo từng bước
                    else:
                        print("Không tìm thấy đường đi!")
                    
                    # Cập nhật giá trị toàn cục
                    global_elapsed_time = elapsed_time
                    global_steps = steps

                elif selected_algorithm == "UCS":
                    start_time = time.time()  # Bắt đầu đo thời gian
                    path = ucs_solve(original_state, target_state)
                    elapsed_time = time.time() - start_time  # Tính thời gian thực thi
                    steps = len(path) - 1 if path else 0  # Tính số bước
                    if path:
                        update_process_state(path)  # Cập nhật trạng thái theo từng bước
                    else:
                        print("Không tìm thấy đường đi!")
                    global_elapsed_time = elapsed_time
                    global_steps = steps
                    
                elif selected_algorithm == "IDS":
                    start_time = time.time()  # Bắt đầu đo thời gian
                    path = ids_solve(original_state, target_state)
                    elapsed_time = time.time() - start_time  # Tính thời gian thực thi
                    steps = len(path) - 1 if path else 0  # Tính số bước
                    if path:
                        update_process_state(path)  # Cập nhật trạng thái theo từng bước
                    else:
                        print("Không tìm thấy đường đi!")
                    # Cập nhật giá trị toàn cục
                    global_elapsed_time = elapsed_time
                    global_steps = steps
                elif selected_algorithm == "Greedy":
                    start_time = time.time()  # Bắt đầu đo thời gian
                    path = greedy_solve(original_state, target_state)
                    elapsed_time = time.time() - start_time  # Tính thời gian thực thi
                    steps = len(path) - 1 if path else 0  # Tính số bước
                    if path:
                        update_process_state(path)  # Cập nhật trạng thái theo từng bước
                    else:
                        print("Không tìm thấy đường đi!")
                    global_elapsed_time = elapsed_time
                    global_steps = steps

                elif selected_algorithm == "A*":
                    start_time = time.time()  # Bắt đầu đo thời gian
                    path = a_star_solve(original_state, target_state)
                    elapsed_time = time.time() - start_time  # Tính thời gian thực thi
                    steps = len(path) - 1 if path else 0  # Tính số bước
                    if path:
                        update_process_state(path)  # Cập nhật trạng thái theo từng bước
                    else:
                        print("Không tìm thấy đường đi!")
                    global_elapsed_time = elapsed_time
                    global_steps = steps

                elif selected_algorithm == "IDA*":
                    start_time = time.time()  # Bắt đầu đo thời gian
                    path = ida_star_solve(original_state, target_state)
                    elapsed_time = time.time() - start_time  # Tính thời gian thực thi
                    steps = len(path) - 1 if path else 0  # Tính số bước
                    if path:
                        update_process_state(path)  # Cập nhật trạng thái theo từng bước
                    else:
                        print("Không tìm thấy đường đi!")
                    global_elapsed_time = elapsed_time
                    global_steps = steps
                elif selected_algorithm == "SHC":
                    start_time = time.time()  # Bắt đầu đo thời gian
                    path = shc_solve(original_state, target_state)
                    elapsed_time = time.time() - start_time  # Tính thời gian thực thi
                    steps = len(path) - 1 if path else 0  # Tính số bước
                    if path:
                        update_process_state(path)  # Cập nhật trạng thái theo từng bước
                    else:
                        print("Không tìm thấy đường đi!")
                    global_elapsed_time = elapsed_time
                    global_steps = steps

                elif selected_algorithm == "S_AHC":
                    start_time = time.time()  # Bắt đầu đo thời gian
                    path = s_ahc_solve(original_state, target_state)
                    elapsed_time = time.time() - start_time  # Tính thời gian thực thi
                    steps = len(path) - 1 if path else 0  # Tính số bước
                    if path:
                        update_process_state(path)  # Cập nhật trạng thái theo từng bước
                    else:
                        print("Không tìm thấy đường đi!")
                    global_elapsed_time = elapsed_time
                    global_steps = steps

                elif selected_algorithm == "Stochastic":
                    start_time = time.time()  # Bắt đầu đo thời gian
                    path = stochastic_solve(original_state, target_state)
                    elapsed_time = time.time() - start_time  # Tính thời gian thực thi
                    steps = len(path) - 1 if path else 0  # Tính số bước
                    if path:
                        update_process_state(path)  # Cập nhật trạng thái theo từng bước
                    else:
                        print("Không tìm thấy đường đi!")
                    global_elapsed_time = elapsed_time
                    global_steps = steps
                elif selected_algorithm == "Simulated_Annealing":
                    start_time = time.time()
                    path = Simulated_Annealing(original_state, target_state)
                    elapsed_time = time.time() - start_time
                    steps = len(path) - 1 if path else 0
                    if path:
                        update_process_state(path)
                    else:
                        print("Không tìm thấy đường đi!")
                    global_elapsed_time = elapsed_time
                    global_steps = steps

            # Xử lý nút "Steps"
            steps_button_x, steps_button_y = 550, 500
            if steps_button_x <= x <= steps_button_x + BUTTON_WIDTH and  steps_button_y <= y <= steps_button_y + BUTTON_HEIGHT:
                if path:  # Kiểm tra nếu đã có đường đi
                    print_steps_to_terminal(path)  # In ra các trạng thái trên terminal
                else:
                    print("Không có đường đi để hiển thị các bước!")
            
            # Xử lý nút "Random"
            random_button_x, random_button_y = 550, 280
            if random_button_x <= x <= random_button_x + BUTTON_WIDTH and random_button_y <= y <= random_button_y + BUTTON_HEIGHT:
                randomize_matrix()  # Tạo ma trận ngẫu nhiên
                draw_screen()  # Vẽ lại màn hình với ma trận mới

pygame.quit()