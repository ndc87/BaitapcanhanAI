from collections import deque
import heapq
import random
import time
def solve_8puzzle_bfs(initial_state, goal_state):
    queue = deque([(initial_state, [])])
    visited = {state_to_tuple(initial_state)}
    while queue:
        state, path = queue.popleft()
        if state == goal_state:
            return path
        for next_state in get_next_states(state):
            if state_to_tuple(next_state) not in visited:
                visited.add(state_to_tuple(next_state))
                queue.append((next_state, path + [next_state]))
    return None

def solve_8puzzle_dfs(initial_state, goal_state, depth_limit=15):
    stack = [(initial_state, [], 0)]  # (state, path, depth)
    visited = set()

    while stack:
        state, path, depth = stack.pop()
        if state == goal_state:
            return path
        if depth < depth_limit:
            for next_state in get_next_states(state):
                if state_to_tuple(next_state) not in visited:
                    visited.add(state_to_tuple(next_state))
                    stack.append((next_state, path + [next_state], depth + 1))

    return None  # Không tìm thấy lời giải


def solve_8puzzle_ucs(initial_state, goal_state):
    pq = [(0, initial_state, [])]
    visited = {state_to_tuple(initial_state)}
    while pq:
        cost, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path
        for next_state in get_next_states(state):
            if state_to_tuple(next_state) not in visited:
                visited.add(state_to_tuple(next_state))
                heapq.heappush(pq, (cost + 1, next_state, path + [next_state]))
    return None

def solve_8puzzle_ids(initial_state, goal_state, max_depth=30):
    def dls(state, path, depth):
        if state == goal_state:
            return path
        if depth == 0:
            return None
        for next_state in get_next_states(state):
            if state_to_tuple(next_state) not in visited:
                visited.add(state_to_tuple(next_state))
                result = dls(next_state, path + [next_state], depth - 1)
                if result:
                    return result
        return None

    for depth in range(1, max_depth + 1):
        visited = set()
        result = dls(initial_state, [], depth)
        if result:
            return result
    return None  # Không tìm thấy lời giải


def solve_8puzzle_astar(initial_state, goal_state):
    def heuristic(state):
        return sum(abs(i - x) + abs(j - y)
                   for i, row in enumerate(state)
                   for j, val in enumerate(row)
                   if val and (x := (val - 1) // 3) is not None and (y := (val - 1) % 3) is not None)
    
    pq = [(heuristic(initial_state), 0, initial_state, [])]  # (f, g, state, path)
    visited = {state_to_tuple(initial_state)}
    
    while pq:
        _, g, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path
        for next_state in get_next_states(state):
            if state_to_tuple(next_state) not in visited:
                visited.add(state_to_tuple(next_state))
                f = g + 1 + heuristic(next_state)  # f = g + h
                heapq.heappush(pq, (f, g + 1, next_state, path + [next_state]))
    
    return None  # Không tìm thấy lời giải

def solve_8puzzle_greedy(initial_state, goal_state):
    def heuristic(state):
        return sum(abs(i - x) + abs(j - y)
                   for i, row in enumerate(state)
                   for j, val in enumerate(row)
                   if val and (x := (val - 1) // 3) is not None and (y := (val - 1) % 3) is not None)
    
    pq = [(heuristic(initial_state), initial_state, [])]
    visited = {state_to_tuple(initial_state)}
    while pq:
        _, state, path = heapq.heappop(pq)
        if state == goal_state:

            return path
        for next_state in get_next_states(state):
            if state_to_tuple(next_state) not in visited:
                visited.add(state_to_tuple(next_state))
                heapq.heappush(pq, (heuristic(next_state), next_state, path + [next_state]))
    return None



def solve_8puzzle_idastar(initial_state, goal_state):
    def heuristic(state):
        total_distance = 0
        for i, row in enumerate(state):
            for j, val in enumerate(row):
                if val != 0:  # Không tính khoảng cách ô trống
                    x, y = (val - 1) // 3, (val - 1) % 3
                    total_distance += abs(i - x) + abs(j - y)
        return total_distance

    def search(state, g, threshold, path, visited):
        f = g + heuristic(state)
        if f > threshold:
            return f, None
        if state == goal_state:
            return None, path

        min_threshold = float("inf")
        for next_state in get_next_states(state):
            state_tuple = state_to_tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                new_threshold, result = search(next_state, g + 1, threshold, path + [next_state], visited)
                if result is not None:
                    return None, result
                min_threshold = min(min_threshold, new_threshold)
        
        return min_threshold, None

    threshold = heuristic(initial_state)
    while True:
        visited = {state_to_tuple(initial_state)}
        new_threshold, result = search(initial_state, 0, threshold, [], visited)
        if result is not None:
            return result
        if new_threshold == float("inf"):
            return None
        threshold = new_threshold

def state_to_tuple(state):
    return tuple(tuple(row) for row in state)

def get_next_states(state):
    next_states = []
    i, j = find_zero(state)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for di, dj in moves:
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            new_state = [row[:] for row in state]
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
            next_states.append(new_state)
    return next_states

def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return None


def generate_random_state(goal_state, shuffle_moves=20):  # Giảm từ 30 xuống 20
    state = [row[:] for row in goal_state]
    for _ in range(shuffle_moves):
        next_states = get_next_states(state)
        state = random.choice(next_states)
    return state



def solve_8puzzle_hill_climbing(initial_state, goal_state, max_restarts=10):
    def heuristic(state):
        """Tính toán số ô sai vị trí so với trạng thái mục tiêu"""
        return sum(
            1 for i in range(3) for j in range(3)
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]
        )

    best_solution = None
    best_cost = float("inf")

    for _ in range(max_restarts):
        current_state = [row[:] for row in initial_state]  # BẮT ĐẦU từ initial_state của bạn
        path = [current_state]
        visited = set()
        visited.add(state_to_tuple(current_state))

        while True:
            next_states = get_next_states(current_state)
            next_states = [s for s in next_states if state_to_tuple(s) not in visited]

            if not next_states:
                break  # Không còn trạng thái để mở rộng => Dừng

            next_states.sort(key=heuristic)  # Chọn trạng thái có heuristic tốt nhất
            best_next_state = next_states[0]

            if heuristic(best_next_state) >= heuristic(current_state):
                break  # Không cải thiện được heuristic => Dừng

            current_state = best_next_state
            visited.add(state_to_tuple(current_state))
            path.append(current_state)

            if current_state == goal_state:
                return path  # Tìm thấy lời giải tốt nhất, trả về luôn

        # Kiểm tra nếu lời giải này tốt hơn lời giải trước đó
        if len(path) < best_cost:
            best_solution = path
            best_cost = len(path)

    return best_solution if best_solution else None  # Trả về lời giải tốt nhất tìm được

def solve_8puzzle__step_hill_climbing(initial_state, goal_state, max_restarts=10):
    def heuristic(state):
        return sum(1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal_state[i][j])
    best_solution = None
    best_cost = float("inf")
    for _ in range(max_restarts):
        current_state = [row[:] for row in initial_state]
        path = [current_state]
        visited = set()
        visited.add(state_to_tuple(current_state))
        while True:
            next_states = get_next_states(current_state)
            next_states = [s for s in next_states if state_to_tuple(s) not in visited]
            if not next_states:
                break
            next_states.sort(key=heuristic)
            best_next_state = next_states[0]
            if heuristic(best_next_state) >= heuristic(current_state):
                break
            current_state = best_next_state
            visited.add(state_to_tuple(current_state))
            path.append(current_state)
            if current_state == goal_state:
                return path
        if len(path) < best_cost:
            best_solution = path
            best_cost = len(path)
    return best_solution if best_solution else None

