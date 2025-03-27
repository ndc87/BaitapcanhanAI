import pygame
import sys
import random
from Solver import *

# Khởi tạo pygame
pygame.init()

# Kích thước cửa sổ
WIDTH, HEIGHT = 900, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8-Puzzle Solver")

# Màu sắc
BACKGROUND_COLOR = (255, 255, 255)
BUTTON_COLOR = (60, 60, 60)
BUTTON_HOVER_COLOR = (100, 100, 100)
TEXT_COLOR = (0, 0, 0)
GRID_COLOR = (255, 100, 100) 
GOAL_COLOR = (50, 200, 100)
EMPTY_COLOR = (180, 180, 180)

# Kích thước lưới
GRID_SIZE = 80
PADDING = 20
FONT = pygame.font.Font(None, 36)
BUTTON_FONT = pygame.font.Font(None, 28)

# Hiển thị nút bấm
def draw_button(text, x, y, width, height, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    
    color = BUTTON_HOVER_COLOR if x < mouse[0] < x + width and y < mouse[1] < y + height else BUTTON_COLOR
    pygame.draw.rect(screen, color, (x, y, width, height), border_radius=10)
    
    text_surf = BUTTON_FONT.render(text, True, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=(x + width / 2, y + height / 2))
    screen.blit(text_surf, text_rect)
    
    if action and click[0] == 1 and x < mouse[0] < x + width and y < mouse[1] < y + height:
        action()

def draw_text(text, x, y, screen, font_size=24, color=(0,0,0)):
    font = pygame.font.Font(None, font_size)  # Chọn font mặc định
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

# Hiển thị lưới trò chơi
def draw_grid(grid, x, y, goal=False):
    for row in range(3):
        for col in range(3):
            value = grid[row][col]
            rect_x, rect_y = x + col * GRID_SIZE, y + row * GRID_SIZE
            color = GOAL_COLOR if goal else (GRID_COLOR if value != 0 else EMPTY_COLOR)
            pygame.draw.rect(screen, color, (rect_x, rect_y, GRID_SIZE, GRID_SIZE), border_radius=15)
            pygame.draw.rect(screen, BACKGROUND_COLOR, (rect_x, rect_y, GRID_SIZE, GRID_SIZE), 3)
            if value:
                text_surf = FONT.render(str(value), True, TEXT_COLOR)
                text_rect = text_surf.get_rect(center=(rect_x + GRID_SIZE / 2, rect_y + GRID_SIZE / 2))
                screen.blit(text_surf, text_rect)

# Hiệu ứng cập nhật trạng thái
def animate_solution(solution):
    for state in solution:
        pygame.time.delay(300)
        screen.fill(BACKGROUND_COLOR)
        draw_grid(state, 350, 350)
        pygame.display.flip()

def shuffle_grid():
    numbers = list(range(9))
    random.shuffle(numbers)
    return [numbers[i:i+3] for i in range(0, 9, 3)]

def main():
    initial_state = shuffle_grid()
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    current_state = [row[:] for row in initial_state]

    # def solve(algorithm):
    #     nonlocal current_state
    #     if algorithm == "BFS":
    #         solution = solve_8puzzle_bfs(initial_state, goal_state)
    #     elif algorithm == "DFS":
    #         solution = solve_8puzzle_dfs(initial_state, goal_state)
    #     elif algorithm == "UCS":
    #         solution = solve_8puzzle_ucs(initial_state, goal_state)
    #     elif algorithm == "IDS":
    #         solution = solve_8puzzle_ids(initial_state, goal_state)
    #     elif algorithm == "GREEDY":
    #         solution = solve_8puzzle_greedy(initial_state, goal_state)
    #     elif algorithm == "A*":
    #         solution = solve_8puzzle_astar(initial_state, goal_state)
    #     elif algorithm == "IDA*":
    #         solution = solve_8puzzle_idastar(initial_state, goal_state)

    #     if solution:
    #         animate_solution(solution)
    #         current_state[:] = solution[-1]
    def solve(algorithm):
        nonlocal initial_state, current_state
        state_to_solve = [row[:] for row in initial_state]  # Copy trạng thái mới nhất

        if algorithm == "BFS":
            solution = solve_8puzzle_bfs(state_to_solve, goal_state)
        elif algorithm == "DFS":
            solution = solve_8puzzle_dfs(state_to_solve, goal_state)
        elif algorithm == "UCS":
            solution = solve_8puzzle_ucs(state_to_solve, goal_state)
        elif algorithm == "IDS":
            solution = solve_8puzzle_ids(state_to_solve, goal_state)
        elif algorithm == "GREEDY":
            solution = solve_8puzzle_greedy(state_to_solve, goal_state)
        elif algorithm == "A*":
            solution = solve_8puzzle_astar(state_to_solve, goal_state)
        elif algorithm == "IDA*":
            solution = solve_8puzzle_idastar(state_to_solve, goal_state)

        if solution:
            animate_solution(solution)
            current_state[:] = solution[-1]


    def reset():
        nonlocal initial_state, current_state
        initial_state = shuffle_grid()
        current_state = [row[:] for row in initial_state]

    running = True
    while running:
        screen.fill(BACKGROUND_COLOR)
        
        # Vẽ lưới
        draw_text("Initial State", 180, 30, screen)  # Chú thích trạng thái ban đầu
        draw_grid(initial_state, 180, 50)

        draw_text("Goal State", 600, 30, screen)  # Chú thích trạng thái mục tiêu
        draw_grid(goal_state, 600, 50, goal=True)

        draw_text("Current State", 350, 330, screen)  # Chú thích trạng thái hiện tại
        draw_grid(current_state, 350, 350)
        
        # Vẽ các nút chọn thuật toán
        algorithms = [
            ("BFS", lambda: solve("BFS")),
            ("DFS", lambda: solve("DFS")),
            ("UCS", lambda: solve("UCS")),
            ("IDS", lambda: solve("IDS")),
            ("GREEDY", lambda: solve("GREEDY")),
            ("IDA*", lambda: solve("IDA*")),
            ("A*", lambda: solve("A*"))

        ]

        for i, (name, func) in enumerate(algorithms):
            draw_button(name, 50, 100 + i * 60, 100, 40, func)
        
        # Vẽ nút reset
        draw_button("Reset", 50, HEIGHT - 100, 100, 40, reset)
        
        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()