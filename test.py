
def check_constraints(state):
    # Kiểm tra các ràng buộc của thuật toán
    for i in range(3):
        for j in range(3):
            if i > 0:  # Kiểm tra số phía trên
                if state[i][j] - state[i-1][j] < 3:
                    return False
            if j > 0:  # Kiểm tra số phía bên trái
                if state[i][j] - state[i][j-1] < 1:
                    return False
    return True

def backtracking_search(start_state, goal_state, max_depth=30):
    def is_goal(state):
        return state == goal_state

    def get_blank_position(state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j

    def get_neighbors(state):
        neighbors = []
        x, y = get_blank_position(state)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # lên, xuống, trái, phải
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = [row[:] for row in state]
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                
                # Kiểm tra các ràng buộc ngay tại đây
                if check_constraints(new_state):
                    neighbors.append(new_state)
        return neighbors

    def recursive_backtrack(state, assignment, depth):
        state_tuple = tuple(tuple(row) for row in state)
        if state_tuple in assignment:
            return None
        if depth > max_depth:
            return None
        if is_goal(state):
            return [state]

        assignment.add(state_tuple)

        for neighbor in get_neighbors(state):
            result = recursive_backtrack(neighbor, assignment, depth + 1)
            if result is not None:
                return [state] + result

        assignment.remove(state_tuple)
        return None

    result = recursive_backtrack(start_state, set(), 0)
    return result if result is not None else []

def print_solution(path):
    if not path:
        print("❌ Không tìm được lời giải trong độ sâu cho phép.")
        return
    print(f"✅ Tìm thấy lời giải trong {len(path)-1} bước:\n")
    for i, step in enumerate(path):
        print(f"Bước {i}:")
        for row in step:
            print(row)
        print("-" * 20)

def input_state():
    # Ma trận bắt đầu là rỗng
    numbers = [0] * 9  # Ma trận 3x3 rỗng

    print("🔢 Nhập vào từng số một cho ma trận 3x3 (số từ 1 đến 8, 0 là ô trống):")
    
    # Hàm để hiển thị trạng thái hiện tại của ma trận
    def display_state(state):
        print("📊 Trạng thái hiện tại:")
        for i in range(0, 9, 3):
            print(state[i:i+3])
        print("-" * 20)

    # Bắt đầu thuật toán backtracking với ma trận rỗng
    temp_state = [numbers[i:i+3] for i in range(0, 9, 3)]

    while True:
        try:
            # Nhập từng số vào ma trận
            num = int(input(f"Nhập số (1-8, 0 là ô trống): "))
            if num < 0 or num > 8:
                print("⚠️ Chỉ được nhập số từ 0 đến 8.")
                continue
            if num in numbers:
                print("⚠️ Số đã tồn tại trong ma trận.")
                continue

            # Tìm vị trí rỗng (0) để thay thế số mới nhập
            empty_pos = numbers.index(0)
            numbers[empty_pos] = num

            # Chuyển ma trận thành 3x3
            temp_state = [numbers[i:i+3] for i in range(0, 9, 3)]

            # Hiển thị trạng thái sau khi nhập số
            display_state(temp_state)

            # Kiểm tra các ràng buộc sau khi nhập
            if not check_constraints(temp_state):
                print("⚠️ Các ràng buộc không được thỏa mãn, vui lòng nhập lại số.")
                numbers[empty_pos] = 0  # Đặt lại ô rỗng
                display_state(temp_state)
                continue

            # Nếu ma trận đầy đủ (không còn ô trống), tiến hành tìm lời giải
            if numbers.count(0) == 0:
                print("\n🔍 Đang kiểm tra và tìm lời giải...\n")
                goal_state = [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 0]
                ]  # Trạng thái đích
                path = backtracking_search(temp_state, goal_state, max_depth=30)
                print_solution(path)
                break  # Kết thúc khi tìm được lời giải

        except ValueError:
            print("⚠️ Vui lòng nhập số nguyên.")
            continue


# ------------------- Chạy chương trình chính ----------------------

if __name__ == "__main__":
    print("🧩 GIẢI THUẬT BACKTRACKING CHO 8-PUZZLE")
    input_state()  # Bắt đầu nhập và kiểm tra
