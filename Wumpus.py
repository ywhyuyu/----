import tkinter as tk
from tkinter import messagebox
import random
from collections import deque

# 定义方向
DIRECTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

# 定义环境中的感知类型
PERCEPTS = ['stench', 'breeze', 'glitter', 'bump', 'scream']

class Environment:
    def __init__(self, size, pit_probability):
        self.size = size
        self.pit_probability = pit_probability
        self.grid = [[{'pit': False, 'wumpus': False, 'gold': False} for _ in range(self.size)] for _ in range(self.size)]
        self.agent_position = (0, 0)
        self.agent_direction = 'RIGHT'
        self.wumpus_alive = True
        self.scream_heard = False

        self.init_environment()

    def init_environment(self):
        # 放置 Wumpus
        self.place_wumpus()
        # 放置 Gold
        self.place_gold()
        # 放置 Pits
        self.place_pits()

    def place_wumpus(self):
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) != (0, 0) and not self.grid[y][x]['gold']:
                self.grid[y][x]['wumpus'] = True
                break

    def place_gold(self):
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) != (0, 0) and not self.grid[y][x]['pit'] and not self.grid[y][x]['wumpus']:
                self.grid[y][x]['gold'] = True
                break

    def place_pits(self):
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) != (0, 0):
                    if random.random() < self.pit_probability:
                        if not self.grid[y][x]['gold'] and not self.grid[y][x]['wumpus']:
                            self.grid[y][x]['pit'] = True

    def get_percepts(self, position):
        x, y = position
        percepts = []
        # 检查当前房间
        if self.grid[y][x]['gold']:
            percepts.append('glitter')
        # 检查周围的房间
        adjacent_rooms = self.get_adjacent_rooms((x, y))
        stench = False
        breeze = False
        for (ax, ay) in adjacent_rooms:
            if self.grid[ay][ax]['wumpus'] and self.wumpus_alive:
                stench = True
            if self.grid[ay][ax]['pit']:
                breeze = True
        if stench:
            percepts.append('stench')
        if breeze:
            percepts.append('breeze')
        return percepts

    def get_adjacent_rooms(self, position):
        x, y = position
        adjacent = []
        if x > 0:
            adjacent.append((x - 1, y))
        if x < self.size - 1:
            adjacent.append((x + 1, y))
        if y > 0:
            adjacent.append((x, y - 1))
        if y < self.size - 1:
            adjacent.append((x, y + 1))
        return adjacent

    def move_agent(self, action):
        x, y = self.agent_position
        direction = self.agent_direction
        bump = False

        if action == 'Forward':
            if direction == 'UP':
                new_position = (x, y - 1)
            elif direction == 'DOWN':
                new_position = (x, y + 1)
            elif direction == 'LEFT':
                new_position = (x - 1, y)
            elif direction == 'RIGHT':
                new_position = (x + 1, y)

            if self.is_valid_position(new_position):
                self.agent_position = new_position
            else:
                bump = True
        elif action == 'TurnLeft':
            idx = DIRECTIONS.index(direction)
            self.agent_direction = DIRECTIONS[(idx - 1) % 4]
        elif action == 'TurnRight':
            idx = DIRECTIONS.index(direction)
            self.agent_direction = DIRECTIONS[(idx + 1) % 4]
        return bump

    def is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.size and 0 <= y < self.size

    def shoot_arrow(self):
        # 箭射出的方向
        x, y = self.agent_position
        direction = self.agent_direction

        dx, dy = 0, 0
        if direction == 'UP':
            dy = -1
        elif direction == 'DOWN':
            dy = 1
        elif direction == 'LEFT':
            dx = -1
        elif direction == 'RIGHT':
            dx = 1

        nx, ny = x + dx, y + dy
        while self.is_valid_position((nx, ny)):
            if self.grid[ny][nx]['wumpus'] and self.wumpus_alive:
                self.grid[ny][nx]['wumpus'] = False
                self.wumpus_alive = False
                self.scream_heard = True
                return True
            nx += dx
            ny += dy
        return False

class KnowledgeBase:
    def __init__(self, size):
        self.size = size
        self.safe = set()
        self.unsafe = set()
        self.visited = set()
        self.potential_pits = {}
        self.potential_wumpus = {}
        self.wumpus_alive = True
        self.wumpus_position = None
        self.pit_locations = set()

        # 初始化所有未知房间的状态
        self.prob_pit = {}
        self.prob_wumpus = {}
        for y in range(size):
            for x in range(size):
                position = (x, y)
                if position != (0, 0):
                    self.prob_pit[position] = 0.2  # 初始设为陷阱概率
                    self.prob_wumpus[position] = 1 / ((size * size) - 1)  # 初始均匀分布

    def update(self, position, percepts):
        self.visited.add(position)
        self.safe.add(position)

        x, y = position
        adjacent = self.get_adjacent_rooms(position)

        # 基于感知更新概率
        if 'breeze' not in percepts:
            for adj in adjacent:
                if adj not in self.safe:
                    self.prob_pit[adj] = 0
        else:
            # 如果有微风，邻居可能有陷阱
            unknown_adj = [adj for adj in adjacent if adj not in self.safe and adj not in self.visited]
            if unknown_adj:
                prob = 1 / len(unknown_adj)
                for adj in unknown_adj:
                    self.prob_pit[adj] = max(self.prob_pit.get(adj, 0), prob)
        if 'stench' not in percepts:
            for adj in adjacent:
                if adj not in self.safe:
                    self.prob_wumpus[adj] = 0
        else:
            # 如果有恶臭，邻居可能有 Wumpus
            unknown_adj = [adj for adj in adjacent if adj not in self.safe and adj not in self.visited]
            if unknown_adj:
                prob = 1 / len(unknown_adj)
                for adj in unknown_adj:
                    self.prob_wumpus[adj] = max(self.prob_wumpus.get(adj, 0), prob)

        # 更新安全的房间
        for pos in self.prob_pit:
            if self.prob_pit[pos] == 0 and self.prob_wumpus[pos] == 0 and pos not in self.safe:
                self.safe.add(pos)

        # 确定 Wumpus 的位置
        if self.wumpus_alive:
            wumpus_candidates = [pos for pos, prob in self.prob_wumpus.items() if prob > 0]
            if len(wumpus_candidates) == 1:
                self.wumpus_position = wumpus_candidates[0]
                print(f"推断出 Wumpus 在位置 {self.wumpus_position}")

    def get_adjacent_rooms(self, position):
        x, y = position
        adjacent = []
        if x > 0:
            adjacent.append((x - 1, y))
        if x < self.size - 1:
            adjacent.append((x + 1, y))
        if y > 0:
            adjacent.append((x, y - 1))
        if y < self.size - 1:
            adjacent.append((x, y + 1))
        return adjacent

    def get_safe_unvisited(self):
        return self.safe - self.visited

    def is_safe(self, position):
        if position in self.safe:
            return True
        if position in self.visited:
            return True
        # 如果位置的陷阱和 Wumpus 概率都是 0，认为安全
        if self.prob_pit.get(position, 1) == 0 and self.prob_wumpus.get(position, 1) == 0:
            self.safe.add(position)
            return True
        return False

    def get_least_risky_unvisited(self):
        unvisited = set(self.prob_pit.keys()) - self.visited
        # 计算总风险
        risks = {pos: self.prob_pit.get(pos, 0) + self.prob_wumpus.get(pos, 0) for pos in unvisited}
        # 按照风险排序
        sorted_risks = sorted(risks.items(), key=lambda item: item[1])
        return [pos for pos, risk in sorted_risks if risk < 1]  # 只考虑风险小于1的房间

class Agent:
    def __init__(self, environment, gui):
        self.env = environment
        self.gui = gui
        self.position = self.env.agent_position
        self.direction = self.env.agent_direction
        self.kb = KnowledgeBase(self.env.size)
        self.has_gold = False
        self.arrow = True
        self.alive = True
        self.score = 0
        self.path = []
        self.plan = deque()
        self.steps = []  # 记录每一步的位置

        self.state = "Waiting to start"

    def perceive(self):
        percepts = self.env.get_percepts(self.position)
        if self.env.scream_heard:
            percepts.append('scream')
            self.env.scream_heard = False
            self.kb.wumpus_alive = False
            self.kb.wumpus_position = None
        return percepts

    def update_kb(self, percepts):
        self.kb.update(self.position, percepts)

    def decide_action(self, percepts):
        # 如果在当前房间发现金子，抓取
        if 'glitter' in percepts and not self.has_gold:
            return 'Grab'

        # 如果知道 Wumpus 的位置并且有箭，尝试射击
        if self.arrow and self.kb.wumpus_position and self.kb.wumpus_alive:
            action = self.plan_shoot()
            if action:
                return action

        # 如果没有计划，生成新的计划
        if not self.plan:
            self.create_plan()
        if self.plan:
            return self.plan.popleft()
        else:
            # 当没有安全的未访问房间，选择风险最小的房间前进
            least_risky = self.kb.get_least_risky_unvisited()
            if least_risky:
                path = self.find_path(self.position, set(least_risky), consider_risk=True)
                if path:
                    actions = self.path_to_actions(path)
                    self.plan.extend(actions)
                    return self.plan.popleft()
            # 如果没有可行路径，回到起点
            if self.position != (0, 0):
                path = self.find_path(self.position, {(0, 0)})
                if path:
                    actions = self.path_to_actions(path)
                    self.plan.extend(actions)
                    self.plan.append('Climb')
            else:
                self.plan.append('Climb')
            if self.plan:
                return self.plan.popleft()
            else:
                return 'Climb'

    def plan_shoot(self):
        # 判断 Wumpus 是否在当前方向的直线上，如果是，朝向它并射击
        x, y = self.position
        wx, wy = self.kb.wumpus_position
        required_direction = None
        if x == wx and y != wy:
            required_direction = 'UP' if wy < y else 'DOWN'
        elif y == wy and x != wx:
            required_direction = 'LEFT' if wx < x else 'RIGHT'

        if required_direction:
            turn_actions = self.get_turn_actions(self.direction, required_direction)
            self.plan.extend(turn_actions)
            self.plan.append('Shoot')
            return self.plan.popleft()
        else:
            # 尝试移动到与 Wumpus 在同一行或列的位置
            path = self.find_path(self.position, {pos for pos in self.kb.safe if pos[0] == wx or pos[1] == wy})
            if path:
                actions = self.path_to_actions(path)
                self.plan.extend(actions)
                return self.plan.popleft()
            else:
                return None

    def create_plan(self):
        # 如果已经获得金子，规划回家的路径
        if self.has_gold:
            path = self.find_path(self.position, {(0, 0)})
            if path:
                actions = self.path_to_actions(path)
                self.plan.extend(actions)
                self.plan.append('Climb')
        else:
            # 优先去未访问且安全的房间
            safe_unvisited = {pos for pos in self.kb.safe if self.kb.is_safe(pos)} - self.kb.visited
            if safe_unvisited:
                path = self.find_path(self.position, safe_unvisited)
                if path:
                    actions = self.path_to_actions(path)
                    self.plan.extend(actions)

    def find_path(self, start, goals, avoid_wumpus=True, consider_risk=False):
        # 使用 BFS 寻找最短路径
        queue = deque()
        queue.append((start, []))
        visited = set()
        while queue:
            current, path = queue.popleft()
            if current in goals:
                return path
            visited.add(current)
            for neighbor in self.kb.get_adjacent_rooms(current):
                if neighbor not in visited:
                    if self.kb.is_safe(neighbor):
                        queue.append((neighbor, path + [neighbor]))
                    elif consider_risk:
                        risk = self.kb.prob_pit.get(neighbor, 0) + self.kb.prob_wumpus.get(neighbor, 0)
                        if risk < 1:
                            queue.append((neighbor, path + [neighbor]))
        return None

    def path_to_actions(self, path):
        actions = []
        current_direction = self.direction
        current_position = self.position
        for next_position in path:
            required_direction = self.get_required_direction(current_position, next_position)
            turn_actions = self.get_turn_actions(current_direction, required_direction)
            actions.extend(turn_actions)
            actions.append('Forward')
            current_direction = required_direction
            current_position = next_position
        self.direction = current_direction
        return actions

    def get_required_direction(self, current, next):
        x1, y1 = current
        x2, y2 = next
        if x2 == x1 + 1:
            return 'RIGHT'
        elif x2 == x1 - 1:
            return 'LEFT'
        elif y2 == y1 - 1:
            return 'UP'
        elif y2 == y1 + 1:
            return 'DOWN'

    def get_turn_actions(self, current_direction, required_direction):
        actions = []
        idx_current = DIRECTIONS.index(current_direction)
        idx_required = DIRECTIONS.index(required_direction)
        diff = (idx_required - idx_current) % 4
        if diff == 0:
            return actions
        elif diff == 1:
            actions.append('TurnRight')
        elif diff == 2:
            actions.append('TurnRight')
            actions.append('TurnRight')
        elif diff == 3:
            actions.append('TurnLeft')
        return actions

    def act(self, action):
        self.score -= 1  # 每个动作扣 1 分
        if action == 'Forward':
            self.state = "Move"
            self.gui.update_state(self.state)
            bump = self.env.move_agent('Forward')
            self.position = self.env.agent_position
            self.steps.append(self.position)
            self.gui.update_grid()
            if bump:
                # 感知到撞击
                pass
            x, y = self.position
            # 检查是否死亡
            if self.env.grid[y][x]['wumpus'] and self.env.wumpus_alive:
                self.alive = False
                self.score -= 1000
                self.state = "DEATH"
                self.gui.update_state(self.state)
                self.gui.update_score(self.score)
                print("被 Wumpus 吞噬，游戏结束！")
            elif self.env.grid[y][x]['pit']:
                self.alive = False
                self.score -= 1000
                self.state = "DEATH"
                self.gui.update_state(self.state)
                self.gui.update_score(self.score)
                print("掉入陷阱，游戏结束！")
        elif action == 'TurnLeft':
            self.env.move_agent('TurnLeft')
            self.direction = self.env.agent_direction
        elif action == 'TurnRight':
            self.env.move_agent('TurnRight')
            self.direction = self.env.agent_direction
        elif action == 'Grab':
            x, y = self.position
            if self.env.grid[y][x]['gold']:
                self.has_gold = True
                self.env.grid[y][x]['gold'] = False
                self.state = "Find Gold!"
                self.gui.update_state(self.state)
                self.gui.update_grid()
                print("成功抓取金子！")
                # 规划回家的路径
                self.plan.clear()
                self.create_plan()
        elif action == 'Climb':
            if self.position == (0, 0):
                if self.has_gold:
                    self.score += 1000
                    self.state = "BackHome"
                    self.gui.update_state(self.state)
                    print("成功带着金子逃出洞穴，胜利！")
                else:
                    print("安全逃出洞穴，游戏结束！")
                self.alive = False
            else:
                print("只能在起始位置攀爬离开！")
        elif action == 'Shoot':
            if self.arrow:
                self.score -= 10  # 射箭扣 10 分
                self.arrow = False
                result = self.env.shoot_arrow()
                self.gui.update_grid()
                if result:
                    self.state = "KillMonster"
                    self.gui.update_state(self.state)
                    print("成功射杀 Wumpus！")
                    self.kb.wumpus_alive = False
                    self.kb.wumpus_position = None
                else:
                    print("射箭未命中。")
            else:
                print("已经没有箭了！")
        self.gui.update_score(self.score)

    def run(self):
        if not self.alive:
            return
        percepts = self.perceive()
        self.update_kb(percepts)
        action = self.decide_action(percepts)
        self.act(action)
        if action == 'Climb' or not self.alive:
            self.gui.update_state("Game Over")
            return
        # 延迟执行下一个动作
        self.gui.root.after(500, self.run)

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Wumpus World")
        self.setup_parameters()
        self.root.mainloop()

    def setup_parameters(self):
        # 参数设置界面
        param_frame = tk.Frame(self.root)
        param_frame.pack(pady=10)

        tk.Label(param_frame, text="Map Size:").grid(row=0, column=0)
        self.size_entry = tk.Entry(param_frame)
        self.size_entry.insert(0, "4")
        self.size_entry.grid(row=0, column=1)

        tk.Label(param_frame, text="Pit Probability (0~1):").grid(row=1, column=0)
        self.pit_entry = tk.Entry(param_frame)
        self.pit_entry.insert(0, "0.2")
        self.pit_entry.grid(row=1, column=1)

        self.start_button = tk.Button(param_frame, text="Start Game", command=self.start_game)
        self.start_button.grid(row=2, column=0, columnspan=2, pady=5)

    def start_game(self):
        try:
            size = int(self.size_entry.get())
            pit_prob = float(self.pit_entry.get())
            if size < 2 or pit_prob < 0 or pit_prob > 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid parameters.")
            return

        # 清除参数设置界面
        for widget in self.root.winfo_children():
            widget.destroy()

        self.canvas_size = 400
        self.cell_size = self.canvas_size // size
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack()
        self.score_label = tk.Label(self.info_frame, text="Score: 0")
        self.score_label.pack(side=tk.LEFT, padx=10)
        self.state_label = tk.Label(self.info_frame, text="State: Waiting to start")
        self.state_label.pack(side=tk.LEFT, padx=10)

        self.env = Environment(size, pit_prob)
        self.agent = Agent(self.env, self)
        self.draw_grid()
        self.draw_legend()  # 绘制图例

        # 开始游戏
        self.state_label.config(text="State: Move")
        self.agent.state = "Move"
        self.agent.run()

    def draw_grid(self):
        self.cells = {}
        size = self.env.size
        for y in range(size):
            for x in range(size):
                x1 = x * self.cell_size
                y1 = y * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                cell = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
                self.cells[(x, y)] = cell
        self.update_grid()

    def update_grid(self):
        size = self.env.size
        for y in range(size):
            for x in range(size):
                cell = self.cells[(x, y)]
                items = self.env.grid[y][x]
                # 清除之前的图标
                self.canvas.delete("icon_{0}_{1}".format(x, y))
                # 绘制元素
                x_center = x * self.cell_size + self.cell_size // 2
                y_center = y * self.cell_size + self.cell_size // 2
                if items['pit']:
                    self.canvas.create_oval(
                        x_center - 10, y_center - 10, x_center + 10, y_center + 10,
                        fill="black", tags="icon_{0}_{1}".format(x, y)
                    )
                if items['wumpus'] and self.env.wumpus_alive:
                    self.canvas.create_rectangle(
                        x_center - 10, y_center - 10, x_center + 10, y_center + 10,
                        fill="red", tags="icon_{0}_{1}".format(x, y)
                    )
                if items['gold']:
                    self.canvas.create_polygon(
                        x_center, y_center - 10, x_center - 10, y_center + 10,
                        x_center + 10, y_center + 10,
                        fill="yellow", tags="icon_{0}_{1}".format(x, y)
                    )
                if (x, y) == self.env.agent_position and self.agent.alive:
                    self.canvas.create_polygon(
                        x_center, y_center - 10, x_center - 10, y_center + 10,
                        x_center + 10, y_center + 10,
                        fill="blue", tags="icon_{0}_{1}".format(x, y)
                    )
                elif (x, y) in self.agent.steps:
                    # 标记已访问的房间
                    self.canvas.create_text(
                        x_center, y_center, text="·", font=("Arial", 24),
                        tags="icon_{0}_{1}".format(x, y)
                    )
        self.canvas.update()

    def update_score(self, score):
        self.score_label.config(text=f"Score: {score}")

    def update_state(self, state):
        self.state_label.config(text=f"State: {state}")

    def draw_legend(self):
        # 在界面下方添加图例
        legend_frame = tk.Frame(self.root)
        legend_frame.pack()
        # 创建颜色和描述的对应关系
        legends = [
            ("蓝色三角形", "智能体"),
            ("黄色三角形", "金子"),
            ("红色方块", "Wumpus"),
            ("黑色圆形", "陷阱"),
            ("·", "已访问房间"),
        ]
        for color, desc in legends:
            legend_item = tk.Frame(legend_frame)
            legend_item.pack(side=tk.LEFT, padx=5)
            color_canvas = tk.Canvas(legend_item, width=20, height=20)
            color_canvas.pack(side=tk.LEFT)
            if color == "蓝色三角形":
                color_canvas.create_polygon(10, 0, 0, 20, 20, 20, fill="blue")
            elif color == "黄色三角形":
                color_canvas.create_polygon(10, 0, 0, 20, 20, 20, fill="yellow")
            elif color == "红色方块":
                color_canvas.create_rectangle(5, 5, 15, 15, fill="red")
            elif color == "黑色圆形":
                color_canvas.create_oval(5, 5, 15, 15, fill="black")
            elif color == "·":
                color_canvas.create_text(10, 10, text="·", font=("Arial", 16))
            label = tk.Label(legend_item, text=desc)
            label.pack(side=tk.LEFT)

def main():
    gui = GUI()

if __name__ == "__main__":
    main()
