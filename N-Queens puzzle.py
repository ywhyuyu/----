import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import random
import time
import threading

# 定义NQueensMinConflict类，用于解决N皇后问题
class NQueensMinConflict:
    def __init__(self, n, max_steps=1000000):
        self.n = n  # 皇后数量
        self.max_steps = max_steps  # 最大迭代步数
        self.positions = [random.randint(0, n-1) for _ in range(n)]  # 每个皇后初始位置
        self.cols = {}  # 列冲突计数
        self.diag1 = {}  # 主对角线冲突计数
        self.diag2 = {}  # 副对角线冲突计数
        self.conflict_rows = set()  # 冲突行集合
        self._initialize_conflicts()  # 初始化冲突

    # 初始化冲突信息
    def _initialize_conflicts(self):
        for row, col in enumerate(self.positions):
            # 记录每列及对角线的冲突数
            self.cols[col] = self.cols.get(col, 0) + 1
            self.diag1[row - col] = self.diag1.get(row - col, 0) + 1
            self.diag2[row + col] = self.diag2.get(row + col, 0) + 1

        # 标记存在冲突的行
        for row, col in enumerate(self.positions):
            conflicts = self.cols[col] + self.diag1[row - col] + self.diag2[row + col] - 3
            if conflicts > 0:
                self.conflict_rows.add(row)

    # 使用最小冲突法解决N皇后问题
    def solve(self, update_callback=None):
        steps = 0  # 计步器
        while self.conflict_rows and steps < self.max_steps:
            # 随机选择冲突行
            row = random.choice(list(self.conflict_rows))
            current_col = self.positions[row]  # 当前列
            min_conflicts = self.n  # 最小冲突数初始化为最大值
            candidates = []  # 候选列列表

            # 遍历列，寻找最小冲突列
            for col in range(self.n):
                conflicts = self.cols.get(col, 0) + self.diag1.get(row - col, 0) + self.diag2.get(row + col, 0)
                if col == current_col:
                    conflicts -= 3  # 减去当前皇后贡献的冲突

                # 更新候选列
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    candidates = [col]
                elif conflicts == min_conflicts:
                    candidates.append(col)

            # 随机选择最小冲突列并移动皇后
            new_col = random.choice(candidates)
            self._move(row, new_col)
            steps += 1

            # 每1000步更新进度
            if update_callback and steps % 1000 == 0:
                update_callback(steps)

        return not self.conflict_rows

    # 移动皇后并更新冲突信息
    def _move(self, row, new_col):
        old_col = self.positions[row]
        # 更新旧列的冲突数
        self.cols[old_col] -= 1
        self.diag1[row - old_col] -= 1
        self.diag2[row + old_col] -= 1

        # 更新新列的冲突数
        self.positions[row] = new_col
        self.cols[new_col] = self.cols.get(new_col, 0) + 1
        self.diag1[row - new_col] = self.diag1.get(row - new_col, 0) + 1
        self.diag2[row + new_col] = self.diag2.get(row + new_col, 0) + 1

        # 更新冲突行集合
        conflicts = self.cols[new_col] + self.diag1[row - new_col] + self.diag2[row + new_col] - 3
        if conflicts > 0:
            self.conflict_rows.add(row)
        else:
            self.conflict_rows.discard(row)

        # 检查受影响的其他行
        affected_rows = []
        for r in range(self.n):
            if r != row:
                # 判断是否在旧列、新列或冲突对角线上
                if self.positions[r] == old_col or self.positions[r] == new_col:
                    affected_rows.append(r)
                if (r - self.positions[r] == row - old_col) or (r + self.positions[r] == row + old_col):
                    affected_rows.append(r)
                if (r - self.positions[r] == row - new_col) or (r + self.positions[r] == row + new_col):
                    affected_rows.append(r)
        
        # 更新受影响行的冲突信息
        for r in affected_rows:
            conflicts = self.cols[self.positions[r]] + self.diag1[r - self.positions[r]] + self.diag2[r + self.positions[r]] - 3
            if conflicts > 0:
                self.conflict_rows.add(r)
            else:
                self.conflict_rows.discard(r)

# 图形用户界面类，基于Tkinter实现
class NQueensGUI:
    def __init__(self, master, n):
        self.master = master
        self.n = n
        self.base_cell_size = max(5, 800 // self.n)  # 基础单元格大小
        self.zoom_factor = 1.0  # 当前缩放比例
        self.cell_size = self.base_cell_size  # 当前单元格大小

        # 创建上部框架放置按钮、进度条和状态标签
        top_frame = tk.Frame(master)
        top_frame.pack(side=tk.TOP, pady=10)

        self.start_button = tk.Button(top_frame, text="开始求解", command=self.start_solving)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.zoom_in_button = tk.Button(top_frame, text="放大", command=lambda: self.zoom(1.2))
        self.zoom_in_button.pack(side=tk.LEFT, padx=5)

        self.zoom_out_button = tk.Button(top_frame, text="缩小", command=lambda: self.zoom(1/1.2))
        self.zoom_out_button.pack(side=tk.LEFT, padx=5)

        # 添加进度条
        self.progress = ttk.Progressbar(top_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=10)

        self.status = tk.Label(top_frame, text="点击按钮开始求解")
        self.status.pack(side=tk.LEFT, padx=10)

        # 创建下部框架放置Canvas和滚动条
        bottom_frame = tk.Frame(master)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # 添加垂直和水平滚动条
        self.v_scroll = tk.Scrollbar(bottom_frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.h_scroll = tk.Scrollbar(bottom_frame, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # 创建画布显示棋盘
        self.canvas = tk.Canvas(bottom_frame, width=800, height=800, bg="white",
                                yscrollcommand=self.v_scroll.set,
                                xscrollcommand=self.h_scroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        # 绑定鼠标滚轮事件进行缩放
        self.canvas.bind("<MouseWheel>", self.mouse_zoom)  # Windows
        self.canvas.bind("<Button-4>", self.mouse_zoom)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.mouse_zoom)    # Linux scroll down

        self.solution = None
        self.chessboard_image = None  # PIL Image
        self.chessboard_photo = None   # ImageTk PhotoImage
        self.draw_chessboard()

    # 绘制棋盘
    def draw_chessboard(self):
        # 移除现有的棋盘图像
        self.canvas.delete("chessboard")

        # 创建棋盘图像
        self.chessboard_image = Image.new("RGB", (self.n, self.n), "white")
        draw = ImageDraw.Draw(self.chessboard_image)
        for row in range(self.n):
            for col in range(self.n):
                color = "white" if (row + col) % 2 == 0 else "black"
                draw.point((col, row), fill=color)

        # 放大图像以匹配当前单元格大小
        resized_image = self.chessboard_image.resize((self.n * self.cell_size, self.n * self.cell_size), Image.NEAREST)
        self.chessboard_photo = ImageTk.PhotoImage(resized_image)

        # 在Canvas上显示棋盘图像
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.chessboard_photo, tags="chessboard")

        # 更新scrollregion
        self.canvas.configure(scrollregion=(0, 0, self.n * self.cell_size, self.n * self.cell_size))

    # 绘制皇后
    def draw_queens(self):
        self.canvas.delete("queen")
        if self.solution:
            for row, col in enumerate(self.solution):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                # 根据N的大小，使用不同表示方式
                if self.n <= 100:
                    radius = self.cell_size / 2
                    self.canvas.create_oval(
                        x1 + radius/4, y1 + radius/4,
                        x2 - radius/4, y2 - radius/4,
                        fill="red", outline="", tags="queen"
                    )
                else:
                    padding = max(1, int(self.cell_size / 4))
                    self.canvas.create_rectangle(
                        x1 + padding, y1 + padding,
                        x2 - padding, y2 - padding,
                        fill="red", outline="", tags="queen"
                    )

    # 开始求解
    def start_solving(self):
        self.start_button.config(state=tk.DISABLED)
        self.zoom_in_button.config(state=tk.DISABLED)
        self.zoom_out_button.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.progress['maximum'] = 1000000
        self.status.config(text="正在求解...")
        self.solution = None
        self.draw_chessboard()
        self.canvas.delete("queen")
        threading.Thread(target=self.solve_and_display).start()

    # 启动求解并显示结果
    def solve_and_display(self):
        solver = NQueensMinConflict(self.n)
        start_time = time.time()

        # 定义一个回调函数用于更新进度
        def update_progress(steps):
            self.master.after(0, self.progress.config, {"value": steps})
            self.master.after(0, self.status.config, {"text": f"已执行 {steps} 步..."})

        success = solver.solve(update_callback=update_progress)
        end_time = time.time()
        if success:
            self.solution = solver.positions
            self.master.after(0, self.draw_queens)
            self.master.after(0, self.status.config, {"text": f"解决方案已找到！用时 {end_time - start_time:.2f} 秒"})
            self.master.after(0, self.progress.config, {"value": solver.max_steps})
        else:
            self.master.after(0, self.status.config, {"text": f"未能在 {solver.max_steps} 步内找到解决方案"})
            self.master.after(0, self.progress.config, {"value": solver.max_steps})

        self.master.after(0, self.start_button.config, {"state": tk.NORMAL})
        self.master.after(0, self.zoom_in_button.config, {"state": tk.NORMAL})
        self.master.after(0, self.zoom_out_button.config, {"state": tk.NORMAL})

    # 缩放功能
    def zoom(self, factor):
        new_zoom = self.zoom_factor * factor
        if new_zoom < 0.5:
            new_zoom = 0.5
            factor = new_zoom / self.zoom_factor
            self.zoom_factor = new_zoom
        elif new_zoom > 5:
            new_zoom = 5
            factor = new_zoom / self.zoom_factor
            self.zoom_factor = new_zoom
        else:
            self.zoom_factor = new_zoom

        self.cell_size = int(self.base_cell_size * self.zoom_factor)
        self.draw_chessboard()
        self.draw_queens()

    # 鼠标滚轮缩放
    def mouse_zoom(self, event):
        if event.delta > 0 or event.num == 4:
            factor = 1.1
        elif event.delta < 0 or event.num == 5:
            factor = 0.9
        else:
            factor = 1.0
        self.zoom(factor)

# 主函数
def main():
    n = 100000  # N皇后问题的规模
    root = tk.Tk()
    root.title(f"{n}-皇后问题 - 最小冲突法求解")
    gui = NQueensGUI(root, n)
    root.mainloop()

if __name__ == "__main__":
    main()
