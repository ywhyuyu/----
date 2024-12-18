import pandas as pd
import heapq
import time
import matplotlib.pyplot as plt
import networkx as nx
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import queue

# 加载 Excel 文件
file_path = r'E:\QQdownloads\罗马尼亚度假问题地图及启发函数值.xlsx'  # 请确保路径正确
xls = pd.ExcelFile(file_path)

# 读取距离数据和启发函数数据
distance_df = pd.read_excel(xls, 'Distances')
heuristic_df = pd.read_excel(xls, 'Heuristics', header=None, names=['City', 'HeuristicValue'])

# 构建邻接表图结构
graph = {}
cities = list(distance_df.columns[1:])  # 提取城市名列表，跳过第一列的空白列名
for index, row in distance_df.iterrows():
    city_a = row['Unnamed: 0']  # 每行第一个元素是城市名
    graph[city_a] = []
    for i, distance in enumerate(row[1:]):  # 跳过第一列
        city_b = cities[i]
        if distance != 1000:  # 跳过距离为 1000 的值，表示没有直接连接
            graph[city_a].append((city_b, distance))

# 去除邻接表中的自连接项
for city in graph:
    graph[city] = [(neighbor, distance) for neighbor, distance in graph[city] if city != neighbor]

# 构建启发函数字典
heuristic = {}
for _, row in heuristic_df.iterrows():
    city = row['City']
    heuristic_value = row['HeuristicValue']
    heuristic[city] = heuristic_value

# 绘制图形界面
G = nx.Graph()
for city, neighbors in graph.items():
    for neighbor, distance in neighbors:
        G.add_edge(city, neighbor, weight=distance)

# 初始化Matplotlib图形
plt.ion()  # 开启交互模式
fig, ax = plt.subplots(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)  # 固定布局
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold', ax=ax)
edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
plt.title("路径搜索动态示意")
plt.show(block=False)

# 创建队列用于线程间通信
update_queue = queue.Queue()

# 搜索算法的实现
# 代价一致的宽度优先算法
def uniform_cost_search(graph, start, goal, update_queue):
    start_time = time.perf_counter()  # 使用 perf_counter 记录开始时间
    nodes_expanded = 0  # 节点扩展计数器

    # 初始化优先队列
    frontier = []
    heapq.heappush(frontier, (0, start, [start]))  # (路径代价, 当前城市, 路径)

    visited = set()

    while frontier:
        cost, current_city, path = heapq.heappop(frontier)
        nodes_expanded += 1  # 扩展节点计数

        # 如果找到了目标城市，发送最终路径和代价
        if current_city == goal:
            end_time = time.perf_counter()  # 使用 perf_counter 记录结束时间
            update_queue.put({
                'type': 'found',
                'path': path,
                'cost': cost,
                'nodes_expanded': nodes_expanded,
                'runtime': end_time - start_time
            })
            return

        # 标记当前城市为已访问
        if current_city not in visited:
            visited.add(current_city)

            # 扩展当前城市的相邻城市
            for neighbor, distance in graph.get(current_city, []):
                if neighbor not in visited:
                    total_cost = cost + distance
                    heapq.heappush(frontier, (total_cost, neighbor, path + [neighbor]))
                    # 发送当前扩展的节点
                    update_queue.put({
                        'type': 'expand',
                        'current_city': current_city,
                        'neighbor': neighbor,
                        'cost': total_cost,
                        'path': path + [neighbor]
                    })

    # 如果没有找到路径，发送未找到的信息
    update_queue.put({
        'type': 'not_found',
        'path': None,
        'cost': float('inf'),
        'nodes_expanded': nodes_expanded,
        'runtime': time.perf_counter() - start_time
    })

# 贪心最佳优先搜索算法
def greedy_best_first_search(graph, heuristic, start, goal, update_queue):
    start_time = time.perf_counter()  # 使用 perf_counter 记录开始时间
    nodes_expanded = 0  # 节点扩展计数器

    # 初始化优先队列
    frontier = []
    heapq.heappush(frontier, (heuristic[start], start, [start], 0))  # (启发值, 当前城市, 路径, 当前总代价)

    visited = set()

    while frontier:
        _, current_city, path, current_cost = heapq.heappop(frontier)
        nodes_expanded += 1  # 扩展节点计数

        # 如果找到了目标城市，发送最终路径和总代价
        if current_city == goal:
            end_time = time.perf_counter()  # 使用 perf_counter 记录结束时间
            update_queue.put({
                'type': 'found',
                'path': path,
                'cost': current_cost,
                'nodes_expanded': nodes_expanded,
                'runtime': end_time - start_time
            })
            return

        # 标记当前城市为已访问
        if current_city not in visited:
            visited.add(current_city)

            # 扩展当前城市的相邻城市
            for neighbor, distance in graph.get(current_city, []):
                if neighbor not in visited:
                    new_cost = current_cost + distance
                    heapq.heappush(frontier, (heuristic.get(neighbor, float('inf')), neighbor, path + [neighbor], new_cost))
                    # 发送当前扩展的节点
                    update_queue.put({
                        'type': 'expand',
                        'current_city': current_city,
                        'neighbor': neighbor,
                        'cost': new_cost,
                        'path': path + [neighbor]
                    })

    # 如果没有找到路径，发送未找到的信息
    update_queue.put({
        'type': 'not_found',
        'path': None,
        'cost': float('inf'),
        'nodes_expanded': nodes_expanded,
        'runtime': time.perf_counter() - start_time
    })

# A*搜索算法
def a_star_search(graph, heuristic, start, goal, update_queue):
    start_time = time.perf_counter()  # 使用 perf_counter 记录开始时间
    nodes_expanded = 0  # 节点扩展计数器

    # 初始化优先队列
    frontier = []
    heapq.heappush(frontier, (heuristic[start], 0, start, [start]))  # (f(n), g(n), 当前城市, 路径)

    visited = {}
    # visited 存储每个城市已知的最低g(n)

    while frontier:
        f_cost, g_cost, current_city, path = heapq.heappop(frontier)
        nodes_expanded += 1  # 扩展节点计数

        # 如果找到了目标城市，发送最终路径和代价
        if current_city == goal:
            end_time = time.perf_counter()  # 使用 perf_counter 记录结束时间
            update_queue.put({
                'type': 'found',
                'path': path,
                'cost': g_cost,
                'nodes_expanded': nodes_expanded,
                'runtime': end_time - start_time
            })
            return

        # 如果当前城市已经有更低的g(n)，跳过
        if current_city in visited and g_cost >= visited[current_city]:
            continue

        # 记录当前城市的g(n)
        visited[current_city] = g_cost

        # 扩展当前城市的相邻城市
        for neighbor, distance in graph.get(current_city, []):
            new_g_cost = g_cost + distance
            if neighbor not in visited or new_g_cost < visited.get(neighbor, float('inf')):
                f_new = new_g_cost + heuristic.get(neighbor, 0)
                heapq.heappush(frontier, (f_new, new_g_cost, neighbor, path + [neighbor]))
                # 发送当前扩展的节点
                update_queue.put({
                    'type': 'expand',
                    'current_city': current_city,
                    'neighbor': neighbor,
                    'cost': new_g_cost,
                    'path': path + [neighbor]
                })

    # 如果没有找到路径，发送未找到的信息
    update_queue.put({
        'type': 'not_found',
        'path': None,
        'cost': float('inf'),
        'nodes_expanded': nodes_expanded,
        'runtime': time.perf_counter() - start_time
    })

# 动态可视化更新函数
def update_visualization():
    while not update_queue.empty():
        update = update_queue.get()
        if update['type'] == 'expand':
            current_city = update['current_city']
            neighbor = update['neighbor']
            path = update['path']
            cost = update['cost']
            # 绘制已扩展的边（绿色）
            edge = tuple(sorted((current_city, neighbor)))
            if edge not in path_lines:
                nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color='green', width=2, ax=ax)
                path_lines.append(edge)
                fig.canvas.draw()
            # 绘制当前路径（蓝色）
            if len(path) > 1:
                current_path_edges = list(zip(path, path[1:]))
                # 仅绘制新的路径部分
                for e in current_path_edges:
                    if e not in path_lines:
                        nx.draw_networkx_edges(G, pos, edgelist=[e], edge_color='blue', width=2, ax=ax)
                        path_lines.append(e)
                nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange', node_size=800, ax=ax)
                fig.canvas.draw()
        elif update['type'] == 'found':
            path = update['path']
            cost = update['cost']
            nodes_expanded = update['nodes_expanded']
            runtime = update['runtime']
            # 高亮显示最终路径（红色）
            if path:
                path_edges = list(zip(path, path[1:]))
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3, ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='yellow', node_size=1000, ax=ax)
                fig.canvas.draw()
            # 更新标签
            cost_label.config(text=f"{cost}" if cost != float('inf') else "N/A")
            nodes_expanded_label.config(text=str(nodes_expanded))
            runtime_label.config(text=f"{runtime:.4f}")
            # 提示搜索完成
            messagebox.showinfo("结果", f"搜索完成！\n耗散值: {cost if cost != float('inf') else 'N/A'}\n生成节点数: {nodes_expanded}\n运行时间: {runtime:.4f} 秒")
        elif update['type'] == 'not_found':
            path = update['path']
            cost = update['cost']
            nodes_expanded = update['nodes_expanded']
            runtime = update['runtime']
            messagebox.showinfo("结果", "未找到路径。")
            # 更新标签
            cost_label.config(text="N/A")
            nodes_expanded_label.config(text=str(nodes_expanded))
            runtime_label.config(text=f"{runtime:.4f}")
    # 继续定期调用
    root.after(100, update_visualization)

# 搜索按钮的线程启动函数
def run_search():
    global search_thread  # 声明为全局变量
    if search_thread and search_thread.is_alive():
        messagebox.showwarning("警告", "当前有搜索正在进行，请稍候。")
        return

    algorithm = algorithm_combo.get()
    start_city = start_city_combo.get()
    goal_city = goal_city_combo.get()

    if start_city == goal_city:
        messagebox.showwarning("输入错误", "起始城市和目标城市不能相同。")
        return

    if not start_city or not goal_city:
        messagebox.showwarning("输入错误", "请确保选择了起始城市和目标城市。")
        return

    # 禁用搜索按钮
    search_button.config(state='disabled')

    # 清除之前的绘图
    ax.clear()
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    path_lines.clear()
    fig.canvas.draw()

    # 重置标签
    cost_label.config(text="N/A")
    nodes_expanded_label.config(text="N/A")
    runtime_label.config(text="N/A")

    # 启动搜索线程
    search_thread = threading.Thread(target=execute_search, args=(algorithm, start_city, goal_city, update_queue))
    search_thread.start()

    # 启动一个监控线程来重新启用搜索按钮
    def monitor_thread():
        search_thread.join()
        search_button.config(state='normal')

    monitor = threading.Thread(target=monitor_thread)
    monitor.start()

def execute_search(algorithm, start_city, goal_city, update_queue):
    try:
        if algorithm == "UCS":
            uniform_cost_search(graph, start_city, goal_city, update_queue)
        elif algorithm == "Greedy":
            greedy_best_first_search(graph, heuristic, start_city, goal_city, update_queue)
        elif algorithm == "A*":
            a_star_search(graph, heuristic, start_city, goal_city, update_queue)
        else:
            messagebox.showerror("错误", "未知的算法选择。")
    except Exception as e:
        messagebox.showerror("错误", f"搜索过程中发生错误：{e}")

# 图形界面使用 Tkinter
root = tk.Tk()
root.title("罗马尼亚度假问题路径搜索")

# 设置界面布局
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# 算法选择
ttk.Label(frame, text="选择算法:").grid(column=0, row=0, sticky=tk.W, pady=5)
algorithm_combo = ttk.Combobox(frame, values=["UCS", "Greedy", "A*"], state="readonly")
algorithm_combo.grid(column=1, row=0, pady=5, padx=5)
algorithm_combo.current(0)

# 出发城市和目标城市
ttk.Label(frame, text="起始城市:").grid(column=0, row=1, sticky=tk.W, pady=5)
start_city_combo = ttk.Combobox(frame, values=list(graph.keys()), state="readonly")
start_city_combo.grid(column=1, row=1, pady=5, padx=5)
start_city_combo.current(0)

ttk.Label(frame, text="目标城市:").grid(column=0, row=2, sticky=tk.W, pady=5)
goal_city_combo = ttk.Combobox(frame, values=list(graph.keys()), state="readonly")
goal_city_combo.grid(column=1, row=2, pady=5, padx=5)
# 默认选择不同的起始和目标城市
if len(graph.keys()) >= 2:
    goal_city_combo.current(1)
else:
    goal_city_combo.current(0)

# 结果显示标签
ttk.Label(frame, text="耗散值:").grid(column=0, row=4, sticky=tk.W, pady=5)
cost_label = ttk.Label(frame, text="N/A")
cost_label.grid(column=1, row=4, sticky=tk.W, pady=5)

ttk.Label(frame, text="生成节点数:").grid(column=0, row=5, sticky=tk.W, pady=5)
nodes_expanded_label = ttk.Label(frame, text="N/A")
nodes_expanded_label.grid(column=1, row=5, sticky=tk.W, pady=5)

ttk.Label(frame, text="运行时间 (秒):").grid(column=0, row=6, sticky=tk.W, pady=5)
runtime_label = ttk.Label(frame, text="N/A")
runtime_label.grid(column=1, row=6, sticky=tk.W, pady=5)

# 搜索按钮
search_button = ttk.Button(frame, text="搜索", command=run_search)
search_button.grid(column=0, row=3, columnspan=2, pady=10)

# 调整列权重以适应窗口大小
for child in frame.winfo_children():
    child.grid_configure(padx=5, pady=5)

# 初始化路径记录
path_lines = []

# 搜索线程控制变量
search_thread = None

# 启动动态可视化更新
root.after(100, update_visualization)

# 启动主循环
root.mainloop()
