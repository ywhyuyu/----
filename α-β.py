import pygame
import sys
import random
import time

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# 初始化Pygame
pygame.init()

# 设置窗口大小
size = width, height = 300, 300
screen = pygame.display.set_mode(size)
pygame.display.set_caption('井字棋')

# 设置棋盘参数
BOARD_SIZE = 3
CELL_SIZE = width // BOARD_SIZE
LINE_WIDTH = 2

# 定义玩家
AI_PLAYER = 'X'
HUMAN_PLAYER = 'O'

class TicTacToe:
    def __init__(self):
        self.board = ['' for _ in range(9)]  # 初始化空棋盘
        self.current_winner = None

    def draw_board(self):
        # 绘制背景
        screen.fill(WHITE)
        # 绘制棋盘线条
        for i in range(1, BOARD_SIZE):
            # 垂直线
            pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, height), LINE_WIDTH)
            # 水平线
            pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (width, i * CELL_SIZE), LINE_WIDTH)
        # 绘制棋子
        for idx, mark in enumerate(self.board):
            x = (idx % BOARD_SIZE) * CELL_SIZE + CELL_SIZE // 2
            y = (idx // BOARD_SIZE) * CELL_SIZE + CELL_SIZE // 2
            if mark == 'O':
                pygame.draw.circle(screen, RED, (x, y), CELL_SIZE // 3, LINE_WIDTH)
            elif mark == 'X':
                start_pos = (x - CELL_SIZE // 3, y - CELL_SIZE // 3)
                end_pos = (x + CELL_SIZE // 3, y + CELL_SIZE // 3)
                pygame.draw.line(screen, BLUE, start_pos, end_pos, LINE_WIDTH)
                start_pos = (x - CELL_SIZE // 3, y + CELL_SIZE // 3)
                end_pos = (x + CELL_SIZE // 3, y - CELL_SIZE // 3)
                pygame.draw.line(screen, BLUE, start_pos, end_pos, LINE_WIDTH)
        pygame.display.update()

    def make_move(self, square, player):
        if self.board[square] == '':
            self.board[square] = player
            if self.winner(square, player):
                self.current_winner = player
            return True
        return False

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == '']

    def empty_squares(self):
        return '' in self.board

    def num_empty_squares(self):
        return self.board.count('')

    def winner(self, square, player):
        # 行
        row_ind = square // BOARD_SIZE
        row = self.board[row_ind*BOARD_SIZE : (row_ind +1)*BOARD_SIZE]
        if all([s == player for s in row]):
            return True
        # 列
        col_ind = square % BOARD_SIZE
        column = [self.board[col_ind+i*BOARD_SIZE] for i in range(BOARD_SIZE)]
        if all([s == player for s in column]):
            return True
        # 对角线
        if square % (BOARD_SIZE + 1) == 0:
            diagonal1 = [self.board[i] for i in range(0, BOARD_SIZE*BOARD_SIZE, BOARD_SIZE+1)]
            if all([s == player for s in diagonal1]):
                return True
        if square % (BOARD_SIZE -1) == 0 and square != 0 and square != BOARD_SIZE*BOARD_SIZE -1:
            diagonal2 = [self.board[i] for i in range(BOARD_SIZE-1, BOARD_SIZE*BOARD_SIZE-1, BOARD_SIZE-1)]
            if all([s == player for s in diagonal2]):
                return True
        return False

def minimax(state, player, alpha, beta):
    max_player = AI_PLAYER
    other_player = HUMAN_PLAYER if player == AI_PLAYER else AI_PLAYER

    # 检查是否上一步已经是胜利状态
    if state.current_winner == other_player:
        return {'position': None, 'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                    state.num_empty_squares() + 1)}

    elif not state.empty_squares():
        return {'position': None, 'score': 0}

    if player == max_player:
        best = {'position': None, 'score': -float('inf')}
    else:
        best = {'position': None, 'score': float('inf')}

    for possible_move in state.available_moves():
        # 尝试一个可能的移动
        state.make_move(possible_move, player)
        sim_score = minimax(state, other_player, alpha, beta)  # 递归调用
        # 撤销移动
        state.board[possible_move] = ''
        state.current_winner = None
        sim_score['position'] = possible_move

        if player == max_player:
            if sim_score['score'] > best['score']:
                best = sim_score  # 更新最佳分数
            alpha = max(alpha, best['score'])
        else:
            if sim_score['score'] < best['score']:
                best = sim_score  # 更新最佳分数
            beta = min(beta, best['score'])

        # 剪枝
        if beta <= alpha:
            break

    return best

def main():
    game = TicTacToe()
    game.draw_board()

    # 随机选择先手
    current_player = AI_PLAYER if random.randint(0,1) == 0 else HUMAN_PLAYER

    while game.empty_squares():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if current_player == AI_PLAYER:
            # AI移动
            move = minimax(game, AI_PLAYER, -float('inf'), float('inf'))['position']
            game.make_move(move, AI_PLAYER)
            game.draw_board()
            if game.current_winner:
                print(f"玩家 {AI_PLAYER} 获胜！")
                time.sleep(2)
                pygame.quit()
                sys.exit()
            current_player = HUMAN_PLAYER
            time.sleep(0.5)
        else:
            # 另一个AI移动
            move = minimax(game, HUMAN_PLAYER, -float('inf'), float('inf'))['position']
            game.make_move(move, HUMAN_PLAYER)
            game.draw_board()
            if game.current_winner:
                print(f"玩家 {HUMAN_PLAYER} 获胜！")
                time.sleep(2)
                pygame.quit()
                sys.exit()
            current_player = AI_PLAYER
            time.sleep(0.5)

    print("平局！")
    time.sleep(2)
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()