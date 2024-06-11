"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if board.count(EMPTY) == 9 or board.count(X) < board.count(O):
        play = X
    else:
        play = O
    return play


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    all_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                all_actions.add((i, j))
    return all_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    new_board = board.deepcopy()
    if new_board[action[0]][action[1]] == EMPTY:
        new_board[action[0]][action[1]] = player(board)
        return new_board
    else:
        raise Exception("Invalid Action")


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # winning by row
    for i in range(3):
        flag = True
        for j in range(1, 3):
            if board[i][j] != board[i][j - 1]:
                flag = False
                break
        if flag:
            return board[i][0]
    # winning by column
    for j in range(3):
        flag = True
        for i in range(1, 3):
            if board[i][j] != board[i - 1][j]:
                flag = False
                break
        if flag:
            return board[0][j]
    # winning by diagonal
    if (board[0][0] == board[1][1] == board[2][2] or
            board[2][0] == board[1][1] == board[0][2]):
        return board[1][1]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    optimal_action = ()
    num_x = -2
    num_o = 2
    play = player(board)
    if terminal(board):
        return None
    for action in actions(board):
        if play == X:
            r = min_value(result(board, action))
            if r > num_x:
                optimal_action = action
                num_x = r
        else:
            r = max_value(result(board, action))
            if r < num_o:
                optimal_action = action
                num_o = r
    return optimal_action


def max_value(board):
    v = -2
    if terminal(board):
        return utility(board)
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


def min_value(board):
    v = 2
    if terminal(board):
        return utility(board)
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v
