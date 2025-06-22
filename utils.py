# utils.py

# 定义棋子的类型ID
class PieceType:
    QUEEN_BEE = 0
    BEETLE = 1
    SPIDER = 2
    ANT = 3
    GRASSHOPPER = 4
    LADYBUG = 5
    MOSQUITO = 6
    PILLBUG = 7

PIECE_TYPE_LIST = [
    PieceType.QUEEN_BEE,
    PieceType.BEETLE,
    PieceType.SPIDER,
    PieceType.ANT,
    PieceType.GRASSHOPPER,
    PieceType.LADYBUG,
    PieceType.MOSQUITO,
    PieceType.PILLBUG
]
PIECE_TYPE_NAME_LIST = [
    "QUEEN_BEE",
    "BEETLE",
    "SPIDER",
    "ANT",
    "GRASSHOPPER",
    "LADYBUG",
    "MOSQUITO",
    "PILLBUG"
]

# 定义棋盘大小
BOARD_SIZE = 10

# 定义六边形方向的偏移量
# 这些偏移量用于计算相邻的六边形
# 考虑到六边形网格的两种常见布局（平顶和尖顶），这里选择一种。
# C++代码中似乎是按照二维数组处理，所以这里也按此处理，但需要注意相邻判断。
# C++代码中的 directions 数组: {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, -1}, {-1, 1}}
# 这表示的是在直角坐标系下的相邻点，对应六边形网格的相邻逻辑需要进一步确认。
# 假设C++的实现是基于一种近似的六边形网格，或者简化为直角坐标系下的相邻。
# 实际的六边形相邻需要考虑奇偶行/列的偏移。
# 鉴于C++代码中直接使用了固定的6个方向，我们暂时也沿用这种简化方式。
DIRECTIONS = [
    (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)
]

PIECE_TYPE_ID_MAP = {
    PieceType.QUEEN_BEE: 0,
    PieceType.BEETLE: 1,
    PieceType.SPIDER: 2,
    PieceType.ANT: 3,
    PieceType.GRASSHOPPER: 4,
    PieceType.LADYBUG: 5,
    PieceType.MOSQUITO: 6,
    PieceType.PILLBUG: 7
}


