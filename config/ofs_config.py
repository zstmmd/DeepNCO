class OFSConfig:
    """OFS系统配置类"""
    # 基础参数
    # 仓库块的长
    WAREHOUSE_BLOCK_LENGTH = 18
    # 仓库块的宽度
    WAREHOUSE_BLOCK_WIDTH = 6
    # 仓库块的高度
    WAREHOUSE_BLOCK_HEIGHT = 10

    # 机器人配置
    ROBOT_CAPACITY = 1  # 机器人载重
    ROBOT_SPEED = 1.0  # 机器人速度
    PACKING_TIME = 5.0  # 上下存储单元时间
    LIFTING_TIME = 2.0  # 伸缩货叉时间
    ROBOT_CAPACITY = 8  # 机器人容量tote(料箱)数
    REMOVE_TOP_TOTE_TIME = 2.0  # 搬运顶层tote一次的时间
    PLACE_TOTE_TIME = 4.0  # 放置tote一次的时间
    # 工作站配置
    PICKING_TIME = 3.0  # 单个SKU拣选时间

    DEFAULT_PICKING_STATION_BUFFER = 10  # 默认拣选站缓存上限10个tote
    # 存储区最高层高
    MAX_LAYER = 10

    #齐套出库时间段
    KIT_DELIVERY_WINDOW = 300   # 齐套出库时间窗，单位秒
    RANDOM_SEED = 42  # 随机数种子，用于可复现的实验