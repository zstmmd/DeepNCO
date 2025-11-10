class OFSConfig:
    """OFS系统配置类"""

    # 机器人配置
    ROBOT_CAPACITY = 1  # 机器人载重
    ROBOT_SPEED = 1.0  # 机器人速度
    PACKING_TIME = 5.0  # 上下存储单元时间
    LIFTING_TIME = 2.0  # 伸缩货叉时间

    # 拣选站配置
    PICKING_TIME = 3.0  # 单个SKU拣选时间
    DEFAULT_PICKING_STATION_BUFFER = 10  # 默认拣选站缓存上限