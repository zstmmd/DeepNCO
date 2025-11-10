from point import Point
from config.ofs_config import OFSConfig


class Robot:
    """机器人类"""

    def __init__(self, robot_id: int = 0, start_point: Point = None):
        """
        :param robot_id: 机器人的id
        :param start_point: 机器人的初始节点
        """
        self.id = robot_id
        self.start_point = start_point
        self.capacity = OFSConfig.ROBOT_CAPACITY  # 机器人的载重
        self.velocity = OFSConfig.ROBOT_SPEED  # 机器人的速度
        self.packing_time = OFSConfig.PACKING_TIME  # 机器人上下存储单元的时间
        self.lifting_time = OFSConfig.LIFTING_TIME  # 机器人伸缩货叉的时间

    def __str__(self):
        return (f"Robot(id={self.id}, start_point={self.start_point}, "
                f"capacity={self.capacity}, velocity={self.velocity}, "
                f"packing_time={self.packing_time}, lifting_time={self.lifting_time})")