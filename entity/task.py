from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, ClassVar
from datetime import datetime

@dataclass
class Task:

    def __init__(self,
                 robotId: Optional[int] = None,
                 sequenceNo: Optional[int] = None,
                 startPointNo: Optional[str] = None,
                 endPointNo: Optional[str] = None,
                 toteId: Optional[str] = None,
                 taskType: Optional[str] = None,
                 layer: Optional[int] = None):

        self.Id: Optional[str] = None  # UUID主键
        self.robotId = robotId
        self.sequenceNo = sequenceNo
        self.startPointNo = startPointNo
        self.endPointNo = endPointNo
        self.toteId = toteId
        self.taskType = taskType
        self.layer = layer

        self.isCompleted: bool = False

        self.createdAt: Optional[datetime] = None  # 创建时间