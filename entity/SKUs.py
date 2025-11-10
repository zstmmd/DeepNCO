class SKUs:
    """SKUs实体类"""

    def __init__(self, sku_id: int = 0):
        self.id = sku_id  # SKUs的id
        self.storeNodeList=[]  # 存储该SKU的所有存储节点列表


    def __str__(self):
        return f"SKUs(id={self.id})"