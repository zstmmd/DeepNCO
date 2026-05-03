class OFSConfig:
    """OFS system configuration."""

    WAREHOUSE_BLOCK_LENGTH = 18
    WAREHOUSE_BLOCK_WIDTH = 6
    WAREHOUSE_BLOCK_HEIGHT = 10

    ROBOT_SPEED = 1.0
    PACKING_TIME = 2.0
    LIFTING_TIME = 1.0
    ROBOT_CAPACITY = 8
    REMOVE_TOP_TOTE_TIME = 2.0
    PLACE_TOTE_TIME = 2.0

    PICKING_TIME = 3.0
    MOVE_EXTRA_TOTE_TIME = 1.0

    DEFAULT_PICKING_STATION_BUFFER = 10
    MAX_LAYER = 10

    KIT_DELIVERY_WINDOW = 300

    ORDER_EST_MIN_SEC = 0
    ORDER_EST_MAX_SEC = 100
    ORDER_KITTING_SPAN_PER_UNIQUE_SKU_SEC = 8
    ORDER_LST_BASE_SEC = 60
    ORDER_LST_PER_QTY_SEC = 2
    ORDER_LST_BUFFER_MIN_SEC = 10
    ORDER_LST_BUFFER_MAX_SEC = 30
    BOM_ARRIVAL_WINDOW_PER_UNIQUE_SKU_SEC = 15.0

    RANDOM_SEED = 42

    @staticmethod
    def effective_bom_arrival_window_sec(base_window_sec: float, unique_sku_count: int) -> float:
        base_window = max(0.0, float(base_window_sec or 0.0))
        unique_skus = max(0, int(unique_sku_count or 0))
        dynamic_window = float(unique_skus) * float(OFSConfig.BOM_ARRIVAL_WINDOW_PER_UNIQUE_SKU_SEC)
        return float(max(base_window, dynamic_window))
