from abc import ABC, abstractmethod
import uuid
from typing import Optional

from ..aggregates.forecast import Forecast

class IForecastRepository(ABC):
    """
    预测(Forecast)聚合的仓储接口。
    定义了数据持久化的标准合同，领域层依赖此接口，而不是具体实现。
    """
    
    @abstractmethod
    def save(self, forecast: Forecast) -> None:
        """
        保存一个Forecast聚合实例。
        如果实例已存在，则更新它。
        """
        raise NotImplementedError

    @abstractmethod
    def find_by_id(self, forecast_id: uuid.UUID) -> Optional[Forecast]:
        """
        通过ID查找并返回一个Forecast聚合实例。
        如果找不到，则返回None。
        """
        raise NotImplementedError 