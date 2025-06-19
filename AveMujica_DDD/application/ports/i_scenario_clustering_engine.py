from abc import ABC, abstractmethod
import pandas as pd

class IScenarioClusteringEngine(ABC):
    """
    应用层端口 (Port): 场景聚类引擎接口。

    封装了用于识别天气场景的聚类算法的具体实现。
    """

    @abstractmethod
    def perform_clustering(self, data: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
        """
        对给定的数据执行聚类分析。

        :param data: 包含特征列的数据。
        :param n_clusters: 要识别的簇（场景）的数量。
        :return: 一个带有'cluster'列的DataFrame，标记了每行数据所属的簇。
        """
        raise NotImplementedError 