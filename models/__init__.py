# from .keras_models import KerasGRU, KerasLSTM
# from .torch_models import TorchConvTransformer, PeakAwareConvTransformer # This line might also be unnecessary if direct imports are used elsewhere

# It's generally better to let modules that need specific models import them directly, e.g.:
# from models.torch_models import TorchConvTransformer
# from models.keras_models import KerasGRU

# If you want to expose certain models at the package level for convenience, you can do so explicitly.
# For now, keeping it clean to avoid unintended imports.

from data.dataset_builder import DatasetBuilder
from data.data_loader import DataLoader

__all__ = ['KerasGRU', 'KerasLSTM', 'TorchForecaster', 'DatasetBuilder']

# 配置默认参数
DEFAULT_SEQ_LENGTH = 96
DEFAULT_PRED_LENGTH = 4
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3

# 全局数据加载器
def load_data():
    return DataLoader('附件1-数据.xlsx')

# 全局数据预处理器
def prepare_dataset(seq_length=DEFAULT_SEQ_LENGTH, pred_length=DEFAULT_PRED_LENGTH):
    data_loader = load_data()
    dataset_builder = DatasetBuilder(data_loader, seq_length, pred_length)
    return dataset_builder.build()