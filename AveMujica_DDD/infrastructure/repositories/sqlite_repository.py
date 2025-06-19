import sqlite3
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional

from AveMujica_DDD.domain.aggregates.prediction_model import PredictionModel, ForecastType
from AveMujica_DDD.domain.repositories.i_model_repository import IModelRepository

class SQLiteUnitOfWork:
    """一个上下文管理器，用于处理数据库连接和事务。"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()

class SQLiteModelRepository(IModelRepository):
    def __init__(self, db_path: str = "data/mujica.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        with SQLiteUnitOfWork(self.db_path) as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    forecast_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    feature_columns TEXT NOT NULL,
                    description TEXT
                )
            """)

    def save(self, model: PredictionModel) -> None:
        with SQLiteUnitOfWork(self.db_path) as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO models (model_id, name, version, forecast_type, file_path, target_column, feature_columns, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(model.model_id), 
                model.name, 
                model.version, 
                model.forecast_type.value, 
                model.file_path, 
                model.target_column,
                json.dumps(model.feature_columns),  # 将列表序列化为JSON字符串
                model.description
            ))

    def find_by_id(self, model_id: uuid.UUID) -> Optional[PredictionModel]:
        with SQLiteUnitOfWork(self.db_path) as cursor:
            cursor.execute("SELECT * FROM models WHERE model_id = ?", (str(model_id),))
            row = cursor.fetchone()
            return self._row_to_aggregate(row) if row else None
            
    def find_by_name_and_version(self, name: str, version: str) -> Optional[PredictionModel]:
        """通过名称和版本查找模型。"""
        with SQLiteUnitOfWork(self.db_path) as cursor:
            cursor.execute("SELECT * FROM models WHERE name = ? AND version = ?", (name, version))
            row = cursor.fetchone()
            return self._row_to_aggregate(row) if row else None

    def _row_to_aggregate(self, row: sqlite3.Row) -> PredictionModel:
        return PredictionModel(
            name=row["name"],
            version=row["version"],
            forecast_type=ForecastType(row["forecast_type"]),
            file_path=row["file_path"],
            target_column=row["target_column"],
            feature_columns=json.loads(row["feature_columns"]),  # 从JSON字符串反序列化为列表
            model_id=uuid.UUID(row["model_id"]),
            description=row["description"]
        )

    # ... rest of the existing methods ... 