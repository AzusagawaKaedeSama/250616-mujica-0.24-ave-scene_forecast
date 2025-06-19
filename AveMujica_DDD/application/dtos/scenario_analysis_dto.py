from dataclasses import dataclass, field
import uuid
from datetime import datetime
from typing import List, Dict, Any

@dataclass(frozen=True)
class ScenarioSummaryDTO:
    """单个场景的摘要信息。"""
    cluster_id: int
    scenario_type_name: str
    description: str
    uncertainty_multiplier: float
    member_count: int
    statistics: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ScenarioAnalysisReportDTO:
    """场景分析报告的DTO。"""
    report_id: uuid.UUID
    province: str
    creation_date: datetime
    scenarios: List[ScenarioSummaryDTO] 