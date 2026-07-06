from typing import Literal, Optional

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    detector_method: str = 'MtsCID'
    dataset: Literal['PSM'] = 'PSM'
    data_path: Optional[str] = None
    threshold_setting: Literal['optimal', 'preset'] = 'optimal'
    anomaly_ratio: float = Field(default=1.0, gt=0.0, le=100.0)
    batch_size: int = Field(default=64, gt=0)
    win_size: int = Field(default=100, gt=0)
    max_chart_points: int = Field(default=100000, ge=100, le=200000)
    device: Optional[str] = None
