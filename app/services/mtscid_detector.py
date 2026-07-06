from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from app.schemas.detection import DetectionRequest
from app.services.result_summarizer import create_detection_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MTSCID_ROOT = PROJECT_ROOT / 'MtsCID'

if str(MTSCID_ROOT) not in sys.path:
    sys.path.insert(0, str(MTSCID_ROOT))

from solver import Solver  # noqa: E402


class MtsCIDDetector:
    def __init__(self, mtscid_root: Path = MTSCID_ROOT):
        self.mtscid_root = Path(mtscid_root)

    def available_datasets(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'PSM',
                'format': 'csv',
                'data_path': str(self.mtscid_root / 'data' / 'PSM'),
                'checkpoint': str(self.mtscid_root / 'checkpoints' / 'PSM_checkpoint.pth'),
                'input_c': 25,
                'description': 'PSM built-in CSV dataset: train.csv, test.csv and test_label.csv.',
            }
        ]

    def detect(self, request: DetectionRequest) -> Dict[str, Any]:
        if request.detector_method != 'MtsCID':
            raise ValueError(f'Unsupported detector_method for current backend: {request.detector_method}')

        config = self._build_config(request)
        solver = Solver(config)
        solver.model_init(config)
        detection = solver.detect(config, include_details=True)
        variable_names = self._load_variable_names(Path(config['data_path']))
        public_config = self._public_config(config)
        return create_detection_record(
            detection,
            config=public_config,
            variable_names=variable_names,
            max_chart_points=request.max_chart_points,
        )

    def _build_config(self, request: DetectionRequest) -> Dict[str, Any]:
        data_path = Path(request.data_path) if request.data_path else self.mtscid_root / 'data' / request.dataset
        device_name = request.device
        if device_name is None:
            device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        config = {
            'detector_method': request.detector_method,
            'framework': 'MtsCID',
            'test_only': True,
            'dataset': request.dataset,
            'win_size': request.win_size,
            'data_path': str(data_path),
            'input_c': 25,
            'output_c': 25,
            'd_model': 25,
            'temperature': 0.1,
            'encoder_layers': 1,
            'branches_group_embedding': 'False_False',
            'multiscale_kernel_size': [5],
            'multiscale_patch_size': [10, 20],
            'branch1_networks': ['fc_linear', 'intra_fc_transformer', 'multiscale_ts_attention'],
            'branch1_match_dimension': 'first',
            'branch2_networks': ['multiscale_conv1d', 'inter_fc_transformer'],
            'branch2_match_dimension': 'first',
            'decoder_networks': ['linear'],
            'decoder_layers': 1,
            'decoder_group_embedding': 'False',
            'embedding_init': 'normal',
            'memory_guided': 'sinusoid',
            'aggregation': 'normal_mean',
            'num_workers': 4,
            'model_save_path': str(self.mtscid_root / 'checkpoints'),
            'num_epochs': 20,
            'batch_size': request.batch_size,
            'patience': 10,
            'peak_lr': 2e-3,
            'end_lr': 5e-5,
            'weight_decay': 5e-5,
            'warmup_epoch': 0,
            'device': torch.device(device_name if torch.cuda.is_available() or device_name == 'cpu' else 'cpu'),
            'alpha': 1.0,
            'threshold_setting': request.threshold_setting,
            'anomaly_ratio': request.anomaly_ratio,
            'run_times': 1,
            'plot_data': 'False',
            'anomaly_only': 'False',
        }
        return config

    def _load_variable_names(self, data_path: Path) -> Optional[List[str]]:
        train_csv = data_path / 'train.csv'
        if not train_csv.exists():
            return None

        with train_csv.open('r', newline='', encoding='utf-8') as f:
            header = next(csv.reader(f), [])
        return header[1:] if len(header) > 1 else None

    def _public_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        exposed = {
            'detector_method',
            'framework',
            'dataset',
            'win_size',
            'data_path',
            'input_c',
            'output_c',
            'd_model',
            'batch_size',
            'threshold_setting',
            'anomaly_ratio',
            'model_save_path',
        }
        public = {key: config[key] for key in exposed if key in config}
        public['device'] = str(config.get('device'))
        return public
