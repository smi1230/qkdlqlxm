# /ml/model_trainer.py

import os
import glob
import logging
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from typing import Dict, Any, Tuple

from tensorflow.keras.utils import Sequence

try:
    from configs import settings
    from ml.predictor import DeepLearningPredictor
except ImportError:
    print("[CRITICAL ERROR] Failed to import modules in model_trainer.py.")
    raise SystemExit("Module loading failed.")

logger = logging.getLogger(__name__)

class DataGenerator(Sequence):
    """
    미리 처리된 NumPy 배열 데이터를 기반으로 훈련 배치를 생성하는 제너레이터.
    각 배치에 대한 샘플 가중치를 직접 생성하여 반환합니다.
    """
    def __init__(self, X_data: np.ndarray, y_data: Dict[str, np.ndarray], class_weights: Dict[int, float], sequence_length: int, batch_size: int, shuffle: bool = True):
        self.X = X_data
        self.y = y_data
        self.class_weights = class_weights
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X) - self.sequence_length + 1)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.floor(len(self.indices) / self.batch_size))

    # def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    #     batch_start_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
    #     X_batch_seq = self.X[batch_start_indices[:, None] + np.arange(self.sequence_length)]
        
    #     target_indices = batch_start_indices + self.sequence_length - 1
    #     y_batch = {key: self.y[key][target_indices] for key in self.y.keys()}
        
    #     # 'price_direction' 타겟을 기준으로 각 샘플에 맞는 가중치를 매핑합니다.
    #     price_direction_labels = y_batch['price_direction']
    #     sample_weights = np.array([self.class_weights.get(label, 1.0) for label in price_direction_labels])
        
    #     return X_batch_seq, y_batch, sample_weights
    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        주어진 배치 내에서 Numpy만을 사용하여 언더샘플링을 수행하고 클래스 균형을 맞춘 최종 배치를 반환합니다.
        """
        # 1. 원본과 동일하게 해당 인덱스의 배치 데이터를 가져옵니다.
        batch_start_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # 만약 이 단계에서 인덱스가 부족하면, 이는 제너레이터의 상태나 호출에 문제가 있다는 의미이므로 오류를 발생시킵니다.
        if len(batch_start_indices) == 0:
            raise ValueError(
                f"배치 index {index}에 해당하는 데이터를 찾을 수 없습니다. "
                f"요청된 index가 생성 가능한 배치의 총 수({self.__len__()})를 초과했을 수 있습니다."
            )
    
        # 2. 이 배치의 'price_direction' 라벨을 가져옵니다.
        target_indices = batch_start_indices + self.sequence_length - 1
        price_direction_labels = self.y['price_direction'][target_indices]
    
        # 3. 배치 내의 고유 라벨과 각 라벨의 개수를 찾습니다.
        unique_labels, counts = np.unique(price_direction_labels, return_counts=True)
        
        # 배치에 유효한 라벨이 없는 경우, 이는 데이터 또는 인덱싱 문제일 수 있으므로 오류를 발생시킵니다.
        if len(unique_labels) == 0:
            raise ValueError(
                f"배치 index {index}에서 유효한 라벨을 찾을 수 없습니다. "
                f"batch_start_indices: {batch_start_indices}. "
                "데이터셋 또는 self.indices 배열의 구성을 확인하십시오."
            )
            
        # 4. 배치 내에 있는 라벨(클래스)별 개수들 중 가장 작은 값을 찾습니다. (예: [10, 5, 2] -> 2)
        min_count_in_batch = np.min(counts)
    
        # 5. 각 라벨 그룹에서 'min_count_in_batch'만큼 무작위로 샘플링합니다.
        balanced_indices_list = []
        for label in unique_labels:
            # 현재 라벨에 해당하는 데이터가 배치 내 어디에 있는지 그 위치(인덱스)를 찾습니다.
            indices_for_label = np.where(price_direction_labels == label)[0]
            
            # 해당 위치들 중에서 min_count_in_batch 개수만큼 무작위로 선택합니다.
            randomly_chosen_indices = np.random.choice(indices_for_label, min_count_in_batch, replace=False)
            
            # 선택된 위치에 해당하는 원본 시작 인덱스(batch_start_indices)를 리스트에 추가합니다.
            balanced_indices_list.append(batch_start_indices[randomly_chosen_indices])
    
        # 6. 최종적으로 사용할 균형잡힌 배치의 시작 인덱스를 하나로 합치고 섞어줍니다.
        final_batch_start_indices = np.concatenate(balanced_indices_list)
        np.random.shuffle(final_batch_start_indices)
        
        # 7. 이 최종 인덱스들을 사용하여 X와 y 데이터를 구성합니다.
        final_target_indices = final_batch_start_indices + self.sequence_length - 1
        
        X_batch_seq = self.X[final_batch_start_indices[:, None] + np.arange(self.sequence_length)]
        y_batch = {key: self.y[key][final_target_indices] for key in self.y.keys()}
        
        # 8. 데이터의 균형을 맞췄으므로 가중치는 모두 1로 설정하여 반환합니다.
        sample_weights = np.ones(len(final_batch_start_indices))
        
        return X_batch_seq, y_batch, sample_weights


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class ModelTrainer:
    """
    [ v5.7 - 로직 단순화 ]
    - 훈련 파이프라인에서 옵티마이저 재컴파일 로직을 제거.
    - 해당 기능은 train.py에서 중앙 관리합니다.
    """
    def __init__(self, predictor: DeepLearningPredictor, data_dir: str, model_dir: str):
        self.predictor = predictor
        self.preprocessed_data_dir = os.path.join(data_dir, "preprocessed")
        self.model_dir = model_dir
        self.sequence_length = settings.HYBRID_MODEL_CONFIG['sequence_length']
        # 파일에서 읽어올 타겟 이름 (접두사 포함)
        self.target_names_from_file = [f'target_{key}' for key in settings.MULTITASK_LOSS_WEIGHTS]

    def _load_preprocessed_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        logger.info(f"Loading preprocessed HDF5 data from: {self.preprocessed_data_dir}")
        hdf5_files = glob.glob(os.path.join(self.preprocessed_data_dir, '*.hdf5'))
        if not hdf5_files:
            raise FileNotFoundError(f"No preprocessed HDF5 files found in {self.preprocessed_data_dir}. "
                                    f"Please run 'preprocess_and_save_features.py' first.")

        all_X, all_y_dict_from_file = [], {name: [] for name in self.target_names_from_file}

        for file_path in hdf5_files:
            try:
                with h5py.File(file_path, 'r') as hf:
                    all_X.append(hf['X'][:])
                    for target_name in self.target_names_from_file:
                        all_y_dict_from_file[target_name].append(hf[target_name][:])
            except Exception as e:
                logger.error(f"Failed to read HDF5 file {file_path}: {e}")
                continue
        
        if not all_X:
            raise ValueError("No data could be loaded from HDF5 files.")

        X_combined = np.concatenate(all_X, axis=0)
        
        y_combined = {
            key.replace('target_', ''): np.concatenate(val, axis=0) 
            for key, val in all_y_dict_from_file.items()
        }
        
        logger.info(f"Data loading complete. Total samples: {len(X_combined)}")
        return X_combined, y_combined
        
    def run_training_pipeline(self, hyperparameters: Dict[str, Any]):
        # [핵심 수정 v5.7] 옵티마이저 교체 로직을 train.py로 이전함에 따라 이 블록은 제거합니다.
        # 이중으로 recompile을 호출하는 문제를 방지하고 코드 흐름을 명확하게 합니다.
        
        X_2d_np, y_dict_np = self._load_preprocessed_data()
        
        X_train_np, X_val_np, _, _ = train_test_split(
            X_2d_np, y_dict_np['price_direction'], test_size=0.2, shuffle=False
        )

        train_size = len(X_train_np)
        y_train_dict = {key: val[:train_size] for key, val in y_dict_np.items()}
        y_val_dict = {key: val[train_size:] for key, val in y_dict_np.items()}
        
        logger.info(f"Data split complete. Train size: {len(X_train_np)}, Validation size: {len(X_val_np)}")

        price_direction_train = y_train_dict['price_direction']
        unique_classes = np.unique(price_direction_train)
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=unique_classes, y=price_direction_train
        )
        class_weights_dict = dict(zip(unique_classes, class_weights))
        logger.info(f"Calculated class weights for generator: {class_weights_dict}")
        
        train_generator = DataGenerator(
            X_train_np, y_train_dict, class_weights_dict, 
            self.sequence_length, hyperparameters['batch_size'], shuffle=True
        )
        val_generator = DataGenerator(
            X_val_np, y_val_dict, class_weights_dict, 
            self.sequence_length, hyperparameters['batch_size'], shuffle=False
        )
        
        self.predictor.train(train_generator, val_generator, hyperparameters)
        logger.info("ModelTrainer has completed the training pipeline.")
