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
    [핵심] 대용량의 훈련 데이터를 메모리에 모두 올리지 않고, 훈련 시 필요한 만큼만 배치(batch) 단위로
    생성하여 모델에 공급하는 역할을 합니다. Keras의 Sequence 클래스를 상속받아 구현되었습니다.
    """
    def __init__(self, X_data: np.ndarray, y_data: Dict[str, np.ndarray], class_weights: Dict[int, float], sequence_length: int, batch_size: int, shuffle: bool = True):
        self.X = X_data  # 피쳐 데이터
        self.y = y_data  # 타겟 데이터 (딕셔너리 형태)
        self.class_weights = class_weights # 클래스 불균형 처리를 위한 가중치
        self.sequence_length = sequence_length # 모델 입력 시퀀스 길이
        self.batch_size = batch_size # 배치 크기
        self.shuffle = shuffle # 매 에포크마다 데이터를 섞을지 여부
        # 생성 가능한 시퀀스의 시작 인덱스들을 계산합니다.
        self.indices = np.arange(len(self.X) - self.sequence_length + 1)
        self.on_epoch_end() # 초기화 시 한 번 섞어줍니다.

    def __len__(self) -> int:
        """한 에포크 당 생성할 배치의 총 개수를 반환합니다."""
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        [핵심 로직] 주어진 인덱스(index)에 해당하는 하나의 배치를 생성합니다.
        - 이 함수는 훈련 중 Keras에 의해 자동으로 호출됩니다.
        - 배치 내에서 클래스 불균형을 해소하기 위해 언더샘플링을 수행합니다.
        """
        # 1. 현재 배치에 해당하는 시작 인덱스들을 가져옵니다.
        batch_start_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # 2. 이 배치의 'trade_outcome' 라벨(정답)을 가져옵니다.
        target_indices = batch_start_indices + self.sequence_length - 1
        trade_outcome_labels = self.y['trade_outcome'][target_indices]
    
        # 3. 배치 내의 각 클래스(0~2)가 몇 개씩 있는지 개수를 셉니다.
        unique_labels, counts = np.unique(trade_outcome_labels, return_counts=True)
        
        if len(unique_labels) == 0:
            raise ValueError(f"Batch index {index} has no valid labels.")
            
        # 4. 가장 적은 클래스의 개수를 기준으로 삼습니다 (언더샘플링).
        min_count_in_batch = np.min(counts) if counts.size > 0 else 0
        if min_count_in_batch == 0: # 배치에 샘플이 없는 경우 방지
             # 다음 배치를 시도하거나, 에포크가 끝났으면 빈 배치를 반환할 수 있습니다.
             # 여기서는 간단하게 다음 배치를 재귀적으로 호출합니다.
             return self.__getitem__((index + 1) % len(self))

        # 5. 모든 클래스에서 'min_count_in_batch' 개수만큼 무작위로 샘플을 추출합니다.
        balanced_indices_list = []
        for label in unique_labels:
            indices_for_label = np.where(trade_outcome_labels == label)[0]
            if len(indices_for_label) > 0:
                randomly_chosen_indices = np.random.choice(indices_for_label, min_count_in_batch, replace=False)
                balanced_indices_list.append(batch_start_indices[randomly_chosen_indices])
    
        if not balanced_indices_list:
            # 균형을 맞출 수 없는 경우 (예: 한 클래스만 존재)
            return self.__getitem__((index + 1) % len(self))

        # 6. 최종적으로 균형이 맞춰진 배치의 시작 인덱스들을 하나로 합치고 섞어줍니다.
        final_batch_start_indices = np.concatenate(balanced_indices_list)
        np.random.shuffle(final_batch_start_indices)
        
        # 7. 이 최종 인덱스들을 사용하여 X(피쳐)와 y(타겟) 데이터를 구성합니다.
        final_target_indices = final_batch_start_indices + self.sequence_length - 1
        X_batch_seq = self.X[final_batch_start_indices[:, None] + np.arange(self.sequence_length)]
        
        # [핵심 수정] Keras는 단일 출력 모델의 경우 y_batch를 딕셔너리가 아닌 NumPy 배열로 받기를 기대합니다.
        y_batch = self.y[settings.ACTIVE_OUTPUTS[0]][final_target_indices]
        
        return X_batch_seq, y_batch

    def on_epoch_end(self):
        """한 에포크가 끝날 때마다 호출되며, 데이터를 섞어줍니다."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class ModelTrainer:
    """
    [ v5.7 - 로직 단순화 ]
    - AI 모델 훈련 파이프라인을 총괄하는 클래스입니다.
    - 전처리된 데이터를 로드하고, 훈련/검증 데이터로 분할한 뒤,
      DataGenerator를 사용하여 모델 훈련을 실행합니다.
    """
    def __init__(self, predictor: DeepLearningPredictor, data_dir: str, model_dir: str):
        self.predictor = predictor
        self.preprocessed_data_dir = os.path.join(data_dir, "preprocessed")
        self.model_dir = model_dir
        self.sequence_length = settings.HYBRID_MODEL_CONFIG['sequence_length']
        # HDF5 파일에서 읽어올 타겟 데이터셋의 이름 목록입니다.
        self.target_names_from_file = [f'target_{key}' for key in settings.ACTIVE_OUTPUTS]

    def _load_preprocessed_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """전처리된 모든 HDF5 파일을 찾아 하나로 합쳐서 로드합니다."""
        logger.info(f"Loading preprocessed HDF5 data from: {self.preprocessed_data_dir}")
        hdf5_files = glob.glob(os.path.join(self.preprocessed_data_dir, '*.hdf5'))
        if not hdf5_files:
            raise FileNotFoundError(f"No preprocessed HDF5 files found in {self.preprocessed_data_dir}. "
                                    f"Please run 'preprocess_and_save_features.py' first.")

        all_X, all_y_dict_from_file = [], {name: [] for name in self.target_names_from_file}

        # 모든 HDF5 파일을 순회하며 데이터를 읽어 리스트에 추가합니다.
        for file_path in hdf5_files:
            try:
                with h5py.File(file_path, 'r') as hf:
                    all_X.append(hf['X'][:].astype(np.float32)) # 메모리 효율을 위해 타입 지정
                    for target_name in self.target_names_from_file:
                        if target_name in hf:
                            all_y_dict_from_file[target_name].append(hf[target_name][:])
                        else:
                             logger.warning(f"Target '{target_name}' not found in {file_path}. Skipping.")

            except Exception as e:
                logger.error(f"Failed to read HDF5 file {file_path}: {e}")
                continue
        
        if not all_X:
            raise ValueError("No data could be loaded from HDF5 files.")

        # 리스트에 담긴 데이터들을 NumPy 배열 하나로 합칩니다.
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = {
            key.replace('target_', ''): np.concatenate(val, axis=0) 
            for key, val in all_y_dict_from_file.items() if val
        }
        
        logger.info(f"Data loading complete. Total samples: {len(X_combined)}")
        return X_combined, y_combined
        
    def run_training_pipeline(self, hyperparameters: Dict[str, Any]):
        """모델 훈련의 전체 과정을 실행합니다."""
        
        # 1. 전처리된 데이터를 로드합니다.
        X_2d_np, y_dict_np = self._load_preprocessed_data()
        
        # 2. 'trade_outcome'을 기준으로 데이터를 훈련 세트와 검증 세트로 분할합니다 (80:20 비율).
        main_target_name = settings.ACTIVE_OUTPUTS[0]
        if main_target_name not in y_dict_np:
            raise ValueError(f"Main target '{main_target_name}' not found in loaded data.")

        X_train_np, X_val_np, _, _ = train_test_split(
            X_2d_np, y_dict_np[main_target_name], test_size=0.2, shuffle=False
        )

        train_size = len(X_train_np)
        y_train_dict = {key: val[:train_size] for key, val in y_dict_np.items()}
        y_val_dict = {key: val[train_size:] for key, val in y_dict_np.items()}
        
        logger.info(f"Data split complete. Train size: {len(X_train_np)}, Validation size: {len(X_val_np)}")

        # 3. 클래스 불균형 문제를 해결하기 위해 클래스별 가중치를 계산합니다.
        trade_outcome_train = y_train_dict[main_target_name]
        unique_classes = np.unique(trade_outcome_train)
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=unique_classes, y=trade_outcome_train
        )
        class_weights_dict = dict(zip(unique_classes, class_weights))
        logger.info(f"Calculated class weights for generator: {class_weights_dict}")
        
        # 4. 훈련용 및 검증용 DataGenerator를 생성합니다.
        train_generator = DataGenerator(
            X_train_np, y_train_dict, class_weights_dict, 
            self.sequence_length, hyperparameters['batch_size'], shuffle=True
        )
        val_generator = DataGenerator(
            X_val_np, y_val_dict, class_weights_dict, 
            self.sequence_length, hyperparameters['batch_size'], shuffle=False
        )
        
        # 5. Predictor 객체를 통해 실제 모델 훈련을 시작합니다.
        self.predictor.train(train_generator, val_generator, hyperparameters)
        logger.info("ModelTrainer has completed the training pipeline.")
