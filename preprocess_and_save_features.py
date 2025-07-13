# /preprocess_and_save_features.py

import logging
import pandas as pd
import numpy as np
import os
import sys
import glob
import joblib
import h5py
from tqdm import tqdm

# # --- 시스템 경로 설정 및 로깅 ---
# project_root = os.path.dirname(os.path.abspath(__file__))
# if project_root not in sys.path:
#     sys.path.append(project_root)

from configs.logging_config import setup_logging
setup_logging()

from ml.enhanced_feature_engineer import EnhancedFeatureEngineer
from configs import settings

logger = logging.getLogger("PREPROCESSOR")

def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    피쳐와 타겟이 결합된 후 최종 정제를 수행하는 헬퍼 함수.
    """
    df_out = df.replace([np.inf, -np.inf], np.nan)
    df_out = df_out.ffill().bfill()
    
    numeric_cols = df_out.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if not str(col).startswith('target_'):
            p1, p99 = df_out[col].quantile(0.01), df_out[col].quantile(0.99)
            if pd.notna(p1) and pd.notna(p99) and p99 > p1:
                df_out[col] = df_out[col].clip(p1, p99)
    
    return df_out.fillna(0)

def main():
    """
    [핵심 전처리 - 종목별 독립 저장 및 책임 분리]
    - 이 스크립트가 데이터 흐름의 모든 책임을 집니다.
    - 각 종목별로 EFE를 호출하여 피쳐/타겟을 생성하고, 최종 정제 후 파일로 저장합니다.
    """
    logger.info("="*80)
    logger.info("===== STARTING SYMBOL-BY-SYMBOL FEATURE PREPROCESSING & SAVING SCRIPT =====")
    logger.info("="*80)

    source_dir = settings.DATA_DIR
    output_dir = os.path.join(source_dir, "preprocessed")
    model_artifact_dir = settings.MODEL_DIR

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    if not os.path.exists(model_artifact_dir):
        os.makedirs(model_artifact_dir)

    csv_files = glob.glob(os.path.join(source_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No original CSV data found in {source_dir}")
    
    feature_engineer = EnhancedFeatureEngineer()
    feature_names_saved = False

    logger.info(f"Starting sequential processing for {len(csv_files)} symbol files...")
    for file in tqdm(csv_files, desc="Processing files"):
        symbol = os.path.basename(file).split('_')[0]
        try:
            df = pd.read_csv(file, low_memory=False)
            
            if df.empty or len(df) < feature_engineer.longest_lookback: 
                logger.warning(f"Skipping {symbol} due to insufficient data ({len(df)} rows).")
                continue
            
            features_df = feature_engineer.create_features(df)
            targets_dict = feature_engineer.create_multitask_targets(df)
            targets_df = pd.DataFrame(targets_dict)

            # [핵심 수정] add_prefix를 사용하여 문법 오류를 해결하고 코드를 명확하게 만듭니다.
            targets_df_renamed = targets_df.add_prefix('target_')

            combined_df = pd.concat([features_df, targets_df_renamed], axis=1)

            cleaned_df = final_cleanup(combined_df)
            final_df = cleaned_df.dropna()
            
            if final_df.empty:
                logger.warning(f"No valid data rows for {symbol} after cleanup and dropna.")
                continue

            target_cols = list(targets_df_renamed.columns)
            feature_cols = [col for col in final_df.columns if col not in target_cols]
            
            X = final_df[feature_cols]
            y = final_df[target_cols]

            # --- 데이터 저장 ---
            csv_output_path = os.path.join(output_dir, f"{symbol}_processed.csv")
            final_df.to_csv(csv_output_path, index=False)
            
            hdf5_output_path = os.path.join(output_dir, f"{symbol}_dataset.hdf5")
            with h5py.File(hdf5_output_path, 'w') as hf:
                hf.create_dataset('X', data=X.to_numpy(dtype=np.float32), compression="gzip")
                for col in y.columns:
                     hf.create_dataset(col, data=y[col].to_numpy(), compression="gzip")
            
            if not feature_names_saved:
                joblib.dump(feature_cols, os.path.join(model_artifact_dir, "feature_names.joblib"))
                logger.info(f"Feature names saved to {os.path.join(model_artifact_dir, 'feature_names.joblib')}")
                feature_names_saved = True

        except Exception as e:
            logger.error(f"Failed to process file for symbol {symbol}: {e}", exc_info=True)

    logger.info("\n" + "="*80)
    logger.info("✅ SYMBOL-BY-SYMBOL PREPROCESSING COMPLETED SUCCESSFULLY!")
    logger.info(f"All preprocessed data saved in: {output_dir}")
    logger.info("You can now modify and run train.py to start model training using these files.")
    logger.info("="*80)

if __name__ == '__main__':
    main()