# dump_feature_data.py

import logging
import pandas as pd
import numpy as np
import os
import sys
import glob

# --- 시스템 경로 설정 및 로깅 ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from configs.logging_config import setup_logging
setup_logging()

# --- 필요한 모듈 임포트 ---
try:
    from ml.enhanced_feature_engineer import EnhancedFeatureEngineer
    from configs import settings
except ImportError as e:
    print(f"모듈 임포트 실패: {e}")
    sys.exit(1)

logger = logging.getLogger("DATA_DUMPER")

def combine_all_data(data_dir: str) -> pd.DataFrame:
    """모든 CSV 파일을 하나로 합칩니다."""
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files: raise FileNotFoundError(f"No CSV data found in {data_dir}")
    
    df_list = []
    for file in csv_files:
        try:
            symbol = os.path.basename(file).split('_')[0]
            df = pd.read_csv(file, low_memory=False)
            if not df.empty:
                df['symbol'] = symbol
                df_list.append(df)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not df_list: raise ValueError("No data loaded.")
    return pd.concat(df_list, ignore_index=True)


def main():
    """
    [값 확인 최종 버전]
    피쳐 엔지니어링의 최종 결과물(X, y)을 모두 포함한
    데이터프레임을 종목별 CSV 파일로 저장합니다.
    """
    logger.info("="*80)
    logger.info("===== STARTING DATA PIPELINE DUMP SCRIPT =====")
    logger.info("="*80)
    
    output_dir = os.path.join(project_root, "debug_outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    try:
        # 1. 피쳐 엔지니어 초기화 및 데이터 로딩
        feature_engineer = EnhancedFeatureEngineer()
        logger.info("Loading and combining all data files...")
        all_data = combine_all_data(settings.DATA_DIR)
        
        # 2. 피쳐 엔지니어링 실행
        logger.info("Running feature engineering process for all symbols...")
        # process_all_data는 이제 (X, y) 튜플의 딕셔너리를 반환합니다.
        processed_data_dict = feature_engineer.process_all_data(all_data)
        
        if not processed_data_dict:
            logger.error("!!! CRITICAL: Feature engineering returned no data. Check logs.")
            return

        logger.info("\n" + "="*80)
        logger.info(f"===== DUMPING FINAL FEATURE SETS TO CSV FILES in '{output_dir}' =====")
        logger.info("="*80)

        # 3. 모든 종목의 최종 데이터를 CSV 파일로 저장
        for symbol, (X, y_dict) in processed_data_dict.items():
            
            logger.info(f"Processing and saving data for [{symbol}]...")
            
            # X와 y를 다시 하나의 데이터프레임으로 결합
            y_df = pd.DataFrame(y_dict)
            # 인덱스가 맞지 않을 수 있으므로 리셋 후 합침
            final_df = pd.concat([X.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)

            # 파일 경로 지정
            output_path = os.path.join(output_dir, f"{symbol}_features_and_targets.csv")
            
            try:
                # CSV 파일로 저장
                final_df.to_csv(output_path, index=False)
                logger.info(f"✅ Successfully saved data for [{symbol}] to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save CSV for [{symbol}]: {e}")



        logger.info("\n" + "="*80)
        logger.info("✅ DATA DUMP COMPLETED.")
        logger.info(f"Please check the CSV files in the '{output_dir}' directory to inspect all generated values.")
        logger.info("="*80)

    except Exception as e:
        logger.critical("An unexpected error occurred during the data dump process.", exc_info=True)


if __name__ == '__main__':
    main()