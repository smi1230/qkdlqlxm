# /core/bybit_client.py

import logging
import time
from typing import Optional, Dict, Any, Callable

# pybit 라이브러리를 직접 임포트합니다.
from pybit.unified_trading import HTTP

# 시스템의 중앙 설정 파일을 임포트합니다.
try:
    from configs import settings
except ImportError:
    print("[치명적 오류] bybit_client.py에서 설정 파일을 임포트하는데 실패했습니다.")
    raise SystemExit("모듈 로딩 실패.")


logger = logging.getLogger(__name__)

class BybitClient:
    """
    [ v5.0 서킷 브레이커 통합 완료 ]
    모든 Bybit API 통신을 위한 중앙 관문(Gateway) 클래스.
    내장된 서킷 브레이커 패턴을 통해 API 서버의 지속적인 오류로부터 시스템을 보호합니다.
    """
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.session: Optional[HTTP] = None
        
        # --- 서킷 브레이커(Circuit Breaker) 상태 변수 ---
        self.state = "CLOSED"  # 상태: CLOSED (정상), OPEN (차단), HALF_OPEN (복구 시도)
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.failure_threshold = settings.CB_FAILURE_THRESHOLD
        self.recovery_timeout = settings.CB_RECOVERY_TIMEOUT
        
        try:
            # 1. 원본 코드와 동일하게 HTTP 세션을 먼저 생성합니다.
            # settings.py에서 TRADING_MODE에 따라 testnet 변수가 올바르게 설정됩니다.
            self.session = HTTP(
                testnet=testnet, api_key=api_key, api_secret=api_secret, recv_window=20000
            )
            
            # 2. ✨ [핵심 수정] 생성된 세션의 endpoint 속성을 사용자의 설정("d", "t", "m")에 맞게 덮어씁니다.
            if settings.TRADING_MODE == "d":
                self.session.endpoint = "https://api-demo.bybit.com"
                logger.warning(f"바이비트 데모 트레이딩 서버에 접속합니다 ({self.session.endpoint})")
            elif settings.TRADING_MODE == "t":
                logger.warning(f"바이비트 테스트넷 서버에 접속합니다 ({self.session.endpoint})")
            else: # "m" 또는 기타
                 logger.info(f"바이비트 메인넷 서버에 접속합니다 ({self.session.endpoint})")

            # 3. 초기 연결 상태 검증 (원본과 동일)
            server_time_res = self.get_server_time()
            if server_time_res and server_time_res.get('retCode') == 0:
                 logger.info("BybitClient 초기화 및 연결 검증 완료.")
            else:
                 raise ConnectionError(f"바이비트 API에 연결할 수 없습니다. 응답: {server_time_res}")

        except Exception as e:
            logger.critical(f"Bybit HTTP 세션 초기화 실패: {e}", exc_info=True)
            raise

    # --- 서킷 브레이커 로직 ---

    def _is_circuit_open(self) -> bool:
        """서킷이 OPEN 상태인지, 그리고 복구 시간이 지났는지 확인합니다."""
        if self.state == "OPEN":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.warning("서킷 브레이커 상태: HALF_OPEN. 테스트 호출을 시도합니다.")
                return False
            else:
                return True
        return False

    def _handle_success(self):
        """요청 성공 시 서킷 브레이커 상태를 리셋합니다."""
        if self.state == "HALF_OPEN":
            logger.info("서킷 브레이커 상태: CLOSED. 시스템 통신이 정상화되었습니다.")
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None

    def _handle_failure(self):
        """요청 실패 시 실패 횟수를 기록하고, 필요 시 서킷을 OPEN합니다."""
        self.failure_count += 1
        logger.warning(f"API 실패 횟수 증가: {self.failure_count}")
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_failure_time = time.time()
            logger.critical(f"🚨 서킷 브레이커 상태: OPEN. {self.failure_count}회 연속 실패로 {self.recovery_timeout}초 동안 API 호출을 차단합니다.")

    # --- 핵심 요청 처리 메서드 ---

    def _request_with_retry(self, api_call: Callable, **kwargs) -> Optional[Dict[str, Any]]:
        """
        [핵심 안정성] 서킷 브레이커와 재시도 로직을 결합한 모든 API 요청의 관문.
        """
        if self._is_circuit_open():
            logger.error(f"API 호출 '{api_call.__name__}' 차단됨. 서킷 브레이커가 OPEN 상태입니다.")
            return None

        retries = settings.API_REQUEST_RETRY_COUNT
        delay = settings.API_REQUEST_RETRY_DELAY

        last_response = None
        for i in range(retries):
            try:
                response = api_call(**kwargs)
                last_response = response
                if response and response.get('retCode') == 0:
                    self._handle_success()
                    return response

                if response and response.get('retCode') in [110007, 10005, 10004, 10003]:
                    logger.error(f"재시도 불가능한 API 오류. 코드: {response.get('retCode')}, 메시지: {response.get('retMsg')}")
                    return response

                logger.warning(f"API 호출 '{api_call.__name__}' 실패 (시도 {i+1}/{retries}). 응답: {response}")

            except Exception as e:
                logger.error(f"API 호출 '{api_call.__name__}' 중 예외 발생 (시도 {i+1}/{retries}): {e}")
                last_response = None # 예외 발생 시 응답 없음

            if i < retries - 1:
                time.sleep(delay)
        
        self._handle_failure()
        return last_response

    # =================================================================
    # [ 공식 API 래퍼 메서드 ]
    # 시스템의 다른 모든 모듈은 아래의 메서드만을 통해 API와 통신해야 합니다.
    # =================================================================

    def get_server_time(self) -> Optional[Dict[str, Any]]:
        """서버 시간을 조회합니다."""
        try:
            return self.session.get_server_time()
        except Exception as e:
            logger.error(f"서버 시간 조회 실패: {e}")
            return None

    def get_instruments_info(self, **kwargs) -> Optional[Dict[str, Any]]:
        """거래 상품 정보를 조회합니다."""
        return self._request_with_retry(self.session.get_instruments_info, **kwargs)

    def get_kline(self, **kwargs) -> Optional[Dict[str, Any]]:
        """K-line(캔들) 데이터를 조회합니다."""
        return self._request_with_retry(self.session.get_kline, **kwargs)

    def get_tickers(self, **kwargs) -> Optional[Dict[str, Any]]:
        """티커 정보를 조회합니다."""
        return self._request_with_retry(self.session.get_tickers, **kwargs)

    def get_positions(self, **kwargs) -> Optional[Dict[str, Any]]:
        """현재 보유 포지션 정보를 조회합니다."""
        return self._request_with_retry(self.session.get_positions, **kwargs)

    def get_wallet_balance(self, **kwargs) -> Optional[Dict[str, Any]]:
        """지갑 잔고를 조회합니다."""
        return self._request_with_retry(self.session.get_wallet_balance, **kwargs)

    def place_order(self, **kwargs) -> Optional[Dict[str, Any]]:
        """주문을 실행합니다."""
        return self._request_with_retry(self.session.place_order, **kwargs)

    def get_open_orders(self, **kwargs) -> Optional[Dict[str, Any]]:
        """미체결 주문 목록을 조회합니다."""
        return self._request_with_retry(self.session.get_open_orders, **kwargs)

    def cancel_order(self, **kwargs) -> Optional[Dict[str, Any]]:
        """단일 주문을 취소합니다."""
        # 주문 취소는 Race Condition 가능성 때문에 재시도 로직을 통과하되,
        # 실패 응답(예: 주문 없음)을 상위 로직에서 잘 처리하는 것이 중요합니다.
        return self._request_with_retry(self.session.cancel_order, **kwargs)

    def cancel_all_orders(self, **kwargs) -> Optional[Dict[str, Any]]:
        """특정 종목의 모든 미체결 주문을 취소합니다."""
        return self._request_with_retry(self.session.cancel_all_orders, **kwargs)

    # =================================================================
    # [ 중앙화된 유틸리티 메서드 ]
    # =================================================================

    def get_current_price(self, symbol: str) -> Optional[float]:
        """단일 종목의 현재가를 조회합니다."""
        try:
            ticker_info = self.get_tickers(category="linear", symbol=symbol)
            if ticker_info and ticker_info.get('retCode') == 0 and ticker_info.get('result', {}).get('list'):
                return float(ticker_info['result']['list'][0]['lastPrice'])
            else:
                 logger.warning(f"[{symbol}] 현재가 조회 실패. API 응답: {ticker_info}")
        except (IndexError, KeyError, ValueError) as e:
            logger.warning(f"[{symbol}] 티커 정보에서 현재가를 파싱할 수 없습니다: {e}")
        except Exception as e:
            logger.error(f"[{symbol}] 현재가 조회 중 예상치 못한 오류 발생: {e}", exc_info=False)
        return None
