# /core/order_manager.py

import logging
import uuid
import math
from time import sleep
from typing import Optional, Dict, Any, Tuple, Union

# 시스템의 다른 핵심 모듈을 임포트합니다.
try:
    from core.bybit_client import BybitClient
    from configs import settings
except ImportError:
    print("[CRITICAL ERROR] Failed to import modules in order_manager.py.")
    raise SystemExit("Module loading failed.")


logger = logging.getLogger(__name__)

class OrderManager:
    """
    [ v5.6 - 동적 주문 전략 적용 ]
    주문 시점의 호가 스프레드를 분석하여 지정가(Maker)와 시장가(Taker) 중
    가장 비용 효율적인 주문 방식을 자동으로 선택합니다.
    """

    def __init__(self, bybit_client: BybitClient):
        self.client = bybit_client
        self.instrument_info_cache: Dict[str, Dict] = {}
        logger.info(f"OrderManager initialized with Dynamic Order Strategy: {settings.DYNAMIC_ORDER_STRATEGY}")

    # ... (기존 _get_instrument_info, _adjust_qty_to_step 함수는 변경 없음)
    def _get_instrument_info(self, symbol: str) -> Optional[Dict]:
        if symbol in self.instrument_info_cache: return self.instrument_info_cache[symbol]
        try:
            response = self.client.get_instruments_info(category="linear", symbol=symbol)
            if response and response.get('retCode') == 0 and response['result']['list']:
                info = response['result']['list'][0]
                self.instrument_info_cache[symbol] = info
                return info
        except Exception as e: logger.error(f"[{symbol}] Exception fetching instrument info: {e}")
        return None

    def _adjust_qty_to_step(self, symbol: str, quantity: float) -> Optional[str]:
        info = self._get_instrument_info(symbol)
        if not info: return None
        lot_size_filter = info.get('lotSizeFilter', {})
        qty_step_str = lot_size_filter.get('qtyStep')
        if not qty_step_str: return str(quantity)
        try:
            qty_step = float(qty_step_str)
            if qty_step <= 0: return str(quantity)
            adjusted_qty = math.floor(quantity / qty_step) * qty_step
            decimal_places = len(qty_step_str.split('.')[1]) if '.' in qty_step_str else 0
            return f"{adjusted_qty:.{decimal_places}f}"
        except (ValueError, TypeError) as e:
            logger.error(f"[{symbol}] Error adjusting quantity '{quantity}': {e}")
            return None
            
    def create_order(self, symbol: str, side: str, qty: Union[float, str], order_type: str, price: Optional[float] = None, reduce_only: bool = False, **kwargs) -> Optional[Dict]:
        """
        [✨ 핵심 수정] 모든 주문 요청을 받아, 동적 주문 전략에 따라 최적의 방식으로 실행합니다.
        """
        qty_str = str(qty)

        # 1. 동적 주문 전략 활성화 시
        if settings.DYNAMIC_ORDER_STRATEGY and order_type == "Market":
            ticker_info = self.client.get_tickers(category="linear", symbol=symbol)
            if not ticker_info or ticker_info.get('retCode') != 0 or not ticker_info['result']['list']:
                logger.error(f"[{symbol}] Could not get ticker for dynamic strategy. Falling back to Market order.")
                return self._execute_standard_order(symbol, side, "Market", qty_str, None, reduce_only, **kwargs)

            ticker_data = ticker_info['result']['list'][0]
            try:
                bid_price = float(ticker_data['bid1Price'])
                ask_price = float(ticker_data['ask1Price'])
                
                if bid_price <= 0 or ask_price <= 0: raise ValueError("Invalid bid/ask price")

                spread_pct = ((ask_price - bid_price) / bid_price) * 100
                logger.debug(f"[{symbol}] Spread: {spread_pct:.4f}%. Threshold: {settings.DYNAMIC_ORDER_SPREAD_THRESHOLD_PCT:.4f}%")

                # 2. 스프레드 기반 의사결정
                # 스프레드가 임계값보다 넓으면 -> 지정가 시도 (비용 절약)
                if spread_pct > settings.DYNAMIC_ORDER_SPREAD_THRESHOLD_PCT:
                    logger.info(f"[{symbol}] Wide spread detected. Attempting MAKER order to save cost.")
                    return self._try_maker_order(symbol, side, qty_str, reduce_only, ticker_data, **kwargs)
                # 스프레드가 좁으면 -> 시장가 즉시 실행 (기회비용 절약)
                else:
                    logger.info(f"[{symbol}] Tight spread detected. Executing TAKER order for speed.")
                    return self._execute_standard_order(symbol, side, "Market", qty_str, None, reduce_only, **kwargs)

            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"[{symbol}] Error processing ticker for dynamic strategy: {e}. Falling back to Market order.")
                return self._execute_standard_order(symbol, side, "Market", qty_str, None, reduce_only, **kwargs)

        # 3. 기존 지정가 우선 방식 (DYNAMIC_ORDER_STRATEGY = False일 때)
        elif settings.PREFER_MAKER_ORDERS and order_type == "Market":
            ticker_data = self.client.get_tickers(category="linear", symbol=symbol)['result']['list'][0]
            logger.info(f"[{symbol}] Using legacy maker-first strategy...")
            maker_result = self._try_maker_order(symbol, side, qty_str, reduce_only, ticker_data, **kwargs)
            if maker_result:
                return maker_result
            logger.info(f"[{symbol}] Maker order failed or timed out. Falling back to market taker order.")
        
        # 4. 모든 조건에 해당하지 않으면 표준 주문 실행
        return self._execute_standard_order(symbol, side, order_type, qty_str, price, reduce_only, **kwargs)

    def _try_maker_order(self, symbol: str, side: str, qty_str: str, reduce_only: bool, ticker_data: Dict, **kwargs) -> Optional[Dict]:
        """Best Bid/Ask를 사용하여 체결 확률이 높은 지정가 주문을 시도합니다."""
        try:
            if side.upper() == 'BUY':
                maker_price = float(ticker_data['bid1Price'])
            else: # SELL
                maker_price = float(ticker_data['ask1Price'])

            if maker_price <= 0:
                logger.error(f"[{symbol}] Invalid maker price ({maker_price}). Aborting maker order.")
                return None

            response = self._execute_standard_order(symbol, side, "Limit", qty_str, maker_price, reduce_only, timeInForce="PostOnly", **kwargs)
            
            if not response or response.get('retCode') != 0: return None
            order_id = response['result'].get('orderId')
            if not order_id: return None
            
            logger.info(f"[{symbol}] Maker order placed (ID: {order_id}) at price ${maker_price}. Waiting up to {settings.MAKER_ORDER_TIMEOUT}s for fill...")
            
            if self._wait_for_order_fill(symbol, order_id):
                logger.info(f"[{symbol}] Maker order (ID: {order_id}) confirmed as FILLED!")
                return response
            
            logger.info(f"[{symbol}] Maker order not filled. Attempting to cancel...")
            cancelled, err_code = self._cancel_order(symbol, order_id)
            
            if cancelled: return None
            if err_code == 110001:
                logger.warning(f"[{symbol}] Cancel failed (Code 110001). Assuming it was filled. SUCCESS.")
                return response
            
            logger.error(f"[{symbol}] CRITICAL: Failed to cancel untracked maker order (ID: {order_id}). Code: {err_code}. Aborting.")
            return None

        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"[{symbol}] Failed to parse ticker data for maker price: {e}")
            return None
        except Exception as e:
            logger.error(f"[{symbol}] Exception in _try_maker_order: {e}", exc_info=True)
            return None

    def _execute_standard_order(self, symbol: str, side: str, order_type: str, qty_str: str, price: Optional[float] = None, reduce_only: bool = False, **kwargs) -> Optional[Dict]:
        """API에 직접 주문 요청을 보내는 최종 실행 함수."""
        try:
            params = {"orderLinkId": f"auto_{symbol.lower()}_{uuid.uuid4().hex[:10]}"}
            if order_type.upper() == 'LIMIT' and price: params['price'] = str(price)
            if reduce_only: params['reduceOnly'] = True
            
            if settings.SET_TPSL_ON_ENTRY and not reduce_only:
                if kwargs.get('take_profit'): params['takeProfit'] = str(kwargs.get('take_profit'))
                if kwargs.get('stop_loss'): params['stopLoss'] = str(kwargs.get('stop_loss'))

            kwargs.pop('take_profit', None)
            kwargs.pop('stop_loss', None)
            params.update(kwargs)

            logger.info(f"==> EXECUTING ORDER: [{symbol}] {side} {qty_str} @ {order_type} | Params: {params}")
            
            return self.client.place_order(
                category="linear", symbol=symbol, side=side, orderType=order_type, qty=qty_str, **params
            )
        except Exception as e:
            logger.error(f"[{symbol}] Exception during standard order execution: {e}", exc_info=True)
            return None

    # ... (이하 _wait_for_order_fill, _get_order_status, _cancel_order, create_scalping_order 함수는 변경 없음)
    def create_scalping_order(self, symbol: str, side: str, capital_allocation: float, **kwargs) -> Optional[Dict]:
        instrument_info = self._get_instrument_info(symbol)
        if not instrument_info:
            logger.error(f"[{symbol}] Could not get instrument info. Skipping order.")
            return None
        lot_size_filter = instrument_info.get('lotSizeFilter', {})
        min_order_amt_str = lot_size_filter.get('minOrderAmt')
        if min_order_amt_str:
            try:
                min_order_amt = float(min_order_amt_str)
                if capital_allocation < min_order_amt:
                    logger.warning(f"[{symbol}] Order REJECTED. Allocated capital ${capital_allocation:,.2f} is below min value of ${min_order_amt:,.2f} USDT.")
                    return None
            except ValueError:
                logger.error(f"[{symbol}] Could not parse minOrderAmt '{min_order_amt_str}'.")
        try:
            current_price = self.client.get_current_price(symbol)
            if current_price is None or current_price <= 0: return None
            raw_quantity = capital_allocation / current_price
            adjusted_qty_str = self._adjust_qty_to_step(symbol, raw_quantity)
            if adjusted_qty_str is None or float(adjusted_qty_str) <= 0:
                logger.error(f"[{symbol}] Adjusted quantity '{adjusted_qty_str}' is invalid for capital ${capital_allocation:,.2f}.")
                return None
            logger.info(f"[{symbol}] Scalping order quantity adjusted: Raw {raw_quantity:.8f} -> Final {adjusted_qty_str}")
            return self.create_order(symbol=symbol, side=side, qty=adjusted_qty_str, order_type="Market", **kwargs)
        except Exception as e:
            logger.error(f"[{symbol}] Error in create_scalping_order: {e}", exc_info=True)
            return None

    def _wait_for_order_fill(self, symbol: str, order_id: str) -> bool:
        for _ in range(settings.MAKER_ORDER_TIMEOUT):
            sleep(1)
            status = self._get_order_status(symbol, order_id)
            if status == "Filled": return True
            if status is None: return True
            if status in ["Cancelled", "Rejected", "Expired"]: return False
        return False

    def _get_order_status(self, symbol: str, order_id: str) -> Optional[str]:
        try:
            response = self.client.get_open_orders(category="linear", symbol=symbol, orderId=order_id)
            if response and response.get('retCode') == 0 and response['result']['list']:
                return response['result']['list'][0].get('orderStatus')
        except Exception: pass
        return None

    def _cancel_order(self, symbol: str, order_id: str) -> Tuple[bool, Optional[int]]:
        try:
            response = self.client.cancel_order(category="linear", symbol=symbol, orderId=order_id)
            if response and response.get('retCode') == 0:
                logger.info(f"[{symbol}] Order {order_id} cancelled successfully.")
                return True, 0
            err_code = response.get('retCode') if response else -1
            logger.warning(f"[{symbol}] Failed to cancel order {order_id}. Response: {response}")
            return False, err_code
        except Exception as e:
            logger.error(f"[{symbol}] Exception cancelling order {order_id}: {e}")
            return False, -1
