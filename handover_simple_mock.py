# -*- coding: utf-8 -*-
import re

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

def handover_simple(conversation_id: str, email: str, summary: str, simulate_fail: bool = False) -> str:
    """
    簡易轉接真人（模擬版）
    Args:
        conversation_id: 模擬的對話 ID
        email: 使用者 Email（此版僅做基本格式檢查）
        summary: 對話摘要（會送入 mock API；此處不做實際保存）
        simulate_fail: 設為 True 可強制模擬 API 失敗
    Returns:
        str: 成功 -> "已為您轉接真人"
             失敗 -> "轉接真人時發生錯誤，請聯繫技術團隊協助"
    """
    # 基本 Email 檢查（不通過就視為失敗）
    if not EMAIL_RE.match(email or ""):
        return "轉接真人時發生錯誤，請聯繫技術團隊協助"

    payload = {
        "conversation_id": str(conversation_id),
        "email": email,
        "summary": (summary or "")[:500],  # 簡單截斷，避免過長
    }

    try:
        ok = _mock_api_call(payload, simulate_fail=simulate_fail)
        return "已為您轉接真人" if ok else "轉接真人時發生錯誤，請聯繫技術團隊協助"
    except Exception:
        return "轉接真人時發生錯誤，請聯繫技術團隊協助"

def _mock_api_call(payload: dict, simulate_fail: bool = False) -> bool:
    """
    假的 API 呼叫：預設成功。
    - simulate_fail=True 時強制回傳失敗
    - 或者當 conversation_id 以 "FAIL" 開頭時回傳失敗（方便寫測試）
    """
    if simulate_fail:
        return False
    if payload.get("conversation_id", "").upper().startswith("FAIL"):
        return False
    return True

# --- 範例 ---
if __name__ == "__main__":
    print(handover_simple("JTCG-CHAT-0001", "user@example.com", "使用者希望轉接真人"))
    # -> 已為您轉接真人

    print(handover_simple("FAIL-0002", "user@example.com", "使用者希望轉接真人"))
    # -> 轉接真人時發生錯誤，請聯繫技術團隊協助

    print(handover_simple("JTCG-CHAT-0003", "bad-email", "使用者希望轉接真人"))
    # -> 轉接真人時發生錯誤，請聯繫技術團隊協助