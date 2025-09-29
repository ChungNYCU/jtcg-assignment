#!/usr/bin/env python3

import os
from typing import Dict, Any
from agents import Agent, Runner, function_tool
from data_processor import DataProcessor
from vector_db import VectorDB
from agent_functions import JTCGAgentFunctions

class JTCGCRMAgent:
    def __init__(self, openai_api_key: str):
        # Initialize data components
        self.data_processor = DataProcessor()
        self.vector_db = VectorDB()
        self.agent_functions = None
        self.agent = None

        # Load data
        self._load_data()

        # Create agent functions
        self.agent_functions = JTCGAgentFunctions(self.data_processor, self.vector_db)

        # Initialize the OpenAI Agent
        self._setup_agent(openai_api_key)

    def _load_data(self):
        """Load all data sources"""
        print("Loading JTCG data...")
        self.data_processor.load_all_data(
            knowledge_csv="ref_data/ai-eng-test-sample-knowledges.csv",
            products_csv="ref_data/ai-eng-test-sample-products.csv",
            orders_json="ref_data/orders.json",
            conversations_json="ref_data/ai-eng-test-sample-conversations.json"
        )

        # Initialize vector database
        self.vector_db.initialize_with_data(self.data_processor)
        print("Data loading complete!")

    def _setup_agent(self, openai_api_key: str):
        """Setup the OpenAI Agent with function tools"""

        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Get model from environment or use default
        model_name = os.getenv("OPENAI_MODEL", "gpt-4")

        # Define function tools using the function_tool decorator
        @function_tool
        def search_knowledge_base(query: str, max_results: int = 3) -> Dict[str, Any]:
            """Search the JTCG knowledge base for FAQ, policies, and support information"""
            return self.agent_functions.search_knowledge_base(query, max_results)

        @function_tool
        def search_products(query: str, max_results: int = 5) -> Dict[str, Any]:
            """Search for JTCG products and provide recommendations based on user requirements"""
            return self.agent_functions.search_products(query, None, max_results)

        @function_tool
        def lookup_user_orders(user_id: str) -> Dict[str, Any]:
            """Look up all orders for a specific user ID (e.g., u_123456)"""
            return self.agent_functions.lookup_user_orders(user_id)

        @function_tool
        def lookup_order_details(order_id: str) -> Dict[str, Any]:
            """Look up detailed information for a specific order ID (e.g., JTCG-202508-10001)"""
            return self.agent_functions.lookup_order_details(order_id)

        @function_tool
        def handover_to_human(email: str, conversation_summary: str, conversation_id: str = "") -> Dict[str, Any]:
            """Transfer conversation to human customer service"""
            return self.agent_functions.handover_to_human(email, conversation_summary, conversation_id if conversation_id else None)

        # Create the agent
        self.agent = Agent(
            name="JTCG_CRM_Agent",
            instructions=self._get_system_instructions(),
            model=model_name,
            tools=[search_knowledge_base, search_products, lookup_user_orders, lookup_order_details, handover_to_human]
        )


    def _get_system_instructions(self) -> str:
        """Get the system instructions for the agent"""
        return """你是 JTCG Shop 的客服人員。JTCG Shop 專注於工作空間體驗與周邊配件的選品與設計，包含螢幕臂、壁掛支架、走線收納與安裝配件等。

品牌主張：Better Desk, Better Focus.
核心特色：相容性清楚、安裝不踩雷、售後好溝通。

重要：絕對不要使用以下 AI 機器人用語：
禁止使用：「簡短回答」「補充說明」「詳細說明」「總結」
禁止使用：「讓我為您...」「我將為您...」等助理語言
禁止回答超出職責範圍的內容，比如: 政治問題、數學問題、程式問題、情感問題

你的回應風格：
- 自然對話，直接回答，像真正的客服人員
- 不使用任何 AI 助理的用語格式
- 先直答、再補充，避免冗長
- 語系跟隨使用者最新訊息（繁體/簡體中文一致）
- 有來源就明確附上連結

良好回應範例：
用戶：「請問你們的退換貨政策是什麼？」
正確回應：「我們提供 7 天鑑賞期（含例假日），商品需保持全新、完整包裝與配件。若非瑕疵退換貨，可能需自行負擔來回運費。

詳細條款請參考：https://example.com/jtcg/policies/returns

需要我協助查詢您的訂單是否可以退換嗎？」

錯誤回應：「簡短回答：退換貨政策如下... 補充說明：...」

用戶：「保固多久？」
正確回應：「保固期間依商品而異，一般臂架產品享有 1 年保固。需要提供訂單編號或發票資訊申請維修。

保固說明請看：https://example.com/jtcg/policies/warranty

要我幫您查詢特定商品的保固狀況嗎？」

錯誤回應：「簡短回答：保固期限依商品而異... 詳細說明請參考...」

用戶：「有雙螢幕臂推薦嗎？」
正確回應：「推薦您這款 JTCG 雙螢幕氣壓臂 Pro，支援至 32 吋螢幕，VESA 75x75 和 100x100 都能裝。承重 2-9kg，適合桌板厚度 10-85mm。

想了解您的螢幕尺寸和桌面情況嗎？這樣我能給您更精準的建議。」

核心功能範圍：
A. FAQ 智能回覆 - 使用 search_knowledge_base 工具
B. 產品探索與建議 - 使用 search_products 工具
C. 訂單服務查詢 - 使用 lookup_user_orders 和 lookup_order_details 工具
D. 真人客服轉接 - 使用 handover_to_human 工具

互動原則：
- 理解使用者最新意圖，必要時簡短澄清
- 先解決當前問題，再適度補充
- 必要資訊就近索取（user_id、Email等）
- 提供可立即執行的下一步引導

記住：以自然、專業、貼心的方式回應，就像真正的 JTCG Shop 客服人員。"""

    def chat(self, message: str, conversation_input=None) -> str:
        """
        Process a chat message and return response

        Args:
            message: User's message or input list for multi-turn conversations
            conversation_input: Optional conversation input list (for multi-turn)

        Returns:
            Agent's response
        """
        try:
            # If conversation_input is provided, use it directly (it includes history)
            if conversation_input is not None:
                # conversation_input already includes the history and current message
                result = Runner.run_sync(self.agent, conversation_input)
                return result.final_output
            else:
                # Single turn conversation
                # Detect intent to provide context
                intent = self.agent_functions.detect_intent(message)

                # Add intent context to the message
                contextual_message = f"[用戶意圖: {intent}] {message}"

                # Run the agent using Runner.run_sync
                result = Runner.run_sync(self.agent, contextual_message)
                return result.final_output

        except Exception as e:
            return f"很抱歉，處理您的請求時發生錯誤。請稍後再試或聯繫我們的客服團隊。錯誤信息：{str(e)}"

    def reset_conversation(self):
        """Reset the conversation state"""
        # OpenAI Agents SDK handles conversation state automatically
        # No need to recreate the agent for each conversation
        pass

def main():
    """Main function for testing the agent"""
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Initialize agent
    print("Initializing JTCG CRM Agent...")
    agent = JTCGCRMAgent(api_key)

    # Interactive chat loop
    print("\nJTCG Shop AI 客服助理已就緒！")
    print("輸入 'quit' 結束對話")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n您：").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("感謝您使用 JTCG Shop 客服服務！")
                break

            if not user_input:
                continue

            print("AI 客服：", end="")
            response = agent.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\n感謝您使用 JTCG Shop 客服服務！")
            break
        except Exception as e:
            print(f"錯誤：{e}")

if __name__ == "__main__":
    main()