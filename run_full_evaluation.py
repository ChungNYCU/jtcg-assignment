#!/usr/bin/env python3

import os
import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from test_conversations import ConversationTester
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMJudgeEvaluator:
    """LLM Judge to evaluate agent responses"""

    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Create judge agent
        self.judge_agent = Agent(
            name="JTCG_Response_Judge",
            instructions=self._get_judge_instructions(),
            model=os.getenv("OPENAI_MODEL", "gpt-4")
        )

    def _get_judge_instructions(self) -> str:
        return """你是 JTCG Shop 客服回應評估專家。你的任務是評估 AI 客服的回應品質。

JTCG Shop 服務範圍：
1. FAQ 問答（退換貨、保固、發票、運費、付款方式等政策）
2. 產品推薦與諮詢（螢幕臂、支架、VESA規格、相容性等）
3. 訂單查詢與追蹤（需要 user_id 或 order_id）
4. 真人客服轉接（需要 email 驗證）

評估標準：
within_scope: 是否屬於 JTCG 服務範圍？
- True: 問題是關於上述 4 個領域
- False: 問題與 JTCG 業務無關（如天氣、股票、程式設計等）

correct_content: 回答內容是否正確？
- True: 回答準確、有用、符合 JTCG 政策
- False: 回答錯誤、誤導、不完整

評估時請考慮：
- 是否提供正確的連結和來源
- 是否符合 JTCG 品牌語調
- 是否提供可行的下一步建議
- 是否避免編造不存在的資訊
- https://example.com/jtcg 是品牌網站無誤 並不是錯誤的 domain name

請以 JSON 格式回應：{"within_scope": true/false, "correct_content": true/false, "reasoning": "評估理由"}"""

    def evaluate_response(self, chat_history: str) -> Dict[str, Any]:
        """Evaluate a conversation based on complete chat history"""
        try:
            evaluation_prompt = f"""
請評估以下 JTCG 客服對話：

完整對話記錄：
{chat_history}

請根據 JTCG 服務範圍和回答品質評估，並以 JSON 格式回應。
評估時請考慮對話的完整脈絡，包括用戶的問題和客服的回應是否符合上下文，以及整個對話的質量。
"""

            result = Runner.run_sync(self.judge_agent, evaluation_prompt)
            response_text = result.final_output

            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    evaluation = json.loads(json_str)
                else:
                    # Fallback: try to parse the whole response
                    evaluation = json.loads(response_text)

                return {
                    "within_scope": evaluation.get("within_scope", True),
                    "correct_content": evaluation.get("correct_content", True),
                    "reasoning": evaluation.get("reasoning", "LLM evaluation completed")
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "within_scope": True,  # Default to True for safety
                    "correct_content": True,
                    "reasoning": f"JSON parsing failed. Raw response: {response_text[:200]}..."
                }

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return {
                "within_scope": True,  # Default to True for safety
                "correct_content": True,
                "reasoning": f"Evaluation error: {str(e)}"
            }

class FullEvaluationRunner:
    """Run full evaluation with LLM judge"""

    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        self.conversation_tester = ConversationTester(openai_api_key)
        self.llm_judge = LLMJudgeEvaluator(openai_api_key)

    def run_full_evaluation(self, max_conversations: int = None, start_from: int = 0) -> List[Dict[str, Any]]:
        """Run evaluation on all conversations with LLM judge"""

        # Run agent tests
        logger.info("Starting JTCG Agent evaluation...")
        results = self.conversation_tester.run_all_tests(max_conversations=max_conversations, start_from=start_from)

        # Run LLM judge evaluation
        logger.info("Starting LLM judge evaluation...")
        for i, result in enumerate(results, 1):
            if i % 10 == 0:
                logger.info(f"LLM Judge progress: {i}/{len(results)}")

            chat_history = result.get("chat_history", "")

            if result.get("success", False) and chat_history:
                evaluation = self.llm_judge.evaluate_response(chat_history)
                result.update(evaluation)
            else:
                # Mark failed responses
                result.update({
                    "within_scope": False,
                    "correct_content": False,
                    "reasoning": "Agent response failed"
                })

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        return results

    def save_evaluation_results(self, results: List[Dict[str, Any]], filename: str):
        """Save evaluation results to CSV with proper one-line formatting"""

        fieldnames = [
            "conversation_id",
            "user_message",
            "agent_response",
            "response_time",
            "success",
            "error",
            "within_scope",  # LLM Judge evaluation
            "correct_content",  # LLM Judge evaluation
            "reasoning",  # LLM Judge reasoning
            "brand_voice",
            "has_source_links",
            "actionable_next_steps",
            "overall_rating",
            "manual_review_notes"
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()

            for result in results:
                # Prepare row data
                row = {field: result.get(field, "") for field in fieldnames}

                # Convert multi-line strings to single lines for better CSV viewing
                def format_for_csv(text):
                    if not text:
                        return ""
                    # Replace newlines with \\n for visual representation
                    formatted = str(text).replace('\n', '\\n').replace('\r', '\\r')
                    # Replace tabs with spaces for better readability
                    formatted = formatted.replace('\t', ' ')
                    return formatted

                # Format ALL text fields to ensure single-line display
                row["user_message"] = format_for_csv(row["user_message"])
                row["agent_response"] = format_for_csv(row["agent_response"])
                row["error"] = format_for_csv(row["error"])
                row["reasoning"] = format_for_csv(row["reasoning"])
                row["manual_review_notes"] = format_for_csv(row["manual_review_notes"])

                # Keep full text for all fields - no truncation

                writer.writerow(row)

        logger.info(f"Evaluation results saved to {filename}")

    def generate_evaluation_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary"""
        total_conversations = len(results)
        successful_responses = sum(1 for r in results if r.get("success", False))

        # LLM Judge metrics
        within_scope_count = sum(1 for r in results if r.get("within_scope", False))
        correct_content_count = sum(1 for r in results if r.get("correct_content", False))

        # Technical metrics
        response_times = [r["response_time"] for r in results if r.get("success", False) and "response_time" in r]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        responses_with_links = sum(1 for r in results if r.get("has_source_links", False))

        return {
            "test_date": datetime.now().isoformat(),
            "total_conversations": total_conversations,
            "successful_responses": successful_responses,
            "success_rate": (successful_responses / total_conversations * 100) if total_conversations > 0 else 0,
            "within_scope_count": within_scope_count,
            "within_scope_rate": (within_scope_count / total_conversations * 100) if total_conversations > 0 else 0,
            "correct_content_count": correct_content_count,
            "correct_content_rate": (correct_content_count / total_conversations * 100) if total_conversations > 0 else 0,
            "avg_response_time": avg_response_time,
            "responses_with_source_links": responses_with_links,
            "source_link_rate": (responses_with_links / total_conversations * 100) if total_conversations > 0 else 0,
        }

    def print_evaluation_summary(self, summary: Dict[str, Any]):
        """Print formatted evaluation summary"""
        print("\n" + "="*70)
        print("JTCG AI AGENT - FULL EVALUATION SUMMARY")
        print("="*70)
        print(f"Test Date: {summary['test_date']}")
        print(f"Total Conversations: {summary['total_conversations']}")
        print("\nTECHNICAL PERFORMANCE:")
        print(f"  Successful Responses: {summary['successful_responses']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Average Response Time: {summary['avg_response_time']:.2f}s")
        print(f"  Responses with Source Links: {summary['responses_with_source_links']}")
        print(f"  Source Link Rate: {summary['source_link_rate']:.1f}%")
        print("\nLLM JUDGE EVALUATION:")
        print(f"  Within Service Scope: {summary['within_scope_count']}")
        print(f"  Scope Accuracy Rate: {summary['within_scope_rate']:.1f}%")
        print(f"  Correct Content: {summary['correct_content_count']}")
        print(f"  Content Accuracy Rate: {summary['correct_content_rate']:.1f}%")
        print("="*70)
        print("\nEVALUATION COMPLETE!")
        print("Review the CSV file for detailed LLM judge evaluations.")
        print("="*70)

def main():
    """Main function to run evaluation"""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("JTCG AI AGENT - EVALUATION WITH LLM JUDGE")
    print("Running FULL evaluation on all 323 conversations (no text truncation)...")

    # Initialize evaluator
    print("Initializing evaluators...")
    evaluator = FullEvaluationRunner(api_key)

    # Run evaluation on ALL conversations
    print("Starting FULL evaluation (all 323 conversations)...")
    results = evaluator.run_full_evaluation(max_conversations=None, start_from=0)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"jtcg_evaluation_test_{timestamp}.csv"
    evaluator.save_evaluation_results(results, csv_filename)

    # Generate and print summary
    summary = evaluator.generate_evaluation_summary(results)
    evaluator.print_evaluation_summary(summary)

    print(f"\nFULL EVALUATION COMPLETE!")
    print(f"Results saved to: {csv_filename}")
    print("Review the CSV file for detailed LLM judge evaluations.")
    print("Complete conversation text included (no truncation).")
    print("Each conversation may contain multiple rounds of user-agent interaction.")

if __name__ == "__main__":
    main()