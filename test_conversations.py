#!/usr/bin/env python3

import os
import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from jtcg_agent import JTCGCRMAgent
from data_processor import DataProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationTester:
    def __init__(self, openai_api_key: str, output_file: str = "conversation_results.csv"):
        self.agent = JTCGCRMAgent(openai_api_key)
        self.output_file = output_file
        self.results = []

    def load_test_conversations(self, json_path: str) -> List[List[Dict]]:
        """Load test conversations from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        return conversations

    def test_single_conversation(self, conversation: List[Dict], conversation_id: int) -> Dict[str, Any]:
        """Test a single conversation and return results"""
        try:
            # Extract user messages from conversation
            if not conversation or len(conversation) == 0:
                return {
                    "conversation_id": conversation_id,
                    "user_message": "",
                    "agent_response": "ERROR: Empty conversation",
                    "chat_history": "ERROR: Empty conversation",
                    "response_time": 0,
                    "success": False,
                    "error": "Empty conversation"
                }

            # Collect all user messages and last agent response
            user_messages = []
            last_agent_response = ""

            for message in conversation:
                if message.get("role") == "user":
                    content = message.get("content", [])
                    if content and len(content) > 0:
                        user_messages.append(content[0].get("text", ""))
                elif message.get("role") == "assistant":
                    content = message.get("content", [])
                    if content and len(content) > 0:
                        last_agent_response = content[0].get("text", "")

            if not user_messages:
                return {
                    "conversation_id": conversation_id,
                    "user_message": "No user message found",
                    "agent_response": "ERROR: No user message in conversation",
                    "chat_history": "ERROR: No user message in conversation",
                    "response_time": 0,
                    "success": False,
                    "error": "No user message found"
                }

            # Create formatted chat history
            chat_history_parts = []

            for message in conversation:
                role = message.get("role", "")
                content = message.get("content", [])
                if content and len(content) > 0:
                    text = content[0].get("text", "")
                    if role == "user":
                        chat_history_parts.append(f"User: {text}")
                    elif role == "assistant":
                        chat_history_parts.append(f"Assistant: {text}")

            # For evaluation, get agent response to the last user message
            last_user_message = user_messages[-1] if user_messages else ""

            # Reset agent state for new conversation
            self.agent.reset_conversation()

            # Measure response time
            start_time = time.time()

            # Get agent response for the last user message
            try:
                if last_user_message:
                    agent_response = self.agent.chat(last_user_message)
                    # Add agent response to chat history
                    chat_history_parts.append(f"JTCG Agent: {agent_response}")
                else:
                    agent_response = "ERROR: No user message to respond to"

                response_time = time.time() - start_time
                success = True
                error = None
            except Exception as e:
                agent_response = f"ERROR: {str(e)}"
                chat_history_parts.append(f"JTCG Agent: {agent_response}")
                response_time = time.time() - start_time
                success = False
                error = str(e)

            # Format complete chat history
            full_chat_history = "\\n".join(chat_history_parts)

            return {
                "conversation_id": conversation_id,
                "user_message": full_chat_history,  # Complete conversation history
                "agent_response": agent_response,  # Final JTCG agent response only
                "chat_history": full_chat_history,  # Keep for LLM judge evaluation
                "response_time": response_time,
                "success": success,
                "error": error
            }

        except Exception as e:
            return {
                "conversation_id": conversation_id,
                "user_message": "ERROR parsing conversation",
                "agent_response": f"ERROR: {str(e)}",
                "chat_history": f"ERROR parsing conversation: {str(e)}",
                "response_time": 0,
                "success": False,
                "error": str(e)
            }

    def evaluate_response_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add manual evaluation fields for response quality assessment"""
        # This creates fields that need to be manually reviewed
        evaluation = {
            "within_scope": None,  # True/False - 是否符合 Agent 服務範圍
            "correct_content": None,  # True/False - 回答內容是否正確
            "brand_voice": None,  # True/False - 是否符合品牌語調
            "has_source_links": None,  # True/False - 是否包含適當的來源連結
            "actionable_next_steps": None,  # True/False - 是否提供可執行的下一步
            "manual_review_notes": "",  # String - 人工檢視備註
            "overall_rating": None  # 1-5 - 整體評分
        }

        # Auto-detect some basic qualities
        chat_history = result.get("chat_history", "").lower()

        # Check for source links in the JTCG Agent responses
        if "http" in chat_history or "[" in chat_history and "](" in chat_history:
            evaluation["has_source_links"] = True
        else:
            evaluation["has_source_links"] = False

        # Check for JTCG brand mentions
        if "jtcg" in chat_history:
            evaluation["brand_voice"] = True

        return evaluation

    def run_all_tests(self, conversations_path: str = "ref_data/ai-eng-test-sample-conversations.json",
                     max_conversations: int = None, start_from: int = 0) -> List[Dict[str, Any]]:
        """Run tests on all conversations"""
        logger.info("Loading test conversations...")
        conversations = self.load_test_conversations(conversations_path)

        # Apply start_from offset
        conversations = conversations[start_from:]

        total_conversations = len(conversations)
        if max_conversations:
            conversations = conversations[:max_conversations]
            total_conversations = len(conversations)

        logger.info(f"Testing {total_conversations} conversations (starting from #{start_from + 1})...")

        results = []
        for i, conversation in enumerate(conversations, 1):
            logger.info(f"Processing conversation {i}/{total_conversations}")

            # Test the conversation
            result = self.test_single_conversation(conversation, i)

            # Add evaluation fields
            evaluation = self.evaluate_response_quality(result)
            result.update(evaluation)

            results.append(result)

            # Log progress every 10 conversations
            if i % 10 == 0:
                success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
                avg_time = sum(r["response_time"] for r in results) / len(results)
                logger.info(f"Progress: {i}/{total_conversations} - Success rate: {success_rate:.1f}% - Avg time: {avg_time:.2f}s")

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        self.results = results
        return results

    def save_results_to_csv(self, results: List[Dict[str, Any]], filename: str = None):
        """Save test results to CSV file for manual review"""
        if not filename:
            filename = self.output_file

        # Ensure output directory exists
        output_path = Path(filename)
        output_path.parent.mkdir(exist_ok=True)

        fieldnames = [
            "conversation_id",
            "user_message",
            "agent_response",
            "response_time",
            "success",
            "error",
            "within_scope",
            "correct_content",
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

                # Format text fields
                row["user_message"] = format_for_csv(row["user_message"])
                row["agent_response"] = format_for_csv(row["agent_response"])
                row["error"] = format_for_csv(row["error"])
                row["manual_review_notes"] = format_for_csv(row["manual_review_notes"])

                # Keep full text for user_message and agent_response - no truncation

                writer.writerow(row)

        logger.info(f"Results saved to {filename}")

    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary report of the test results"""
        total_conversations = len(results)
        successful_responses = sum(1 for r in results if r["success"])
        failed_responses = total_conversations - successful_responses

        response_times = [r["response_time"] for r in results if r["success"]]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0

        # Count responses with source links
        responses_with_links = sum(1 for r in results if r.get("has_source_links"))

        report = {
            "test_date": datetime.now().isoformat(),
            "total_conversations": total_conversations,
            "successful_responses": successful_responses,
            "failed_responses": failed_responses,
            "success_rate": (successful_responses / total_conversations * 100) if total_conversations > 0 else 0,
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "responses_with_source_links": responses_with_links,
            "source_link_rate": (responses_with_links / total_conversations * 100) if total_conversations > 0 else 0
        }

        return report

    def print_summary_report(self, report: Dict[str, Any]):
        """Print a formatted summary report"""
        print("\n" + "="*60)
        print("JTCG AI AGENT - CONVERSATION TEST SUMMARY")
        print("="*60)
        print(f"Test Date: {report['test_date']}")
        print(f"Total Conversations: {report['total_conversations']}")
        print(f"Successful Responses: {report['successful_responses']}")
        print(f"Failed Responses: {report['failed_responses']}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        print(f"Average Response Time: {report['avg_response_time']:.2f}s")
        print(f"Min Response Time: {report['min_response_time']:.2f}s")
        print(f"Max Response Time: {report['max_response_time']:.2f}s")
        print(f"Responses with Source Links: {report['responses_with_source_links']}")
        print(f"Source Link Rate: {report['source_link_rate']:.1f}%")
        print("="*60)
        print("\nNext Steps:")
        print("1. Review the CSV file for manual evaluation")
        print("2. Mark 'within_scope' and 'correct_content' columns")
        print("3. Add overall ratings and review notes")
        print("4. Generate final evaluation report")
        print("="*60)

def main():
    """Main function to run conversation tests"""
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        print("Copy .env.example to .env and add your API key")
        return

    # Initialize tester
    print("Initializing conversation tester...")
    tester = ConversationTester(api_key)

    # Ask user for test scope
    try:
        max_conv = input("Enter max conversations to test (press Enter for all 323): ").strip()
        max_conversations = int(max_conv) if max_conv else None
    except ValueError:
        max_conversations = None

    # Run tests
    print(f"Starting conversation tests...")
    results = tester.run_all_tests(max_conversations=max_conversations)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"conversation_results_{timestamp}.csv"
    tester.save_results_to_csv(results, csv_filename)

    # Generate and print summary
    report = tester.generate_summary_report(results)
    tester.print_summary_report(report)

    print(f"\nResults saved to: {csv_filename}")
    print("Please review the CSV file and mark the evaluation columns manually.")

if __name__ == "__main__":
    main()