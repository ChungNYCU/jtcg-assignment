from typing import List, Dict, Any, Optional
import json
import uuid
from data_processor import DataProcessor
from vector_db import VectorDB
from handover_simple_mock import handover_simple
import re

class JTCGAgentFunctions:
    def __init__(self, data_processor: DataProcessor, vector_db: VectorDB):
        self.data_processor = data_processor
        self.vector_db = vector_db

    def search_knowledge_base(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Search the knowledge base for FAQ and support information

        Args:
            query: User's question or search terms
            max_results: Maximum number of results to return

        Returns:
            Dictionary with search results and formatted response
        """
        try:
            results = self.vector_db.search_knowledge(query, n_results=max_results)

            if not results:
                return {
                    "success": False,
                    "message": "很抱歉，目前無法找到相關的資訊。建議您聯繫我們的客服團隊以獲得進一步協助。",
                    "results": []
                }

            # Format the response with source links
            formatted_results = []
            for result in results:
                item = {
                    "title": result["title"],
                    "relevance_score": 1 - result["distance"],  # Convert distance to relevance
                    "url": result["url_href"] if result["url_href"] else "",
                    "url_label": result["url_label"] if result["url_label"] else "",
                    "image_url": result["image_url"] if result["image_url"] else "",
                    "tags": result["tags"]
                }
                formatted_results.append(item)

            # Get full content for the best match
            best_match = results[0]
            kb_item = self.data_processor.get_knowledge_by_id(best_match["id"])

            response_parts = []
            if kb_item:
                response_parts.append(kb_item.content)
                if kb_item.url_href:
                    response_parts.append(f"\n\n詳細資訊請參考：[{kb_item.url_label or '詳細說明'}]({kb_item.url_href})")
                if kb_item.image_url:
                    response_parts.append(f"\n\n相關圖片：{kb_item.image_url}")

            return {
                "success": True,
                "message": "".join(response_parts),
                "results": formatted_results,
                "primary_source": {
                    "title": best_match["title"],
                    "url": best_match["url_href"],
                    "url_label": best_match["url_label"]
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": "搜尋時發生錯誤，請稍後再試或聯繫客服團隊。",
                "error": str(e),
                "results": []
            }

    def search_products(self, query: str, specifications: Optional[Dict] = None, max_results: int = 5) -> Dict[str, Any]:
        """
        Search for products based on user requirements

        Args:
            query: User's product search query
            specifications: Optional dict with specific requirements (size, vesa, etc.)
            max_results: Maximum number of products to return

        Returns:
            Dictionary with product recommendations
        """
        try:
            # Enhance query with specifications if provided
            enhanced_query = query
            if specifications:
                spec_parts = []
                for key, value in specifications.items():
                    if value:
                        spec_parts.append(f"{key}: {value}")
                if spec_parts:
                    enhanced_query += " " + " ".join(spec_parts)

            results = self.vector_db.search_products(enhanced_query, n_results=max_results)

            if not results:
                return {
                    "success": False,
                    "message": "很抱歉，沒有找到符合您需求的產品。請提供更多資訊，如螢幕尺寸、VESA規格或使用情境，以便我為您推薦合適的產品。",
                    "products": []
                }

            # Format product results with compatibility info
            formatted_products = []
            for result in results:
                product_info = {
                    "sku": result["sku"],
                    "name": result["name"],
                    "arm_type": result["arm_type"],
                    "max_size": result["size_max_inch"],
                    "vesa_options": result["vesa_options"],
                    "weight_capacity": result["weight_per_arm_kg"],
                    "desk_thickness": result["desk_thickness_mm"],
                    "compatibility_notes": result["compatibility_notes"],
                    "url": result["url"],
                    "image_url": result["image_url"],
                    "includes": result["includes"],
                    "relevance_score": 1 - result["distance"]
                }
                formatted_products.append(product_info)

            # Generate response with top recommendations
            response_parts = ["以下是為您推薦的產品：\n"]

            for i, product in enumerate(formatted_products[:3], 1):
                response_parts.append(f"\n{i}. **{product['name']}** ({product['sku']})")

                specs = []
                if product['max_size']:
                    specs.append(f"支援至 {product['max_size']} 吋")
                if product['vesa_options']:
                    specs.append(f"VESA: {', '.join(product['vesa_options'])}")
                if product['weight_capacity']:
                    specs.append(f"承重: {product['weight_capacity']} kg")

                if specs:
                    response_parts.append(f" - {', '.join(specs)}")

                if product['compatibility_notes']:
                    response_parts.append(f"\n   注意事項: {product['compatibility_notes']}")

                if product['url']:
                    response_parts.append(f"\n   [查看詳情]({product['url']})")

                response_parts.append("\n")

            response_parts.append("\n如需更詳細的建議，請告訴我您的螢幕尺寸、桌面厚度或特殊需求。")

            return {
                "success": True,
                "message": "".join(response_parts),
                "products": formatted_products,
                "total_found": len(results)
            }

        except Exception as e:
            return {
                "success": False,
                "message": "產品搜尋時發生錯誤，請稍後再試或聯繫客服團隊。",
                "error": str(e),
                "products": []
            }

    def lookup_user_orders(self, user_id: str) -> Dict[str, Any]:
        """
        Look up orders for a specific user

        Args:
            user_id: User identifier

        Returns:
            Dictionary with user's orders
        """
        try:
            orders = self.data_processor.get_orders_by_user_id(user_id)

            if not orders:
                return {
                    "success": False,
                    "message": f"查無用戶 {user_id} 的訂單記錄。請確認用戶ID是否正確，或聯繫客服協助查詢。",
                    "orders": []
                }

            # Format orders for display
            formatted_orders = []
            for order in orders:
                order_info = {
                    "order_id": order.order_id,
                    "placed_at": order.placed_at,
                    "status": order.status,
                    "carrier": order.carrier,
                    "tracking": order.tracking,
                    "eta": order.eta,
                    "items": order.items,
                    "shipping_address": order.shipping_address,
                    "contact_phone": order.contact_phone,
                    "order_url": order.order_url
                }
                formatted_orders.append(order_info)

            # Generate response
            response_parts = [f"找到用戶 {user_id} 的 {len(orders)} 筆訂單：\n"]

            for i, order in enumerate(formatted_orders, 1):
                status_display = {
                    "processing": "處理中",
                    "shipped": "已出貨",
                    "in_transit": "運送中",
                    "delivered": "已送達"
                }.get(order["status"], order["status"])

                response_parts.append(f"\n{i}. 訂單 {order['order_id']}")
                response_parts.append(f" - 狀態: {status_display}")

                if order["carrier"] and order["tracking"]:
                    response_parts.append(f" - 物流: {order['carrier']} ({order['tracking']})")

                if order["eta"]:
                    response_parts.append(f" - 預計到貨: {order['eta']}")

                response_parts.append(f" - 商品: {', '.join([item['name'] for item in order['items']])}")

                if order["order_url"]:
                    response_parts.append(f" - [查看訂單詳情]({order['order_url']})")

                response_parts.append("\n")

            response_parts.append("如需查詢特定訂單的詳細資訊，請提供訂單編號。")

            return {
                "success": True,
                "message": "".join(response_parts),
                "orders": formatted_orders,
                "user_id": user_id,
                "total_orders": len(orders)
            }

        except Exception as e:
            return {
                "success": False,
                "message": "查詢訂單時發生錯誤，請稍後再試或聯繫客服團隊。",
                "error": str(e),
                "orders": []
            }

    def lookup_order_details(self, order_id: str) -> Dict[str, Any]:
        """
        Look up detailed information for a specific order

        Args:
            order_id: Order identifier

        Returns:
            Dictionary with order details
        """
        try:
            order = self.data_processor.get_order_by_id(order_id)

            if not order:
                return {
                    "success": False,
                    "message": f"查無訂單 {order_id}。請確認訂單編號是否正確，或聯繫客服協助查詢。",
                    "order": None
                }

            # Format order details
            status_display = {
                "processing": "處理中",
                "shipped": "已出貨",
                "in_transit": "運送中",
                "delivered": "已送達"
            }.get(order.status, order.status)

            response_parts = [f"訂單 {order.order_id} 詳細資訊：\n"]
            response_parts.append(f"狀態: {status_display}")
            response_parts.append(f"下單時間: {order.placed_at}")

            if order.carrier:
                response_parts.append(f"物流商: {order.carrier}")
            if order.tracking:
                response_parts.append(f"追蹤號碼: {order.tracking}")
            if order.eta:
                response_parts.append(f"預計到貨: {order.eta}")

            response_parts.append(f"\n購買商品:")
            for item in order.items:
                response_parts.append(f"- {item['name']} x{item['qty']}")

            response_parts.append(f"\n配送地址: {order.shipping_address}")
            response_parts.append(f"聯絡電話: {order.contact_phone}")

            if order.order_url:
                response_parts.append(f"\n[查看完整訂單]({order.order_url})")

            return {
                "success": True,
                "message": "\n".join(response_parts),
                "order": {
                    "order_id": order.order_id,
                    "status": order.status,
                    "placed_at": order.placed_at,
                    "carrier": order.carrier,
                    "tracking": order.tracking,
                    "eta": order.eta,
                    "items": order.items,
                    "shipping_address": order.shipping_address,
                    "contact_phone": order.contact_phone,
                    "order_url": order.order_url
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": "查詢訂單詳情時發生錯誤，請稍後再試或聯繫客服團隊。",
                "error": str(e),
                "order": None
            }

    def handover_to_human(self, email: str, conversation_summary: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Transfer conversation to human customer service

        Args:
            email: User's email address
            conversation_summary: Summary of the conversation so far
            conversation_id: Optional conversation identifier

        Returns:
            Dictionary with handover result
        """
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = f"JTCG-CHAT-{str(uuid.uuid4())[:8]}"

            # Validate email format
            email_pattern = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
            if not email_pattern.match(email or ""):
                return {
                    "success": False,
                    "message": "請提供有效的Email地址以便我們為您轉接真人客服。",
                    "conversation_id": conversation_id
                }

            # Call the handover function
            result = handover_simple(conversation_id, email, conversation_summary)

            if result == "已為您轉接真人":
                return {
                    "success": True,
                    "message": f"已為您轉接真人客服，請稍候。您的服務案件編號是: {conversation_id}",
                    "conversation_id": conversation_id,
                    "email": email
                }
            else:
                return {
                    "success": False,
                    "message": result,
                    "conversation_id": conversation_id
                }

        except Exception as e:
            return {
                "success": False,
                "message": "轉接真人客服時發生錯誤，請聯繫技術團隊協助。",
                "error": str(e),
                "conversation_id": conversation_id or "ERROR"
            }

    def detect_intent(self, user_message: str) -> str:
        """
        Detect the user's intent from their message

        Args:
            user_message: User's input message

        Returns:
            Intent category: 'faq', 'product', 'order', 'handover', or 'general'
        """
        message_lower = user_message.lower()

        # Order-related keywords
        order_keywords = ['訂單', '物流', '追蹤', '配送', '出貨', '到貨', 'order', 'tracking', 'shipping', 'delivery']
        if any(keyword in message_lower for keyword in order_keywords):
            return 'order'

        # Product search keywords
        product_keywords = ['產品', '臂架', '支架', '螢幕', '推薦', '規格', '尺寸', 'vesa', 'arm', 'monitor', 'product']
        if any(keyword in message_lower for keyword in product_keywords):
            return 'product'

        # Human handover keywords
        handover_keywords = ['真人', '客服', '人工', '轉接', '協助', 'human', 'help', 'support', 'agent']
        if any(keyword in message_lower for keyword in handover_keywords):
            return 'handover'

        # FAQ keywords
        faq_keywords = ['政策', '退換貨', '保固', '發票', '運費', '付款', 'policy', 'return', 'warranty', 'payment']
        if any(keyword in message_lower for keyword in faq_keywords):
            return 'faq'

        # Default to general for broader questions
        return 'general'