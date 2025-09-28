#!/usr/bin/env python3

from data_processor import DataProcessor

def test_data_processor():
    """Test the data processor functionality"""
    processor = DataProcessor()

    # Test loading all data
    try:
        processor.load_all_data(
            knowledge_csv="ref_data/ai-eng-test-sample-knowledges.csv",
            products_csv="ref_data/ai-eng-test-sample-products.csv",
            orders_json="ref_data/orders.json",
            conversations_json="ref_data/ai-eng-test-sample-conversations.json"
        )

        print("Data loading successful!")

        # Test knowledge base
        print(f"Knowledge items loaded: {len(processor.knowledge_base)}")
        if processor.knowledge_base:
            sample_kb = processor.knowledge_base[0]
            print(f"   Sample: {sample_kb.title} - {sample_kb.content[:50]}...")

        # Test products
        print(f"Products loaded: {len(processor.products)}")
        if processor.products:
            sample_product = processor.products[0]
            print(f"   Sample: {sample_product.name} ({sample_product.sku})")

        # Test orders
        print(f"Order users loaded: {len(processor.orders_db)}")
        total_orders = sum(len(orders) for orders in processor.orders_db.values())
        print(f"   Total orders: {total_orders}")

        # Test conversations
        print(f"Conversations loaded: {len(processor.conversations)}")

        # Test lookup functions
        print("\nTesting lookup functions:")

        # Test knowledge lookup
        kb_item = processor.get_knowledge_by_id("FAQ-RET-001")
        if kb_item:
            print(f"   Knowledge lookup: {kb_item.title}")

        # Test product lookup
        product = processor.get_product_by_sku("JTCG-ARM-DUAL-PRO-32")
        if product:
            print(f"   Product lookup: {product.name}")

        # Test user orders lookup
        user_orders = processor.get_orders_by_user_id("u_123456")
        print(f"   User orders: {len(user_orders)} orders for u_123456")

        # Test order lookup by ID
        order = processor.get_order_by_id("JTCG-202508-10001")
        if order:
            print(f"   Order lookup: {order.order_id} - {order.status}")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_data_processor()