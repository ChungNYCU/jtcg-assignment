#!/usr/bin/env python3

from data_processor import DataProcessor
from vector_db import VectorDB

def test_vector_db():
    """Test the vector database functionality"""
    print("Initializing data processor...")
    processor = DataProcessor()
    processor.load_all_data(
        knowledge_csv="ref_data/ai-eng-test-sample-knowledges.csv",
        products_csv="ref_data/ai-eng-test-sample-products.csv",
        orders_json="ref_data/orders.json",
        conversations_json="ref_data/ai-eng-test-sample-conversations.json"
    )

    print("Setting up vector database...")
    vector_db = VectorDB()
    vector_db.initialize_with_data(processor)

    print("\nTesting knowledge search...")
    # Test FAQ search
    kb_results = vector_db.search_knowledge("退換貨政策", n_results=2)
    print(f"Found {len(kb_results)} knowledge results for '退換貨政策':")
    for result in kb_results:
        print(f"  - {result['title']} (distance: {result['distance']:.3f})")
        if result['url_href']:
            print(f"    Link: {result['url_href']}")

    print("\nTesting product search...")
    # Test product search
    product_results = vector_db.search_products("雙螢幕臂架", n_results=3)
    print(f"Found {len(product_results)} product results for '雙螢幕臂架':")
    for result in product_results:
        print(f"  - {result['name']} ({result['sku']}) (distance: {result['distance']:.3f})")
        if result['url']:
            print(f"    Link: {result['url']}")

    print("\nTesting English search...")
    # Test English search
    en_results = vector_db.search_products("dual monitor arm", n_results=2)
    print(f"Found {len(en_results)} product results for 'dual monitor arm':")
    for result in en_results:
        print(f"  - {result['name']} ({result['sku']}) (distance: {result['distance']:.3f})")

    print("\nTesting specification search...")
    # Test spec-based search
    spec_results = vector_db.search_products("32 inch VESA 100x100", n_results=3)
    print(f"Found {len(spec_results)} product results for '32 inch VESA 100x100':")
    for result in spec_results:
        print(f"  - {result['name']} ({result['sku']}) (distance: {result['distance']:.3f})")
        print(f"    VESA: {result['vesa_options']}, Max size: {result['size_max_inch']}\"")

    print("\nVector database tests completed!")

if __name__ == "__main__":
    test_vector_db()