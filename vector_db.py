import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import hashlib
import json
from data_processor import DataProcessor, KnowledgeItem, Product

class VectorDB:
    def __init__(self, db_path: str = "chroma_db"):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.knowledge_collection = None
        self.products_collection = None

    def setup_collections(self):
        """Initialize ChromaDB collections for knowledge and products"""
        # Knowledge base collection
        try:
            self.knowledge_collection = self.client.get_collection("knowledge_base")
        except:
            self.knowledge_collection = self.client.create_collection(
                name="knowledge_base",
                metadata={"description": "JTCG knowledge base for FAQ and support"}
            )

        # Products collection
        try:
            self.products_collection = self.client.get_collection("products")
        except:
            self.products_collection = self.client.create_collection(
                name="products",
                metadata={"description": "JTCG product catalog"}
            )

    def _create_knowledge_document(self, item: KnowledgeItem) -> str:
        """Create a searchable document from knowledge item"""
        doc_parts = [
            f"Title: {item.title}",
            f"Content: {item.content}",
            f"Tags: {', '.join(item.tags)}" if item.tags else ""
        ]
        return " ".join(part for part in doc_parts if part)

    def _create_product_document(self, product: Product) -> str:
        """Create a searchable document from product"""
        doc_parts = [
            f"Name: {product.name}",
            f"SKU: {product.sku}",
            f"Type: {product.arm_type}",
            f"Size: {product.size_max_inch} inch" if product.size_max_inch else "",
            f"VESA: {', '.join(product.vesa_options)}" if product.vesa_options else "",
            f"Weight: {product.weight_per_arm_kg} kg" if product.weight_per_arm_kg else "",
            f"Desk: {product.desk_thickness_mm} mm" if product.desk_thickness_mm else "",
            f"Notes: {product.compatibility_notes}" if product.compatibility_notes else "",
            f"Includes: {', '.join(product.includes)}" if product.includes else ""
        ]
        return " ".join(part for part in doc_parts if part)

    def populate_knowledge_base(self, knowledge_items: List[KnowledgeItem]):
        """Populate the knowledge base collection"""
        if not self.knowledge_collection:
            self.setup_collections()

        # Check if already populated
        existing_count = self.knowledge_collection.count()
        if existing_count > 0:
            print(f"Knowledge base already has {existing_count} items. Skipping population.")
            return

        documents = []
        metadatas = []
        ids = []

        for item in knowledge_items:
            doc = self._create_knowledge_document(item)
            documents.append(doc)

            metadata = {
                "id": item.id,
                "title": item.title,
                "url_label": item.url_label,
                "url_href": item.url_href,
                "image_url": item.image_url,
                "tags": json.dumps(item.tags),
                "type": "knowledge"
            }
            metadatas.append(metadata)
            ids.append(item.id)

        self.knowledge_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} knowledge items to vector database")

    def populate_products(self, products: List[Product]):
        """Populate the products collection"""
        if not self.products_collection:
            self.setup_collections()

        # Check if already populated
        existing_count = self.products_collection.count()
        if existing_count > 0:
            print(f"Products already has {existing_count} items. Skipping population.")
            return

        documents = []
        metadatas = []
        ids = []

        for product in products:
            doc = self._create_product_document(product)
            documents.append(doc)

            metadata = {
                "sku": product.sku,
                "name": product.name,
                "arm_type": product.arm_type,
                "size_max_inch": product.size_max_inch,
                "vesa_options": json.dumps(product.vesa_options),
                "weight_per_arm_kg": product.weight_per_arm_kg,
                "desk_thickness_mm": product.desk_thickness_mm,
                "compatibility_notes": product.compatibility_notes,
                "url": product.url,
                "image_url": product.image_url,
                "includes": json.dumps(product.includes),
                "type": "product"
            }
            metadatas.append(metadata)
            ids.append(product.sku)

        self.products_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} products to vector database")

    def search_knowledge(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant FAQ items"""
        if not self.knowledge_collection:
            self.setup_collections()

        results = self.knowledge_collection.query(
            query_texts=[query],
            n_results=n_results
        )

        formatted_results = []
        if results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                result = {
                    "id": metadata["id"],
                    "title": metadata["title"],
                    "url_label": metadata["url_label"],
                    "url_href": metadata["url_href"],
                    "image_url": metadata["image_url"],
                    "tags": json.loads(metadata["tags"]),
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "document": results["documents"][0][i] if results["documents"] else ""
                }
                formatted_results.append(result)

        return formatted_results

    def search_products(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search products for relevant items"""
        if not self.products_collection:
            self.setup_collections()

        results = self.products_collection.query(
            query_texts=[query],
            n_results=n_results
        )

        formatted_results = []
        if results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                result = {
                    "sku": metadata["sku"],
                    "name": metadata["name"],
                    "arm_type": metadata["arm_type"],
                    "size_max_inch": metadata["size_max_inch"],
                    "vesa_options": json.loads(metadata["vesa_options"]),
                    "weight_per_arm_kg": metadata["weight_per_arm_kg"],
                    "desk_thickness_mm": metadata["desk_thickness_mm"],
                    "compatibility_notes": metadata["compatibility_notes"],
                    "url": metadata["url"],
                    "image_url": metadata["image_url"],
                    "includes": json.loads(metadata["includes"]),
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "document": results["documents"][0][i] if results["documents"] else ""
                }
                formatted_results.append(result)

        return formatted_results

    def initialize_with_data(self, data_processor: DataProcessor):
        """Initialize vector database with data from processor"""
        self.setup_collections()
        self.populate_knowledge_base(data_processor.knowledge_base)
        self.populate_products(data_processor.products)

    def reset_database(self):
        """Reset the vector database (delete all collections)"""
        try:
            self.client.delete_collection("knowledge_base")
        except:
            pass
        try:
            self.client.delete_collection("products")
        except:
            pass
        self.knowledge_collection = None
        self.products_collection = None