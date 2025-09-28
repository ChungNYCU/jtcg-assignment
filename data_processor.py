import pandas as pd
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path

class KnowledgeItem(BaseModel):
    id: str
    title: str
    content: str
    url_label: str = ""
    url_href: str = ""
    image_url: str = ""
    tags: List[str] = Field(default_factory=list)

class Product(BaseModel):
    sku: str
    name: str
    arm_type: str = ""
    size_max_inch: str = ""
    vesa_options: List[str] = Field(default_factory=list)
    weight_per_arm_kg: str = ""
    desk_thickness_mm: str = ""
    rotation: str = ""
    tilt: str = ""
    swivel: str = ""
    usb_hub: bool = False
    compatibility_notes: str = ""
    url: str = ""
    image_url: str = ""
    reach_mm: str = ""
    tray_size_inch: str = ""
    includes: List[str] = Field(default_factory=list)

class Order(BaseModel):
    order_id: str
    placed_at: str
    status: str
    carrier: str = ""
    tracking: str = ""
    eta: str = ""
    items: List[Dict[str, Any]] = Field(default_factory=list)
    shipping_address: str = ""
    contact_phone: str = ""
    order_url: str = ""

class DataProcessor:
    def __init__(self):
        self.knowledge_base: List[KnowledgeItem] = []
        self.products: List[Product] = []
        self.orders_db: Dict[str, List[Order]] = {}
        self.conversations: List[List[Dict]] = []

    def load_knowledge_base(self, csv_path: str) -> List[KnowledgeItem]:
        """Load and parse the knowledge base CSV file"""
        df = pd.read_csv(csv_path)

        knowledge_items = []
        for _, row in df.iterrows():
            # Handle tags - extract from columns that start with 'tags/'
            tags = []
            for col in df.columns:
                if col.startswith('tags/') and pd.notna(row[col]):
                    tags.append(row[col])

            # Handle URLs
            url_label = row.get('urls/0/label', '') if pd.notna(row.get('urls/0/label')) else ''
            url_href = row.get('urls/0/href', '') if pd.notna(row.get('urls/0/href')) else ''

            # Handle images
            image_url = row.get('images/0', '') if pd.notna(row.get('images/0')) else ''

            item = KnowledgeItem(
                id=row['id'],
                title=row['title'],
                content=row['content'],
                url_label=url_label,
                url_href=url_href,
                image_url=image_url,
                tags=tags
            )
            knowledge_items.append(item)

        self.knowledge_base = knowledge_items
        return knowledge_items

    def load_products(self, csv_path: str) -> List[Product]:
        """Load and parse the products CSV file"""
        df = pd.read_csv(csv_path)

        products = []
        for _, row in df.iterrows():
            # Handle VESA options
            vesa_options = []
            for col in df.columns:
                if col.startswith('specs/vesa/') and pd.notna(row[col]):
                    vesa_options.append(row[col])

            # Handle includes
            includes = []
            for col in df.columns:
                if col.startswith('specs/includes/') and pd.notna(row[col]):
                    includes.append(row[col])

            # Helper function to get safe value
            def safe_get(column, default=""):
                value = row.get(column, default) if pd.notna(row.get(column)) else default
                return str(value) if value is not None else default

            product = Product(
                sku=row['sku'],
                name=row['name'],
                arm_type=safe_get('specs/arm_type'),
                size_max_inch=str(safe_get('specs/size_max_inch')),
                vesa_options=vesa_options,
                weight_per_arm_kg=safe_get('specs/weight_per_arm_kg'),
                desk_thickness_mm=safe_get('specs/desk_thickness_mm'),
                rotation=safe_get('specs/rotation'),
                tilt=safe_get('specs/tilt'),
                swivel=safe_get('specs/swivel'),
                usb_hub=bool(safe_get('specs/usb_hub')),
                compatibility_notes=safe_get('compatibility_notes'),
                url=safe_get('url'),
                image_url=safe_get('images/0'),
                reach_mm=safe_get('specs/reach_mm'),
                tray_size_inch=safe_get('specs/tray_size_inch'),
                includes=includes
            )
            products.append(product)

        self.products = products
        return products

    def load_orders_from_json(self, json_path: str) -> Dict[str, List[Order]]:
        """Load orders from a JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            orders_data = json.load(f)

        # Parse orders into Order objects
        parsed_orders = {}
        orders_db = orders_data.get("orders_db", orders_data)

        for user_id, user_data in orders_db.items():
            user_orders = []
            for order_data in user_data["orders"]:
                order = Order(
                    order_id=order_data["order_id"],
                    placed_at=order_data["placed_at"],
                    status=order_data["status"],
                    carrier=order_data.get("carrier", "") or "",
                    tracking=order_data.get("tracking", "") or "",
                    eta=order_data.get("eta", "") or "",
                    items=order_data["items"],
                    shipping_address=order_data["shipping_address"],
                    contact_phone=order_data["contact_phone"],
                    order_url=order_data["order_url"]
                )
                user_orders.append(order)
            parsed_orders[user_id] = user_orders

        self.orders_db = parsed_orders
        return parsed_orders

    def load_conversations(self, json_path: str) -> List[List[Dict]]:
        """Load test conversations from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        self.conversations = conversations
        return conversations

    def load_all_data(self, knowledge_csv: str, products_csv: str, orders_json: str, conversations_json: str):
        """Load all data sources from specified paths"""
        self.load_knowledge_base(knowledge_csv)
        self.load_products(products_csv)
        self.load_orders_from_json(orders_json)
        self.load_conversations(conversations_json)

    def get_knowledge_by_id(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """Get knowledge item by ID"""
        for item in self.knowledge_base:
            if item.id == knowledge_id:
                return item
        return None

    def get_product_by_sku(self, sku: str) -> Optional[Product]:
        """Get product by SKU"""
        for product in self.products:
            if product.sku == sku:
                return product
        return None

    def get_orders_by_user_id(self, user_id: str) -> List[Order]:
        """Get orders for a specific user"""
        return self.orders_db.get(user_id, [])

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Get specific order by order ID"""
        for user_orders in self.orders_db.values():
            for order in user_orders:
                if order.order_id == order_id:
                    return order
        return None