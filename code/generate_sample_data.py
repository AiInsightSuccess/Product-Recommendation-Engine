"""Create deterministic sample CSV files for the recommendation workbook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 20260419


CATEGORY_PRODUCTS = {
    "Kitchen": [
        "Stainless Saucepan",
        "Bamboo Chopping Board",
        "Ceramic Dinner Plate Set",
        "Glass Food Storage Set",
        "Compact Air Fryer",
        "Digital Kitchen Scale",
        "Insulated Lunch Box",
        "Reusable Baking Mat",
        "Manual Coffee Grinder",
        "Countertop Blender",
    ],
    "Home": [
        "Cotton Towel Set",
        "Soft Throw Blanket",
        "Storage Basket",
        "Table Lamp",
        "Scent-Free Candle Set",
        "Wall Shelf",
        "Laundry Hamper",
        "Door Mat",
        "Desk Organizer",
        "Memory Foam Pillow",
    ],
    "Personal Care": [
        "Electric Toothbrush",
        "Travel Toiletry Bag",
        "Hair Dryer",
        "Reusable Face Pads",
        "Hand Care Set",
        "Shower Caddy",
        "Bath Brush",
        "Sleep Mask",
        "Compact Mirror",
        "Grooming Kit",
    ],
    "Office Tech": [
        "Wireless Keyboard",
        "Ergonomic Mouse",
        "Laptop Stand",
        "USB-C Hub",
        "Noise Reducing Headset",
        "Desk Webcam",
        "Portable Monitor Stand",
        "Cable Management Kit",
        "Rechargeable Desk Light",
        "Notebook Sleeve",
    ],
    "Fitness": [
        "Yoga Mat",
        "Resistance Band Set",
        "Foam Roller",
        "Adjustable Dumbbell",
        "Sports Water Bottle",
        "Exercise Ball",
        "Ankle Weights",
        "Skipping Rope",
        "Cooling Towel",
        "Gym Bag",
    ],
    "Outdoor": [
        "Lightweight Backpack",
        "Picnic Blanket",
        "Camping Lantern",
        "Thermal Flask",
        "Waterproof Phone Pouch",
        "Foldable Chair",
        "Garden Gloves",
        "Solar Path Light",
        "Travel Umbrella",
        "Compact First Aid Kit",
    ],
    "Pet Supplies": [
        "Pet Food Bowl",
        "Adjustable Pet Lead",
        "Pet Grooming Brush",
        "Washable Pet Bed",
        "Treat Storage Jar",
        "Pet Travel Carrier",
        "Interactive Pet Toy",
        "Pet Waste Bag Holder",
        "Pet Cooling Mat",
        "Pet Blanket",
    ],
}


PRICE_RANGES = {
    "Kitchen": (8, 115),
    "Home": (7, 95),
    "Personal Care": (5, 85),
    "Office Tech": (9, 160),
    "Fitness": (6, 130),
    "Outdoor": (7, 120),
    "Pet Supplies": (5, 105),
}


SEGMENT_PREFERENCES = {
    "practical_home": ["Home", "Kitchen", "Personal Care"],
    "workspace_upgrader": ["Office Tech", "Home", "Personal Care"],
    "active_lifestyle": ["Fitness", "Outdoor", "Personal Care"],
    "outdoor_planner": ["Outdoor", "Fitness", "Kitchen"],
    "pet_household": ["Pet Supplies", "Home", "Outdoor"],
}


def choose_margin_band(price: float) -> str:
    if price < 20:
        return "low"
    if price < 70:
        return "medium"
    return "high"


def build_catalog(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    item_number = 1

    for category, product_names in CATEGORY_PRODUCTS.items():
        low, high = PRICE_RANGES[category]
        for product_name in product_names:
            price = round(float(rng.uniform(low, high)), 2)
            rows.append(
                {
                    "item_id": f"ITEM{item_number:03d}",
                    "product_name": product_name,
                    "category": category,
                    "price": price,
                    "margin_band": choose_margin_band(price),
                    "average_rating": round(float(rng.uniform(3.6, 4.9)), 1),
                    "stock_status": rng.choice(
                        ["in_stock", "in_stock", "in_stock", "limited_stock"], p=[0.55, 0.2, 0.15, 0.1]
                    ),
                    "product_age_days": int(rng.integers(20, 540)),
                }
            )
            item_number += 1

    return pd.DataFrame(rows)


def build_customers(rng: np.random.Generator, n_customers: int = 140) -> pd.DataFrame:
    rows = []
    segments = list(SEGMENT_PREFERENCES)

    for customer_number in range(1, n_customers + 1):
        segment = str(rng.choice(segments))
        rows.append(
            {
                "user_id": f"CUST{customer_number:03d}",
                "customer_segment": segment,
                "preferred_price_band": str(rng.choice(["budget", "mid_range", "premium"], p=[0.35, 0.45, 0.2])),
                "loyalty_tier": str(rng.choice(["new", "standard", "frequent"], p=[0.25, 0.55, 0.2])),
            }
        )

    return pd.DataFrame(rows)


def item_affinity(
    row: pd.Series,
    preferred_categories: list[str],
    price_band: str,
    rng: np.random.Generator,
) -> float:
    score = 0.2 + 0.8 * (row["category"] in preferred_categories)
    score += 0.25 * (preferred_categories[0] == row["category"])
    score += 0.08 * (row["average_rating"] - 3.5)

    if price_band == "budget":
        score += 0.35 * (row["price"] <= 30) - 0.2 * (row["price"] >= 90)
    elif price_band == "premium":
        score += 0.2 * (row["price"] >= 70)
    else:
        score += 0.2 * (30 < row["price"] < 90)

    score += float(rng.normal(0, 0.08))
    return max(score, 0.01)


def build_interactions(
    catalog: pd.DataFrame,
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    start_time = pd.Timestamp("2026-01-01 09:00:00")
    session_number = 1

    for _, customer in customers.iterrows():
        preferred_categories = SEGMENT_PREFERENCES[customer["customer_segment"]]
        affinities = catalog.apply(
            item_affinity,
            axis=1,
            preferred_categories=preferred_categories,
            price_band=customer["preferred_price_band"],
            rng=rng,
        ).to_numpy()
        probabilities = affinities / affinities.sum()
        n_items = int(rng.integers(10, 24))
        chosen_indexes = rng.choice(catalog.index, size=n_items, replace=False, p=probabilities)

        for item_index in chosen_indexes:
            product = catalog.loc[item_index]
            affinity = affinities[item_index]
            event_time = start_time + pd.Timedelta(days=int(rng.integers(0, 100))) + pd.Timedelta(
                minutes=int(rng.integers(0, 900))
            )
            session_id = f"SESSION{session_number:05d}"
            session_number += 1

            events = ["view"]
            if rng.random() < min(0.85, 0.2 + affinity / 2.2):
                events.append("click")
            if rng.random() < min(0.65, 0.08 + affinity / 3.0):
                events.append("add_to_cart")
            if "add_to_cart" in events and rng.random() < min(0.55, 0.08 + affinity / 3.5):
                events.append("purchase")
            if "purchase" in events and rng.random() < 0.18:
                events.append("review")

            for step, event_type in enumerate(events):
                quantity = int(rng.integers(1, 4)) if event_type == "purchase" else 1
                rows.append(
                    {
                        "user_id": customer["user_id"],
                        "item_id": product["item_id"],
                        "event_type": event_type,
                        "event_strength": {
                            "view": 1.0,
                            "click": 2.0,
                            "add_to_cart": 4.0,
                            "purchase": 8.0,
                            "review": 10.0,
                        }[event_type],
                        "quantity": quantity,
                        "unit_price": product["price"],
                        "interaction_datetime": (event_time + pd.Timedelta(minutes=step * 3)).isoformat(),
                        "session_id": session_id,
                    }
                )

    return pd.DataFrame(rows).sort_values(["interaction_datetime", "user_id", "item_id"])


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    base_dir = Path(__file__).resolve().parents[1]

    catalog = build_catalog(rng)
    customers = build_customers(rng)
    interactions = build_interactions(catalog, customers, rng)

    catalog.to_csv(base_dir / "product_catalog.csv", index=False)
    customers.to_csv(base_dir / "customer_segments.csv", index=False)
    interactions.to_csv(base_dir / "customer_product_interactions.csv", index=False)

    print(f"Created {len(catalog)} products")
    print(f"Created {len(customers)} customers")
    print(f"Created {len(interactions)} interaction events")


if __name__ == "__main__":
    main()
