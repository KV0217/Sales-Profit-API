from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, List, Dict
import joblib
import pandas as pd

app = FastAPI(
    title="Sales Profit Margin Prediction API",
    description="Predicts profit margin using Gradient Boosting trained on Superstore dataset — R² 0.9436",
    version="1.0.0"
)

model = joblib.load("sales_model.joblib")

FEATURE_COLS = [
    'Quantity', 'Discount', 'Discount_x_Quantity', 'High_Discount',
    'Days_to_Ship', 'Category_enc', 'Sub_Category_enc', 'Region_enc',
    'Segment_enc', 'Ship_Mode_enc', 'Order_Month', 'Order_Quarter',
    'Is_Q4', 'Order_Year'
]

CATEGORY_MAP     = {'Furniture': 0, 'Office Supplies': 1, 'Technology': 2}
SUB_CATEGORY_MAP = {
    'Accessories': 0, 'Appliances': 1, 'Art': 2, 'Binders': 3,
    'Bookcases': 4, 'Chairs': 5, 'Copiers': 6, 'Envelopes': 7,
    'Fasteners': 8, 'Furnishings': 9, 'Labels': 10, 'Machines': 11,
    'Paper': 12, 'Phones': 13, 'Storage': 14, 'Supplies': 15, 'Tables': 16
}
REGION_MAP    = {'Central': 0, 'East': 1, 'South': 2, 'West': 3}
SEGMENT_MAP   = {'Consumer': 0, 'Corporate': 1, 'Home Office': 2}
SHIP_MODE_MAP = {'First Class': 0, 'Same Day': 1, 'Second Class': 2, 'Standard Class': 3}


class OrderData(BaseModel):
    Quantity:     int
    Discount:     float
    Days_to_Ship: int
    Category:     Literal["Furniture", "Office Supplies", "Technology"]
    Sub_Category: Literal[
        "Accessories", "Appliances", "Art", "Binders", "Bookcases",
        "Chairs", "Copiers", "Envelopes", "Fasteners", "Furnishings",
        "Labels", "Machines", "Paper", "Phones", "Storage", "Supplies", "Tables"
    ]
    Region:    Literal["Central", "East", "South", "West"]
    Segment:   Literal["Consumer", "Corporate", "Home Office"]
    Ship_Mode: Literal["First Class", "Same Day", "Second Class", "Standard Class"]
    Order_Month:   int  # 1-12
    Order_Quarter: int  # 1-4
    Order_Year:    int  # e.g. 2024

    class Config:
        json_schema_extra = {
            "example": {
                "Quantity": 3, "Discount": 0.2, "Days_to_Ship": 3,
                "Category": "Technology", "Sub_Category": "Phones",
                "Region": "West", "Segment": "Consumer",
                "Ship_Mode": "Standard Class",
                "Order_Month": 11, "Order_Quarter": 4, "Order_Year": 2024
            }
        }


def get_insights(order: OrderData, margin: float) -> List[Dict]:
    insights = []

    if order.Discount > 0.3:
        insights.append({
            "area": "High Discount",
            "issue": f"{order.Discount*100:.0f}% discount is destroying margin",
            "impact": "High",
            "recommendation": "Cap discounts at 20% — beyond 30% consistently generates losses",
            "estimated_margin_improvement": "15-25 percentage points"
        })

    if order.Sub_Category in ["Tables", "Bookcases"]:
        insights.append({
            "area": "Loss-Making Sub-Category",
            "issue": f"{order.Sub_Category} is a historically loss-making sub-category",
            "impact": "High",
            "recommendation": "Review pricing strategy or discontinue heavy discounting on this category",
            "estimated_margin_improvement": "10-20 percentage points"
        })

    if order.Discount > 0.2 and order.Quantity > 5:
        insights.append({
            "area": "Bulk Discount Abuse",
            "issue": "High quantity + high discount — margin erosion pattern",
            "impact": "Medium",
            "recommendation": "Set minimum margin floor for bulk orders — use tiered discount caps",
            "estimated_margin_improvement": "8-15 percentage points"
        })

    if order.Ship_Mode == "Same Day":
        insights.append({
            "area": "Shipping Cost",
            "issue": "Same Day shipping significantly reduces profit margin",
            "impact": "Medium",
            "recommendation": "Offer Same Day only for Technology/high-margin items — not Furniture",
            "estimated_margin_improvement": "5-10 percentage points"
        })

    if order.Order_Quarter == 4 and order.Discount > 0.2:
        insights.append({
            "area": "Q4 Discount Strategy",
            "issue": "Heavy discounting in Q4 despite high natural demand",
            "impact": "Medium",
            "recommendation": "Reduce Q4 discounts — demand is naturally high, discounts are unnecessary",
            "estimated_margin_improvement": "5-12 percentage points"
        })

    if not insights:
        insights.append({
            "area": "Overall",
            "issue": "No major margin risk factors detected",
            "impact": "Low",
            "recommendation": "Maintain current pricing and discount strategy",
            "estimated_margin_improvement": "0%"
        })

    return insights


def encode(order: OrderData) -> pd.DataFrame:
    row = {
        'Quantity':             order.Quantity,
        'Discount':             order.Discount,
        'Discount_x_Quantity':  order.Discount * order.Quantity,
        'High_Discount':        int(order.Discount > 0.2),
        'Days_to_Ship':         order.Days_to_Ship,
        'Category_enc':         CATEGORY_MAP[order.Category],
        'Sub_Category_enc':     SUB_CATEGORY_MAP[order.Sub_Category],
        'Region_enc':           REGION_MAP[order.Region],
        'Segment_enc':          SEGMENT_MAP[order.Segment],
        'Ship_Mode_enc':        SHIP_MODE_MAP[order.Ship_Mode],
        'Order_Month':          order.Order_Month,
        'Order_Quarter':        order.Order_Quarter,
        'Is_Q4':                int(order.Order_Quarter == 4),
        'Order_Year':           order.Order_Year
    }
    return pd.DataFrame([row])[FEATURE_COLS]


@app.get("/")
def root():
    return {
        "message": "Sales Profit Margin Prediction API is live",
        "model": "Gradient Boosting — R² 0.9436 on Superstore dataset",
        "docs": "/docs",
        "version": "1.0"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(order: OrderData):
    try:
        input_df = encode(order)
        margin   = round(float(model.predict(input_df)[0]), 4)
        margin_pct = round(margin * 100, 2)

        if margin < 0:
            status = "loss"
            recommendation = "This order will generate a loss — review discount and sub-category"
        elif margin < 0.1:
            status = "low_margin"
            recommendation = "Low margin — consider reducing discount or upgrading ship mode"
        elif margin < 0.3:
            status = "acceptable"
            recommendation = "Acceptable margin — monitor discount levels"
        else:
            status = "healthy"
            recommendation = "Healthy margin — good order profile"

        insights = get_insights(order, margin)

        return {
            "predicted_profit_margin": margin,
            "profit_margin_pct": f"{margin_pct}%",
            "status": status,
            "recommendation": recommendation,
            "high_discount_flag": bool(order.Discount > 0.2),
            "is_loss_making_subcategory": order.Sub_Category in ["Tables", "Bookcases"],
            "insights": insights,
            "summary": f"{len(insights)} insight(s) identified"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(orders: list[OrderData]):
    results = [predict(o) for o in orders]
    return {"predictions": results, "count": len(results)}


@app.post("/whatif")
def whatif(order: OrderData):
    """Compare current discount vs reduced discount scenarios"""
    try:
        base        = predict(order)
        order_low   = order.model_copy(update={"Discount": max(0, order.Discount - 0.1)})
        order_zero  = order.model_copy(update={"Discount": 0.0})
        low_disc    = predict(order_low)
        zero_disc   = predict(order_zero)

        return {
            "current_discount":   f"{order.Discount*100:.0f}%",
            "current_margin":     base["profit_margin_pct"],
            "reduce_10pct_discount": {
                "discount": f"{order_low.Discount*100:.0f}%",
                "margin":   low_disc["profit_margin_pct"],
                "improvement": f"{round((low_disc['predicted_profit_margin'] - base['predicted_profit_margin'])*100, 2)}pp"
            },
            "zero_discount": {
                "discount": "0%",
                "margin":   zero_disc["profit_margin_pct"],
                "improvement": f"{round((zero_disc['predicted_profit_margin'] - base['predicted_profit_margin'])*100, 2)}pp"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    