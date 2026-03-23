# Sales Profit Margin Prediction API

Production REST API predicting profit margin — R² 0.9436 on Superstore dataset.

## Live
| | URL |
|--|--|
| API | https://sales-profit-api.onrender.com |
| Docs | https://sales-profit-api.onrender.com/docs |
| Health | https://sales-profit-api.onrender.com/health |

## Screenshots
![API Docs](screenshots/api_docs.png)
![Predict Response](screenshots/predict_response.png)
![What-If Response](screenshots/whatif_response.png)

## Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /predict | Predict profit margin for an order |
| POST | /predict/batch | Batch order predictions |
| POST | /whatif | Compare margin at different discount levels |

## The /whatif Endpoint
Unique to this API — quantifies the exact cost of every discount decision:
- Send an order with current discount
- Get back margin at current, -10%, and 0% discount
- Instantly shows how much margin is being sacrificed

## Run Locally
```bash
git clone https://github.com/KV0217/Sales-Profit-API.git
cd Sales-Profit-API
pip install -r requirements.txt
uvicorn main:app --reload
```

## Tech Stack
FastAPI · Gradient Boosting · Scikit-learn · Docker · Render

## Related
- Analysis notebook: [Retail-Sales-Revenue](https://github.com/KV0217/retail-sales-revenue)
