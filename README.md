# Retail Sales ML Platform

Full-stack machine learning application for retail sales forecasting with FastAPI and React.

[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://github.com/KapProgr/retail-sales-api)
[![Frontend](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge&logo=react)](https://github.com/KapProgr/retail-sales-frontend)
[![Live Demo](https://img.shields.io/badge/Live-Demo-blueviolet?style=for-the-badge)](https://retail-sales.vercel.app)

---

## Quick Links

| Purpose | Link | Description |
|---|---|---|
| Live App | [retail-sales-ml.vercel.app](https://retail-sales.vercel.app) | Production web application |
| API Docs | [retail-sales-api.onrender.com/docs](https://retail-sales-api.onrender.com/docs) | Interactive API documentation |
| Backend Repo | [retail-sales-api](https://github.com/KapProgr/retail-sales-api) | Python FastAPI and ML models |
| Frontend Repo | [retail-sales-frontend](https://github.com/KapProgr/retail-sales-frontend) | React, Vite, Tailwind |

---

## Project Overview

This is a production-ready full-stack ML application that predicts retail sales using multiple machine learning models.

### Key Capabilities

- Upload CSV files with historical sales data
- Generate synthetic data for testing (365 days)
- Train 3 ML models simultaneously
- Interactive visualizations of predictions
- Compare model performance (MAE, RMSE, R2, MAPE)
- 30-day future forecasts
- Feature importance analysis
- Export results to CSV

---

## Architecture

```text
User Browser (Vercel frontend URL)
	-> HTTP/JSON
Frontend (Vercel)
	React 18 + Vite + Tailwind + Recharts
	-> REST API
Backend (Render)
	FastAPI + Python
	Endpoints: /upload, /predict, /sample, /health
	-> scikit-learn
ML Pipeline
	Feature engineering + model training + evaluation
```

---

## Repository Structure

### Backend Repository

[retail-sales-api](https://github.com/KapProgr/retail-sales-api)

```text
retail-sales-api/
	main.py
	requirements.txt
	README.md
```

Tech stack:

- FastAPI
- scikit-learn
- pandas
- numpy

Endpoints:

- GET /
- GET /health
- GET /sample
- POST /upload
- POST /predict

---

### Frontend Repository

[retail-sales-frontend](https://github.com/KapProgr/retail-sales-frontend)

```text
retail-sales-frontend/
	src/
		App.jsx
		index.css
		main.jsx
	public/
	package.json
	vite.config.js
	tailwind.config.js
```

Tech stack:

- React 18
- Vite 5
- Tailwind CSS
- Recharts
- Lucide React

---

## Getting Started

### Prerequisites

- Backend: Python 3.9+ and pip
- Frontend: Node.js 18+ and npm

### Local Development

#### 1. Clone Repositories

```bash
# Backend
git clone https://github.com/KapProgr/retail-sales-api.git
cd retail-sales-api

# Frontend
git clone https://github.com/KapProgr/retail-sales-frontend.git
cd retail-sales-frontend
```

#### 2. Setup Backend

```bash
cd retail-sales-api
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend local URLs:

- http://localhost:8000
- http://localhost:8000/docs

#### 3. Setup Frontend

```bash
cd retail-sales-frontend
npm install
npm run dev
```

Frontend local URL:

- http://localhost:5173

---

## Machine Learning Models

### Models Implemented

| Model | Type | Best For | Example Performance |
|---|---|---|---|
| Linear Regression | Baseline | Fast and interpretable | MAE around 192 |
| Random Forest | Ensemble | Robust and stable | MAE around 195 |
| Gradient Boosting | Ensemble | High predictive power | MAE around 206 |

### Features Engineered

1. Lag features: sales from 1, 7, 14, and 30 days ago
2. Rolling statistics: mean and std over 7, 14, 30-day windows
3. Time features: day of week, month, quarter, weekend, holiday
4. Cyclical features: sin/cos seasonality transforms

### Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 (R-squared)
- MAPE (Mean Absolute Percentage Error)

---

## Performance

Example output on synthetic retail data:

```text
Model: Linear Regression
Accuracy: 87.6%
MAE: $191.76
RMSE: $251.41
R2: 0.4879
MAPE: 12.45%
```

---

## Deployment

### Backend (Render)

1. Push backend repo to GitHub
2. Connect repo to Render
3. Configure:
	 - Build Command: `pip install -r requirements.txt`
	 - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy

### Frontend (Vercel)

1. Push frontend repo to GitHub
2. Import repo in Vercel
3. Configure:
	 - Framework Preset: Vite
	 - Build Command: `npm run build`
	 - Output Directory: `dist`
4. Deploy

---

## Testing

### Backend

```bash
curl http://localhost:8000/health
curl http://localhost:8000/sample
```

### Frontend

```bash
npm run dev
npm run build
npm run preview
```

---

## Data Format

```csv
date,sales
2023-01-01,1200.50
2023-01-02,1350.75
2023-01-03,1180.25
```

Requirements:

- Minimum 100 records
- Date format: YYYY-MM-DD
- Sales values: positive numbers

---

## Security

- No API keys required
- CORS configured
- Input validation on backend
- No sensitive data storage
- HTTPS in production

---

## Roadmap

- Authentication and user accounts
- Database for saved predictions
- Additional models (LSTM, Prophet)
- Email reports
- Mobile app
- Real-time predictions
- A/B testing workflows
- Custom model training options

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Open a pull request

---

## License

MIT License

---

## Author

KapProgr

- GitHub: [@KapProgr](https://github.com/KapProgr)
- Backend: [retail-sales-api](https://github.com/KapProgr/retail-sales-api)
- Frontend: [retail-sales-frontend](https://github.com/KapProgr/retail-sales-frontend)

---

## Acknowledgments

- FastAPI
- React
- scikit-learn
- Tailwind CSS
- Recharts
- Render
- Vercel

---

## Support

- Issues: open a GitHub issue
- Questions: use GitHub Discussions


