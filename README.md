# 🚀 Retail Sales ML Platform

> Full-stack Machine Learning application for retail sales forecasting with FastAPI + React

[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://github.com/KapProgr/retail-sales-api)
[![Frontend](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge&logo=react)](https://github.com/KapProgr/retail-sales-frontend)
[![Live Demo](https://img.shields.io/badge/Live-Demo-blueviolet?style=for-the-badge)](https://retail-sales.vercel.app)

---

## 🌐 Quick Links

| 🎯 Purpose | 🔗 Link | 📝 Description |
|-----------|---------|---------------|
| **Live App** | [retail-sales-ml.vercel.app](https://retail-sales.vercel.app) | Production web application |
| **API Docs** | [retail-sales-api.onrender.com/docs](https://retail-sales-api.onrender.com/docs) | Interactive API documentation |
| **Backend Repo** | [retail-sales-api](https://github.com/KapProgr/retail-sales-api) | Python FastAPI + ML models |
| **Frontend Repo** | [retail-sales-frontend](https://github.com/KapProgr/retail-sales-frontend) | React + Vite + Tailwind |

---

## 📊 Project Overview

This is a **production-ready full-stack ML application** that predicts retail sales using multiple machine learning models. Built with modern technologies and deployed on cloud platforms.

### 🎯 Key Capabilities

- 📤 **Upload CSV files** with historical sales data
- 🎲 **Generate synthetic data** for testing (365 days)
- 🤖 **Train 3 ML models** simultaneously
- 📊 **Interactive visualizations** of predictions
- 🎯 **Compare model performance** (MAE, RMSE, R², MAPE)
- 🔮 **30-day future forecasts**
- 📈 **Feature importance analysis**
- 💾 **Export results** to CSV

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    USER BROWSER                          │
│            https://retail-sales.vercel.app              │
└─────────────────────┬───────────────────────────────────┘
                      │
                      │ HTTP/JSON
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  FRONTEND (Vercel)                       │
│  React 18 + Vite + Tailwind CSS + Recharts              │
│  - Upload UI                                             │
│  - Charts & Visualizations                               │
│  - Model Comparison Dashboard                            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      │ REST API
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  BACKEND (Render)                        │
│  FastAPI + Python 3.13                                   │
│  - /upload - File upload endpoint                        │
│  - /predict - Model training endpoint                    │
│  - /sample - Sample data generator                       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      │ scikit-learn
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 ML PIPELINE                              │
│  1. Feature Engineering                                  │
│     - Lag features (1, 7, 14, 30 days)                  │
│     - Rolling statistics                                 │
│     - Cyclical time features                             │
│  2. Model Training                                       │
│     - Linear Regression                                  │
│     - Random Forest                                      │
│     - Gradient Boosting                                  │
│  3. Evaluation & Prediction                              │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Repository Structure

### 🔧 Backend Repository
**[retail-sales-api](https://github.com/KapProgr/retail-sales-api)**
```
retail-sales-api/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
└── README.md           # Backend documentation
```

**Tech Stack:**
- FastAPI 0.115
- scikit-learn 1.5.2
- pandas 2.2.3
- numpy 2.1.0

**Endpoints:**
- `GET /` - API info
- `GET /health` - Health check
- `GET /sample` - Generate sample data
- `POST /upload` - Upload CSV file
- `POST /predict` - Train models & predict

---

### 🎨 Frontend Repository
**[retail-sales-frontend](https://github.com/KapProgr/retail-sales-frontend)**
```
retail-sales-frontend/
├── src/
│   ├── App.jsx         # Main React component
│   ├── index.css       # Tailwind styles
│   └── main.jsx        # Entry point
├── public/             # Static assets
├── package.json        # Dependencies
├── vite.config.js      # Vite configuration
└── tailwind.config.js  # Tailwind configuration
```

**Tech Stack:**
- React 18
- Vite 5
- Tailwind CSS 3.4
- Recharts 2.x
- Lucide React (icons)

---

## 🚀 Getting Started

### Prerequisites

- **Backend**: Python 3.9+, pip
- **Frontend**: Node.js 18+, npm

### Local Development

#### 1️⃣ Clone Repositories
```bash
# Clone backend
git clone https://github.com/KapProgr/retail-sales-api.git
cd retail-sales-api

# Clone frontend (in another terminal)
git clone https://github.com/KapProgr/retail-sales-frontend.git
cd retail-sales-frontend
```

#### 2️⃣ Setup Backend
```bash
cd retail-sales-api

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload

# API will be available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

#### 3️⃣ Setup Frontend
```bash
cd retail-sales-frontend

# Install dependencies
npm install

# Update API URL in src/App.jsx
# const API_URL = 'http://localhost:8000';

# Run dev server
npm run dev

# App will be available at http://localhost:5173
```

---

## 🤖 Machine Learning Models

### Models Implemented

| Model | Type | Best For | Performance |
|-------|------|----------|-------------|
| **Linear Regression** | Baseline | Fast training, interpretable | MAE: ~$192 |
| **Random Forest** | Ensemble | Feature importance, robust | MAE: ~$195 |
| **Gradient Boosting** | Ensemble | High accuracy | MAE: ~$206 |

### Features Engineered

1. **Lag Features**: Sales from 1, 7, 14, and 30 days ago
2. **Rolling Statistics**: Mean and std dev over 7, 14, 30 day windows
3. **Time Features**:
   - Day of week, month, quarter
   - Weekend indicator
   - Holiday indicator
4. **Cyclical Features**: Sin/cos transformations for seasonality

### Evaluation Metrics

- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R²** (R-squared): Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): Error as percentage

---

## 📈 Performance

Real-world performance on synthetic retail data:
```
Model: Linear Regression (BEST)
├─ Accuracy: 87.6%
├─ MAE: $191.76
├─ RMSE: $251.41
├─ R²: 0.4879
└─ MAPE: 12.45%
```

---

## 🛠️ Deployment

### Backend (Render)

1. Push to GitHub
2. Connect to Render
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy ✅

### Frontend (Vercel)

1. Push to GitHub
2. Import to Vercel
3. Configure:
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Deploy ✅

---


## 🧪 Testing

### Backend Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test sample data
curl http://localhost:8000/sample

# Test with Swagger UI
open http://localhost:8000/docs
```

### Frontend Testing
```bash
# Run dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 📊 Data Format

CSV files should have these columns:
```csv
date,sales
2023-01-01,1200.50
2023-01-02,1350.75
2023-01-03,1180.25
```

**Requirements:**
- Minimum 100 records
- Date format: YYYY-MM-DD
- Sales: Positive numbers

---

## 🔒 Security

- ✅ No API keys required
- ✅ CORS properly configured
- ✅ Input validation on backend
- ✅ No sensitive data storage
- ✅ HTTPS in production

---

## 🚧 Roadmap

- [ ] Authentication & user accounts
- [ ] Database for storing predictions
- [ ] More ML models (LSTM, Prophet)
- [ ] Email reports
- [ ] Mobile app
- [ ] Real-time predictions
- [ ] A/B testing framework
- [ ] Custom model training

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the relevant repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - feel free to use this project for learning or commercial purposes!

---

## 👤 Author

**KapProgr**

- GitHub: [@KapProgr](https://github.com/KapProgr)
- Backend: [retail-sales-api](https://github.com/KapProgr/retail-sales-api)
- Frontend: [retail-sales-frontend](https://github.com/KapProgr/retail-sales-frontend)

---

## 🙏 Acknowledgments

Built with amazing open-source tools:

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - UI library
- [scikit-learn](https://scikit-learn.org/) - ML library
- [Tailwind CSS](https://tailwindcss.com/) - CSS framework
- [Recharts](https://recharts.org/) - React charting library
- [Render](https://render.com/) - Backend hosting
- [Vercel](https://vercel.com/) - Frontend hosting

---

## 💬 Support

- 🐛 **Issues**: Open an issue in the relevant repository
- 💡 **Questions**: Use GitHub Discussions
- 📧 **Contact**: your-email@example.com

---

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/KapProgr/retail-sales-ml-platform?style=social)](https://github.com/KapProgr/retail-sales-ml-platform)

---

**Made with lots of ☕**
