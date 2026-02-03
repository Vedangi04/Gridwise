# GridWise - Energy Intelligence Platform

A modern, AI-powered energy management dashboard for monitoring, analyzing, and predicting renewable energy generation with real-time insights.

![GridWise Banner](https://img.shields.io/badge/Energy-Management-brightgreen?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-blue?style=flat-square)
![React](https://img.shields.io/badge/React-18.2.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=flat-square)

## 🌟 Features

### Dashboard & Monitoring
- **Real-Time Overview**: Live energy generation stats with status indicators
- **Wind Power Tracking**: Actual vs predicted wind generation with hourly breakdown
- **Solar Power Tracking**: Actual vs predicted solar generation with hourly breakdown
- **Machine Consumption**: Monitor 5 machines with energy consumption and temperature tracking
- **Anomaly Detection**: Real-time alerts for unusual energy consumption or temperature spikes

### Analytics & Insights
- **Model Performance Dashboard**: Comprehensive metrics including MAE, RMSE, MAPE, and accuracy trends
- **Prediction Accuracy Analysis**: Compare original vs improved predictions
- **Residual Charts**: Visualize prediction errors and model performance over time
- **Historical Trend Analysis**: Track model accuracy across 30 days of data

### Machine Learning
- **Model Training**: Train Gradient Boosting or Random Forest models
- **Feature Engineering**: Advanced time-based features (hour, day, month, cyclical encodings)
- **Model Comparison**: Compare original predictions with trained model improvements
- **Persistent Models**: Save trained models using joblib for future predictions

### User Interface
- **Dark Theme**: Modern dark UI with Tailwind CSS
- **Responsive Design**: Works seamlessly on desktop and tablet
- **Interactive Charts**: Recharts for visualization (Line, Area, Pie, Bar charts)
- **Sidebar Navigation**: Easy access to all sections with collapsible menu
- **Date Picker**: Select historical data from available dates

## 🏗️ Architecture

```
GridWise/
├── backend/                    # FastAPI Python backend
│   ├── server.py              # Main API server
│   ├── models/                # Trained ML models (joblib)
│   ├── requirements.txt        # Python dependencies
│   ├── test_predictions.csv    # Wind power data (19,656 samples)
│   ├── test_solar_predictions.csv # Solar power data (19,656 samples)
│   └── machine_test_data.csv   # Machine consumption data
│
├── frontend/                   # React dashboard
│   ├── src/
│   │   ├── App.js             # Main application component
│   │   ├── App.css            # Styling
│   │   └── index.js           # Entry point
│   ├── package.json           # NPM dependencies
│   └── public/
│       └── index.html         # HTML template
│
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** with pip
- **Node.js 14+** with npm
- **macOS, Linux, or Windows** with bash/terminal

### Installation

1. **Clone/Navigate to project directory**
   ```bash
   cd GridWise
   ```

2. **Install Backend Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   # Or on macOS with system Python:
   pip install --break-system-packages -r requirements.txt
   ```

3. **Install Frontend Dependencies**
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

**Terminal 1 - Start Backend Server**
```bash
cd backend
python3 -m uvicorn server:app --host 0.0.0.0 --port 8001
```

**Terminal 2 - Start Frontend Server**
```bash
cd frontend
PORT=3000 npm start
```

### Access the Application
- **Dashboard**: http://localhost:3000
- **API Docs**: http://localhost:8001/docs (Swagger UI)
- **API Health**: http://localhost:8001/api/health

## 📡 API Endpoints

### Health & Status
- `GET /api/health` - Server health check
- `GET /api/dates` - Available dates in dataset
- `GET /api/model-status` - Check trained model status

### Energy Data
- `POST /api/wind-prediction` - Get wind power data for a date
- `POST /api/solar-prediction` - Get solar power data for a date
- `POST /api/machine-consumption` - Get machine energy/temp data
- `POST /api/dashboard-summary` - Aggregated energy stats

### Analytics
- `POST /api/model-performance` - Performance metrics and trends
- `POST /api/ai-insights` - AI-powered insights on energy patterns

### Machine Learning
- `POST /api/train-model` - Train prediction models
  ```json
  {
    "model_type": "wind|solar|both",
    "algorithm": "gradient_boosting|random_forest",
    "test_size": 0.2
  }
  ```
- `POST /api/predict-with-trained` - Get improved predictions from trained models

### Real-Time
- `WebSocket /api/ws/realtime` - Real-time data streaming (2-second updates)

## 📊 Data Overview

### Wind & Solar Predictions
- **19,656 samples** per dataset
- **Columns**: `time`, `ActualPower`, `PredictedPower`
- **Date Range**: Historical power generation data

### Machine Consumption
- **5 Machines** monitored simultaneously
- **Columns per machine**: Energy Consumed (kWh), Temperature (°C)
- **Anomaly Detection**: Alerts when temp > 47°C or energy > 9 kWh

## 🤖 Machine Learning Models

### Feature Engineering
The models use advanced time-based features:
- Hour of day (0-23)
- Hour sine/cosine encoding (cyclical features for 24-hour cycle)
- Day of month sine encoding
- Month sine encoding
- Previous predicted power (baseline model's prediction)
- Hour squared (polynomial feature)

### Model Performance
- **Wind Model**: ~64.7% test accuracy (Gradient Boosting)
- **Solar Model**: Improved with feature engineering
- **Algorithm Comparison**: Support for both Gradient Boosting and Random Forest

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern async Python web framework
- **Uvicorn** - ASGI server
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Machine learning models
- **Joblib** - Model persistence
- **Python WebSockets** - Real-time data streaming

### Frontend
- **React 18** - UI library
- **Tailwind CSS** - Utility-first styling
- **Recharts** - Interactive charts
- **Lucide React** - Icon library
- **Framer Motion** - Animation library
- **Axios** - HTTP client

### Styling & UI
- **Dark Theme**: Zinc color palette
- **Responsive**: Mobile-friendly design
- **Animations**: Smooth transitions and loading states

## 📁 Project Structure

```
backend/server.py (642 lines)
├── CORS middleware configuration
├── CSV data loading
├── Feature extraction for ML
├── API endpoints
├── WebSocket manager
└── ML model training & prediction

frontend/src/App.js (955 lines)
├── Navigation & sidebar
├── Dashboard view
├── Wind/Solar power views
├── Machine consumption view
├── Model performance analytics
├── Custom components (charts, cards, metrics)
└── API integration with axios
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the backend directory (optional):
```bash
# Not required for basic functionality
# EMERGENT_LLM_KEY=your_api_key_here
```

### Customization
- **Backend Port**: Edit `server.py` line 638 to change port
- **Frontend Port**: Set `PORT` environment variable before `npm start`
- **API URL**: Frontend uses relative proxy from `package.json`

## 📈 Model Training Guide

1. Navigate to **Model Performance** tab
2. Click **Train Models**
3. Select:
   - **Algorithm**: Gradient Boosting (recommended) or Random Forest
   - **Model Type**: Wind, Solar, or Both
   - **Test Size**: Train/test split percentage (default 20%)
4. View training results with accuracy metrics
5. Compare original vs improved predictions

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check if port 8001 is in use
lsof -i :8001
# Kill process if needed: kill -9 <PID>

# Reinstall dependencies
pip install --break-system-packages -r requirements.txt
```

### Frontend won't start
```bash
# Clear cache and reinstall
rm -rf frontend/node_modules package-lock.json
cd frontend && npm install
npm start
```

### API connection errors
- Ensure backend is running on `http://localhost:8001`
- Check CORS settings in `backend/server.py`
- Verify no firewall is blocking the ports

### Missing data
- Ensure CSV files are in `backend/` directory
- Check file names match in `server.py` (lines 35-37)

## 📝 Sample API Response

```bash
# Get wind predictions for a date
curl -X POST http://localhost:8001/api/wind-prediction \
  -H "Content-Type: application/json" \
  -d '{"date": "02-01-2022"}'

# Response
{
  "data": [
    {
      "time": "00:00",
      "actual": 1234.56,
      "predicted": 1200.34
    },
    ...
  ],
  "stats": {
    "average": 1250.45,
    "maximum": 1500.23,
    "total": 30010.80
  }
}
```

## 🎯 Use Cases

### Data Analysts
- Analyze energy generation patterns
- Compare prediction accuracy across models
- Track performance trends over time
- Generate data-driven insights

### Grid Operators
- Monitor real-time energy generation
- Receive anomaly alerts
- Track machine health
- Identify bottlenecks

### ML Engineers
- Train and evaluate prediction models
- Compare different algorithms
- Analyze residuals and errors
- Optimize model parameters

### Management
- View high-level energy statistics
- Track efficiency metrics
- Understand generation distribution
- Make informed operational decisions

## 📚 Learning Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **React**: https://react.dev/
- **Scikit-learn ML**: https://scikit-learn.org/
- **Recharts**: https://recharts.org/
- **Tailwind CSS**: https://tailwindcss.com/

## 🤝 Contributing

This is a showcase project for data engineering and ML skills. Feel free to:
- Add new visualizations
- Improve model accuracy
- Add more data sources
- Enhance UI/UX
- Add additional features

## 📄 License

This project is provided as-is for educational and portfolio purposes.

## 👨‍💻 Author

Created as a modern energy management dashboard showcasing:
- Full-stack development (React + FastAPI)
- Machine learning model training
- Real-time data processing
- Data visualization & analytics
- System design & architecture

---

**Last Updated**: February 3, 2026  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
