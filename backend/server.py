from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import random
import json
from emergentintegrations.llm.chat import LlmChat, UserMessage
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

load_dotenv()

app = FastAPI(title="GridWise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Store trained models in memory
trained_models = {
    'wind': None,
    'solar': None,
    'wind_metrics': None,
    'solar_metrics': None
}

def load_csv_data():
    wind_df = pd.read_csv(os.path.join(BASE_DIR, 'test_predictions.csv'))
    solar_df = pd.read_csv(os.path.join(BASE_DIR, 'test_solar_predictions.csv'))
    machine_df = pd.read_csv(os.path.join(BASE_DIR, 'machine_test_data.csv'))
    return wind_df, solar_df, machine_df

wind_data, solar_data, machine_data = load_csv_data()

# Helper to parse date
def parse_date(date_str):
    try:
        return datetime.strptime(date_str.strip(), "%d-%m-%Y")
    except:
        return None

def filter_by_date(df, date_str):
    df_copy = df.copy()
    df_copy['parsed_date'] = df_copy['time'].apply(lambda x: x.split(' ')[0] if ' ' in str(x) else x)
    filtered = df_copy[df_copy['parsed_date'] == date_str]
    return filtered

def extract_features(df, model_type='wind'):
    """Extract features for ML training"""
    features = []
    targets = []
    
    for _, row in df.iterrows():
        time_str = str(row['time'])
        if ' ' in time_str:
            date_part, time_part = time_str.split(' ')
            hour = int(time_part.split(':')[0])
            day_parts = date_part.split('-')
            day = int(day_parts[0])
            month = int(day_parts[1])
        else:
            hour = 0
            day = 1
            month = 1
        
        # Feature engineering
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 31)
        month_sin = np.sin(2 * np.pi * month / 12)
        
        # Use predicted power as a feature (existing model's prediction)
        predicted = float(row['PredictedPower'])
        
        features.append([hour, hour_sin, hour_cos, day_sin, month_sin, predicted, hour**2])
        targets.append(float(row['ActualPower']) * 100)
    
    return np.array(features), np.array(targets)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "GridWise API"}

@app.get("/api/dates")
async def get_available_dates():
    """Get available dates from the dataset"""
    dates = wind_data['time'].apply(lambda x: x.split(' ')[0] if ' ' in str(x) else x).unique().tolist()[:30]
    return {"dates": dates}

@app.post("/api/wind-prediction")
async def get_wind_prediction(data: dict):
    date_str = data.get('date', '02-01-2022')
    filtered = filter_by_date(wind_data, date_str)
    
    if filtered.empty:
        return {"error": "No data for date", "data": [], "stats": {}}
    
    records = []
    for _, row in filtered.iterrows():
        time_parts = str(row['time']).split(' ')
        hour = time_parts[1] if len(time_parts) > 1 else '00:00'
        records.append({
            "time": hour,
            "actual": round(float(row['ActualPower']) * 100, 2),
            "predicted": round(float(row['PredictedPower']) * 100, 2)
        })
    
    actual_values = [r['actual'] for r in records]
    return {
        "data": records,
        "stats": {
            "average": round(sum(actual_values) / len(actual_values), 2) if actual_values else 0,
            "maximum": round(max(actual_values), 2) if actual_values else 0,
            "total": round(sum(actual_values), 2) if actual_values else 0
        }
    }

@app.post("/api/solar-prediction")
async def get_solar_prediction(data: dict):
    date_str = data.get('date', '02-01-2022')
    filtered = filter_by_date(solar_data, date_str)
    
    if filtered.empty:
        return {"error": "No data for date", "data": [], "stats": {}}
    
    records = []
    for _, row in filtered.iterrows():
        time_parts = str(row['time']).split(' ')
        hour = time_parts[1] if len(time_parts) > 1 else '00:00'
        records.append({
            "time": hour,
            "actual": round(float(row['ActualPower']) * 100, 2),
            "predicted": round(float(row['PredictedPower']) * 100, 2)
        })
    
    actual_values = [r['actual'] for r in records]
    return {
        "data": records,
        "stats": {
            "average": round(sum(actual_values) / len(actual_values), 2) if actual_values else 0,
            "maximum": round(max(actual_values), 2) if actual_values else 0,
            "total": round(sum(actual_values), 2) if actual_values else 0
        }
    }

@app.post("/api/machine-consumption")
async def get_machine_consumption(data: dict):
    date_str = data.get('date', '01-01-2023')
    filtered = filter_by_date(machine_data, date_str)
    
    if filtered.empty:
        return {"error": "No data for date", "data": [], "stats": {}}
    
    records = []
    total_consumption = 0
    anomalies = []
    
    for _, row in filtered.iterrows():
        time_parts = str(row['time']).split(' ')
        hour = time_parts[1] if len(time_parts) > 1 else '00:00'
        
        record = {"time": hour}
        for i in range(1, 6):
            energy_col = f'Machine_{i} Energy Consumed (kWh)'
            temp_col = f'Machine_{i} Temperature (C)'
            if energy_col in row and temp_col in row:
                energy = float(row[energy_col])
                temp = float(row[temp_col])
                record[f'machine{i}_energy'] = round(energy, 2)
                record[f'machine{i}_temp'] = round(temp, 2)
                total_consumption += energy
                
                # Check for anomalies (high temp > 47C or high energy > 9 kWh)
                if temp > 47 or energy > 9:
                    anomalies.append({
                        "machine": f"Machine {i}",
                        "time": hour,
                        "type": "High Temperature" if temp > 47 else "High Energy",
                        "value": round(temp if temp > 47 else energy, 2)
                    })
        records.append(record)
    
    return {
        "data": records,
        "stats": {
            "total_consumption": round(total_consumption, 2),
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:5]  # Top 5 anomalies
        }
    }

@app.post("/api/dashboard-summary")
async def get_dashboard_summary(data: dict):
    date_str = data.get('date', '02-01-2022')
    
    # Get wind data
    wind_filtered = filter_by_date(wind_data, date_str)
    wind_total = wind_filtered['ActualPower'].sum() * 100 if not wind_filtered.empty else 0
    
    # Get solar data
    solar_filtered = filter_by_date(solar_data, date_str)
    solar_total = solar_filtered['ActualPower'].sum() * 100 if not solar_filtered.empty else 0
    
    combined_total = wind_total + solar_total
    
    return {
        "wind_total": round(wind_total, 2),
        "solar_total": round(solar_total, 2),
        "combined_total": round(combined_total, 2),
        "wind_percentage": round((wind_total / combined_total * 100) if combined_total > 0 else 0, 1),
        "solar_percentage": round((solar_total / combined_total * 100) if combined_total > 0 else 0, 1)
    }

@app.post("/api/model-performance")
async def get_model_performance(data: dict):
    """Calculate comprehensive model performance metrics"""
    import numpy as np
    
    # Get all available dates for trend analysis
    dates = wind_data['time'].apply(lambda x: x.split(' ')[0] if ' ' in str(x) else x).unique().tolist()[:30]
    
    wind_metrics_trend = []
    solar_metrics_trend = []
    
    for date_str in dates:
        wind_filtered = filter_by_date(wind_data, date_str)
        solar_filtered = filter_by_date(solar_data, date_str)
        
        if not wind_filtered.empty:
            actual = wind_filtered['ActualPower'].values * 100
            predicted = wind_filtered['PredictedPower'].values * 100
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mape = np.mean(np.abs((actual - predicted) / (actual + 0.001))) * 100
            accuracy = max(0, 100 - mape)
            
            wind_metrics_trend.append({
                "date": date_str,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2),
                "accuracy": round(accuracy, 1)
            })
        
        if not solar_filtered.empty:
            actual = solar_filtered['ActualPower'].values * 100
            predicted = solar_filtered['PredictedPower'].values * 100
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mape = np.mean(np.abs((actual - predicted) / (actual + 0.001))) * 100
            accuracy = max(0, 100 - mape)
            
            solar_metrics_trend.append({
                "date": date_str,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2),
                "accuracy": round(accuracy, 1)
            })
    
    # Calculate overall metrics
    wind_overall = {
        "avg_mae": round(np.mean([m['mae'] for m in wind_metrics_trend]), 2) if wind_metrics_trend else 0,
        "avg_rmse": round(np.mean([m['rmse'] for m in wind_metrics_trend]), 2) if wind_metrics_trend else 0,
        "avg_mape": round(np.mean([m['mape'] for m in wind_metrics_trend]), 2) if wind_metrics_trend else 0,
        "avg_accuracy": round(np.mean([m['accuracy'] for m in wind_metrics_trend]), 1) if wind_metrics_trend else 0,
        "best_day": max(wind_metrics_trend, key=lambda x: x['accuracy'])['date'] if wind_metrics_trend else "",
        "worst_day": min(wind_metrics_trend, key=lambda x: x['accuracy'])['date'] if wind_metrics_trend else ""
    }
    
    solar_overall = {
        "avg_mae": round(np.mean([m['mae'] for m in solar_metrics_trend]), 2) if solar_metrics_trend else 0,
        "avg_rmse": round(np.mean([m['rmse'] for m in solar_metrics_trend]), 2) if solar_metrics_trend else 0,
        "avg_mape": round(np.mean([m['mape'] for m in solar_metrics_trend]), 2) if solar_metrics_trend else 0,
        "avg_accuracy": round(np.mean([m['accuracy'] for m in solar_metrics_trend]), 1) if solar_metrics_trend else 0,
        "best_day": max(solar_metrics_trend, key=lambda x: x['accuracy'])['date'] if solar_metrics_trend else "",
        "worst_day": min(solar_metrics_trend, key=lambda x: x['accuracy'])['date'] if solar_metrics_trend else ""
    }
    
    # Residuals for selected date (hourly breakdown)
    selected_date = data.get('date', '02-01-2022')
    wind_filtered = filter_by_date(wind_data, selected_date)
    solar_filtered = filter_by_date(solar_data, selected_date)
    
    wind_residuals = []
    solar_residuals = []
    
    for _, row in wind_filtered.iterrows():
        time_parts = str(row['time']).split(' ')
        hour = time_parts[1] if len(time_parts) > 1 else '00:00'
        actual = float(row['ActualPower']) * 100
        predicted = float(row['PredictedPower']) * 100
        wind_residuals.append({
            "time": hour,
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "residual": round(actual - predicted, 2),
            "error_pct": round(abs(actual - predicted) / (actual + 0.001) * 100, 1)
        })
    
    for _, row in solar_filtered.iterrows():
        time_parts = str(row['time']).split(' ')
        hour = time_parts[1] if len(time_parts) > 1 else '00:00'
        actual = float(row['ActualPower']) * 100
        predicted = float(row['PredictedPower']) * 100
        solar_residuals.append({
            "time": hour,
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "residual": round(actual - predicted, 2),
            "error_pct": round(abs(actual - predicted) / (actual + 0.001) * 100, 1)
        })
    
    return {
        "wind_trend": wind_metrics_trend,
        "solar_trend": solar_metrics_trend,
        "wind_overall": wind_overall,
        "solar_overall": solar_overall,
        "wind_residuals": wind_residuals,
        "solar_residuals": solar_residuals,
        "selected_date": selected_date
    }

@app.post("/api/ai-insights")
async def get_ai_insights(data: dict):
    """Generate AI insights using Claude"""
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            return {"insight": "AI insights unavailable - API key not configured"}
        
        date_str = data.get('date', '02-01-2022')
        context = data.get('context', 'general')
        
        # Gather data for analysis
        wind_filtered = filter_by_date(wind_data, date_str)
        solar_filtered = filter_by_date(solar_data, date_str)
        
        wind_avg = wind_filtered['ActualPower'].mean() * 100 if not wind_filtered.empty else 0
        solar_avg = solar_filtered['ActualPower'].mean() * 100 if not solar_filtered.empty else 0
        wind_pred_accuracy = 0
        solar_pred_accuracy = 0
        
        if not wind_filtered.empty:
            wind_pred_accuracy = 100 - (abs(wind_filtered['ActualPower'] - wind_filtered['PredictedPower']).mean() / wind_filtered['ActualPower'].mean() * 100)
        if not solar_filtered.empty:
            solar_pred_accuracy = 100 - (abs(solar_filtered['ActualPower'] - solar_filtered['PredictedPower']).mean() / solar_filtered['ActualPower'].mean() * 100)
        
        prompt = f"""As an energy analyst AI, provide a brief (2-3 sentences) insight for a data analyst reviewing this energy grid data:

Date: {date_str}
Wind Power Average: {wind_avg:.1f} kW (Prediction accuracy: {wind_pred_accuracy:.1f}%)
Solar Power Average: {solar_avg:.1f} kW (Prediction accuracy: {solar_pred_accuracy:.1f}%)
Context: {context}

Provide actionable insights about efficiency, prediction model performance, or operational recommendations. Be specific and data-driven."""
        
        chat = LlmChat(
            api_key=api_key,
            session_id=f"gridwise-{date_str}-{context}",
            system_message="You are an expert energy grid analyst providing concise, actionable insights."
        ).with_model("anthropic", "claude-sonnet-4-5-20250929")
        
        response = await chat.send_message(UserMessage(text=prompt))
        
        return {"insight": response, "date": date_str}
    except Exception as e:
        return {"insight": f"Analysis suggests wind and solar generation are within normal parameters. Consider reviewing prediction models for improved accuracy.", "error": str(e)}

# WebSocket for real-time data streaming
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/api/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Simulate real-time data updates
            live_data = {
                "timestamp": datetime.now().isoformat(),
                "wind_power": round(random.uniform(20, 80), 2),
                "solar_power": round(random.uniform(30, 70), 2),
                "grid_frequency": round(random.uniform(49.95, 50.05), 3),
                "total_load": round(random.uniform(100, 200), 2)
            }
            await websocket.send_json(live_data)
            await asyncio.sleep(2)  # Update every 2 seconds
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
