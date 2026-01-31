import requests
import sys
import json
from datetime import datetime

class GridWiseAPITester:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=10):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                self.failed_tests.append({
                    "test": name,
                    "endpoint": endpoint,
                    "expected": expected_status,
                    "actual": response.status_code,
                    "response": response.text[:200]
                })
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.failed_tests.append({
                "test": name,
                "endpoint": endpoint,
                "error": str(e)
            })
            return False, {}

    def test_health_check(self):
        """Test health endpoint"""
        return self.run_test("Health Check", "GET", "api/health", 200)

    def test_get_dates(self):
        """Test available dates endpoint"""
        success, response = self.run_test("Get Available Dates", "GET", "api/dates", 200)
        if success and 'dates' in response:
            print(f"   Found {len(response['dates'])} available dates")
            return response['dates']
        return []

    def test_wind_prediction(self, date="02-01-2022"):
        """Test wind prediction endpoint"""
        success, response = self.run_test(
            "Wind Prediction",
            "POST",
            "api/wind-prediction",
            200,
            data={"date": date}
        )
        if success:
            if 'data' in response and 'stats' in response:
                print(f"   Data points: {len(response['data'])}")
                print(f"   Stats: {response['stats']}")
                return True
        return False

    def test_solar_prediction(self, date="02-01-2022"):
        """Test solar prediction endpoint"""
        success, response = self.run_test(
            "Solar Prediction",
            "POST",
            "api/solar-prediction",
            200,
            data={"date": date}
        )
        if success:
            if 'data' in response and 'stats' in response:
                print(f"   Data points: {len(response['data'])}")
                print(f"   Stats: {response['stats']}")
                return True
        return False

    def test_machine_consumption(self, date="01-01-2023"):
        """Test machine consumption endpoint"""
        success, response = self.run_test(
            "Machine Consumption",
            "POST",
            "api/machine-consumption",
            200,
            data={"date": date}
        )
        if success:
            if 'data' in response and 'stats' in response:
                print(f"   Data points: {len(response['data'])}")
                print(f"   Anomalies: {response['stats'].get('anomaly_count', 0)}")
                return True
        return False

    def test_dashboard_summary(self, date="02-01-2022"):
        """Test dashboard summary endpoint"""
        success, response = self.run_test(
            "Dashboard Summary",
            "POST",
            "api/dashboard-summary",
            200,
            data={"date": date}
        )
        if success:
            required_keys = ['wind_total', 'solar_total', 'combined_total', 'wind_percentage', 'solar_percentage']
            if all(key in response for key in required_keys):
                print(f"   Wind: {response['wind_total']} kW ({response['wind_percentage']}%)")
                print(f"   Solar: {response['solar_total']} kW ({response['solar_percentage']}%)")
                return True
        return False

    def test_ai_insights(self, date="02-01-2022"):
        """Test AI insights endpoint (may take longer)"""
        success, response = self.run_test(
            "AI Insights",
            "POST",
            "api/ai-insights",
            200,
            data={"date": date, "context": "dashboard"},
            timeout=30  # Longer timeout for AI processing
        )
        if success:
            if 'insight' in response:
                print(f"   Insight length: {len(response['insight'])} characters")
                return True
        return False

    def test_model_performance(self, date="02-01-2022"):
        """Test model performance endpoint"""
        success, response = self.run_test(
            "Model Performance",
            "POST",
            "api/model-performance",
            200,
            data={"date": date},
            timeout=15  # May take time for calculations
        )
        if success:
            required_keys = ['wind_trend', 'solar_trend', 'wind_overall', 'solar_overall', 'wind_residuals', 'solar_residuals']
            if all(key in response for key in required_keys):
                print(f"   Wind trend data points: {len(response['wind_trend'])}")
                print(f"   Solar trend data points: {len(response['solar_trend'])}")
                print(f"   Wind overall metrics: {response['wind_overall']}")
                print(f"   Solar overall metrics: {response['solar_overall']}")
                print(f"   Wind residuals: {len(response['wind_residuals'])}")
                print(f"   Solar residuals: {len(response['solar_residuals'])}")
                return True
        return False

    def test_model_status(self):
        """Test model status endpoint"""
        success, response = self.run_test(
            "Model Status",
            "GET",
            "api/model-status",
            200
        )
        if success:
            required_keys = ['wind_trained', 'solar_trained']
            if all(key in response for key in required_keys):
                print(f"   Wind trained: {response['wind_trained']}")
                print(f"   Solar trained: {response['solar_trained']}")
                if response.get('wind_metrics'):
                    print(f"   Wind metrics: {response['wind_metrics']}")
                if response.get('solar_metrics'):
                    print(f"   Solar metrics: {response['solar_metrics']}")
                return True, response
        return False, {}

    def test_train_model(self, algorithm="gradient_boosting"):
        """Test model training endpoint"""
        success, response = self.run_test(
            f"Train Model ({algorithm})",
            "POST",
            "api/train-model",
            200,
            data={
                "model_type": "both",
                "algorithm": algorithm,
                "test_size": 0.2
            },
            timeout=60  # Training may take time
        )
        if success:
            required_keys = ['status', 'algorithm', 'results']
            if all(key in response for key in required_keys):
                print(f"   Status: {response['status']}")
                print(f"   Algorithm: {response['algorithm']}")
                print(f"   Results: {response['results']}")
                return True, response
        return False, {}

    def test_predict_with_trained(self, date="02-01-2022"):
        """Test predictions with trained models"""
        success, response = self.run_test(
            "Predict with Trained Models",
            "POST",
            "api/predict-with-trained",
            200,
            data={"date": date},
            timeout=15
        )
        if success:
            print(f"   Wind predictions: {len(response.get('wind_predictions', []))}")
            print(f"   Solar predictions: {len(response.get('solar_predictions', []))}")
            if response.get('wind_improvement'):
                print(f"   Wind improvement: {response['wind_improvement']}")
            if response.get('solar_improvement'):
                print(f"   Solar improvement: {response['solar_improvement']}")
            return True, response
        return False, {}

def main():
    print("🚀 Starting GridWise API Testing...")
    print("=" * 50)
    
    tester = GridWiseAPITester()
    
    # Test basic connectivity
    if not tester.test_health_check()[0]:
        print("❌ Health check failed - API may be down")
        return 1
    
    # Get available dates
    dates = tester.test_get_dates()
    test_date = dates[0] if dates else "02-01-2022"
    machine_date = "01-01-2023"  # Machine data uses different dates
    
    print(f"\n📅 Using test date: {test_date}")
    print(f"📅 Using machine date: {machine_date}")
    
    # Test all endpoints
    test_results = []
    test_results.append(tester.test_wind_prediction(test_date))
    test_results.append(tester.test_solar_prediction(test_date))
    test_results.append(tester.test_machine_consumption(machine_date))
    test_results.append(tester.test_dashboard_summary(test_date))
    test_results.append(tester.test_model_performance(test_date))
    test_results.append(tester.test_ai_insights(test_date))
    
    # Test new Model Training endpoints
    print(f"\n🧠 Testing Model Training Features...")
    model_status_success, model_status = tester.test_model_status()
    test_results.append(model_status_success)
    
    # Test model training with Gradient Boosting
    train_gb_success, train_gb_response = tester.test_train_model("gradient_boosting")
    test_results.append(train_gb_success)
    
    # Test model training with Random Forest
    train_rf_success, train_rf_response = tester.test_train_model("random_forest")
    test_results.append(train_rf_success)
    
    # Test predictions with trained models (only if training succeeded)
    if train_gb_success or train_rf_success:
        predict_success, predict_response = tester.test_predict_with_trained(test_date)
        test_results.append(predict_success)
    else:
        print("⚠️  Skipping trained predictions test - no models trained successfully")
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} passed")
    
    if tester.failed_tests:
        print("\n❌ Failed Tests:")
        for failure in tester.failed_tests:
            error_msg = failure.get('error', f"Status {failure.get('actual', 'unknown')}")
            print(f"   - {failure['test']}: {error_msg}")
    
    success_rate = (tester.tests_passed / tester.tests_run * 100) if tester.tests_run > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    return 0 if success_rate >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())