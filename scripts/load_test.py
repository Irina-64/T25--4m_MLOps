import requests
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import json

def generate_random_payload():
    """Генерация случайного payload для тестирования."""
    return {
        "SeniorCitizen": random.randint(0, 1),
        "tenure": random.randint(0, 72),
        "MonthlyCharges": random.uniform(20, 120),
        "TotalCharges": random.uniform(100, 8000),
        "gender": random.randint(0, 1),
        "Partner": random.randint(0, 1),
        "Dependents": random.randint(0, 1),
        "PhoneService": random.randint(0, 1),
        "MultipleLines": random.randint(0, 1),
        "InternetService_DSL": random.randint(0, 1),
        "InternetService_Fiber optic": random.randint(0, 1),
        "InternetService_No": random.randint(0, 1),
        "Contract_Month-to-month": random.randint(0, 1),
        "Contract_One year": random.randint(0, 1),
        "Contract_Two year": random.randint(0, 1)
    }

def send_request(url, request_id):
    """Отправка одного запроса к API."""
    try:
        start_time = time.time()
        payload = generate_random_payload()
        
        response = requests.post(
            url,
            json=payload,
            timeout=10
        )
        
        latency = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request {request_id}: {latency:.3f}s - Prediction: {result.get('delay_prob', 0):.3f}")
            return True
        else:
            print(f"❌ Request {request_id}: HTTP {response.status_code} - {latency:.3f}s")
            return False
            
    except Exception as e:
        print(f"❌ Request {request_id}: Error - {str(e)}")
        return False

def run_load_test(url, num_requests, concurrency):
    """Запуск нагрузочного теста."""
    print(f"🚀 Starting load test: {num_requests} requests with {concurrency} concurrent threads")
    print(f"📡 Target URL: {url}")
    print("-" * 50)
    
    successful = 0
    failed = 0
    latencies = []
    
    def worker(request_id):
        nonlocal successful, failed
        success = send_request(url, request_id)
        if success:
            successful += 1
        else:
            failed += 1
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i) for i in range(num_requests)]
        
        for future in futures:
            future.result()
    
    total_time = time.time() - start_time
    rps = num_requests / total_time if total_time > 0 else 0
    
    print("-" * 50)
    print(f"📊 Load Test Results:")
    print(f"   Total Requests: {num_requests}")
    print(f"   Successful: {successful} ({successful/num_requests*100:.1f}%)")
    print(f"   Failed: {failed} ({failed/num_requests*100:.1f}%)")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Requests per Second: {rps:.2f}")
    print(f"   Average Latency: {total_time/num_requests*1000:.1f}ms")
    
    return successful, failed, rps

def monitor_metrics(prometheus_url):
    """Мониторинг метрик из Prometheus."""
    try:
        response = requests.get(f"{prometheus_url}/api/v1/query?query=rate(http_requests_total[1m])")
        data = response.json()
        
        print("\n📈 Current Metrics from Prometheus:")
        if data['data']['result']:
            for result in data['data']['result']:
                endpoint = result['metric'].get('endpoint', 'unknown')
                value = result['value'][1]
                print(f"   {endpoint}: {value} req/s")
    except Exception as e:
        print(f"⚠ Could not fetch metrics: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test for Telco Churn API")
    parser.add_argument("--url", default="http://localhost:8080/predict", help="API URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent threads")
    parser.add_argument("--prometheus", default="http://localhost:9090", help="Prometheus URL")
    parser.add_argument("--monitor", action="store_true", help="Monitor metrics during test")
    
    args = parser.parse_args()
    
    # Запуск нагрузочного теста
    run_load_test(args.url, args.requests, args.concurrency)
    
    # Мониторинг метрик
    if args.monitor:
        monitor_metrics(args.prometheus)
    
    print("\n✅ Load test completed!")