import requests
import time
import sys

def check_metrics_endpoint(api_url="http://localhost:8080/metrics"):
    """Проверка доступности эндпоинта с метриками."""
    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Metrics endpoint is accessible: {api_url}")
            
            # Проверка наличия ключевых метрик
            content = response.text
            metrics_to_check = [
                "http_requests_total",
                "http_request_duration_seconds",
                "prediction_probability",
                "model_load_total"
            ]
            
            for metric in metrics_to_check:
                if metric in content:
                    print(f"   ✓ Found metric: {metric}")
                else:
                    print(f"   ⚠ Missing metric: {metric}")
            
            return True
        else:
            print(f"❌ Metrics endpoint returned HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to metrics endpoint: {api_url}")
        return False
    except Exception as e:
        print(f"❌ Error checking metrics: {e}")
        return False

def check_prometheus_target(prometheus_url="http://localhost:9090"):
    """Проверка, что Prometheus видит наш target."""
    try:
        response = requests.get(f"{prometheus_url}/api/v1/targets")
        if response.status_code == 200:
            data = response.json()
            
            print("\n🔍 Prometheus Targets Status:")
            targets = data['data']['activeTargets']
            
            found = False
            for target in targets:
                if 'telco-churn-api' in str(target['labels']):
                    found = True
                    print(f"   ✅ Target found: {target['scrapeUrl']}")
                    print(f"      Health: {target['health']}")
                    print(f"      Last scrape: {target['lastScrape']}")
            
            if not found:
                print("   ⚠ Telco Churn API target not found in Prometheus")
            
            return found
            
    except Exception as e:
        print(f"⚠ Could not check Prometheus targets: {e}")
        return False

def check_grafana(grafana_url="http://localhost:3000"):
    """Проверка доступности Grafana."""
    try:
        response = requests.get(f"{grafana_url}/api/health", timeout=5)
        if response.status_code == 200:
            print(f"\n✅ Grafana is accessible: {grafana_url}")
            return True
        else:
            print(f"\n❌ Grafana returned HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"\n⚠ Could not connect to Grafana: {e}")
        return False

def main():
    """Основная функция проверки мониторинга."""
    print("=" * 60)
    print("🔧 MONITORING SYSTEM HEALTH CHECK")
    print("=" * 60)
    
    # Проверка API метрик
    print("\n1. Checking API Metrics Endpoint:")
    api_ok = check_metrics_endpoint()
    
    # Проверка Prometheus
    print("\n2. Checking Prometheus:")
    prometheus_ok = check_prometheus_target()
    
    # Проверка Grafana
    print("\n3. Checking Grafana:")
    grafana_ok = check_grafana()
    
    # Итоговый статус
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"   API Metrics: {'✅ OK' if api_ok else '❌ FAILED'}")
    print(f"   Prometheus:  {'✅ OK' if prometheus_ok else '❌ FAILED'}")
    print(f"   Grafana:     {'✅ OK' if grafana_ok else '❌ FAILED'}")
    
    if api_ok and prometheus_ok and grafana_ok:
        print("\n🎉 All monitoring components are working correctly!")
        print("\n📌 Access URLs:")
        print("   - API: http://localhost:8080")
        print("   - API Metrics: http://localhost:8080/metrics")
        print("   - Prometheus: http://localhost:9090")
        print("   - Grafana: http://localhost:3000 (admin/admin)")
        return 0
    else:
        print("\n⚠ Some monitoring components are not working.")
        return 1

if __name__ == "__main__":
    sys.exit(main())    