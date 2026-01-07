import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8080/predict"

def generate_random_request():
    """Генератор случайных валидных запросов"""
    return {
        "USD_RUB": random.uniform(70, 100),
        "EUR_RUB": random.uniform(80, 110),
        "GBP_RUB": random.uniform(90, 120),
        "day_of_week": random.randint(0, 6),
        "is_weekend": random.randint(0, 1),
        "USD_RUB_lag_1": random.uniform(70, 100),
        "USD_RUB_lag_2": random.uniform(70, 100),
        "USD_RUB_lag_3": random.uniform(70, 100),
        "USD_RUB_lag_5": random.uniform(70, 100),
        "USD_RUB_lag_7": random.uniform(70, 100),
        "EUR_RUB_lag_1": random.uniform(80, 110),
        "EUR_RUB_lag_2": random.uniform(80, 110),
        "EUR_RUB_lag_3": random.uniform(80, 110),
        "EUR_RUB_lag_5": random.uniform(80, 110),
        "EUR_RUB_lag_7": random.uniform(80, 110),
        "GBP_RUB_lag_1": random.uniform(90, 120),
        "GBP_RUB_lag_2": random.uniform(90, 120),
        "GBP_RUB_lag_3": random.uniform(90, 120),
        "GBP_RUB_lag_5": random.uniform(90, 120),
        "GBP_RUB_lag_7": random.uniform(90, 120),
        "USD_RUB_MA_3": random.uniform(70, 100),
        "USD_RUB_MA_5": random.uniform(70, 100),
        "USD_RUB_MA_7": random.uniform(70, 100),
        "EUR_RUB_MA_3": random.uniform(80, 110),
        "EUR_RUB_MA_5": random.uniform(80, 110),
        "EUR_RUB_MA_7": random.uniform(80, 110),
        "GBP_RUB_MA_3": random.uniform(90, 120),
        "GBP_RUB_MA_5": random.uniform(90, 120),
        "GBP_RUB_MA_7": random.uniform(90, 120),
        "USD_RUB_change_1": random.uniform(-5, 5),
        "USD_RUB_change_3": random.uniform(-10, 10)
    }

def make_request():
    """Отправка запроса с случайной задержкой для имитации нагрузки"""
    try:
        # 10% шанс на долгий запрос
        if random.random() < 0.1:
            time.sleep(random.uniform(2, 5))
        
        response = requests.post(API_URL, json=generate_random_request(), timeout=10)
        return response.status_code
    except Exception as e:
        return str(e)

def generate_load(requests_per_second=5, duration_seconds=60):
    """Генерация нагрузки на API"""
    print(f"Генерация нагрузки: {requests_per_second} запр/сек в течение {duration_seconds} сек")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            futures = []
            for _ in range(requests_per_second):
                futures.append(executor.submit(make_request))
                time.sleep(1/requests_per_second)
            
            # Обработка результатов
            for future in futures:
                try:
                    result = future.result(timeout=10)
                    if result != 200:
                        print(f"Неудачный запрос: {result}")
                except:
                    pass
    
    print("Нагрузка завершена")

if __name__ == "__main__":
    # Запуск: 5 запросов в секунду в течение 60 секунд
    generate_load(requests_per_second=5, duration_seconds=60)