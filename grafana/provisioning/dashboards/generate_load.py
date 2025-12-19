import requests
import time
import threading
import random

def make_request():
    url = "http://localhost:8080/predict"
    data = {
        "Time_broken_spent_Alone": random.uniform(1, 10),
        "Stage_fear": random.randint(0, 1),
        "Social_event_attendance": random.uniform(1, 10),
        "Going_outside": random.uniform(1, 10),
        "Drained_after_socializing": random.randint(0, 1),
        "Friends_circle_size": random.uniform(1, 20),
        "Post_frequency": random.uniform(1, 10)
    }
    
    try:
        start = time.time()
        response = requests.post(url, json=data, timeout=5)
        latency = time.time() - start
        print(f"Status: {response.status_code}, Latency: {latency:.3f}s")
    except Exception as e:
        print(f"Error: {e}")

print("Starting load test...")
threads = []
for i in range(20):
    t = threading.Thread(target=make_request)
    t.start()
    threads.append(t)
    time.sleep(0.1)

for t in threads:
    t.join()

print("Load test complete!")