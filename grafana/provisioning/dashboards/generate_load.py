import requests
import time
import threading
import random
import signal
import sys

running = True

def signal_handler(sig, frame):
    global running
    print("\nStopping load test...")
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def make_request(worker_id, request_count):
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
        response = requests.post(url, json=data, timeout=3)
        latency = time.time() - start
        if request_count % 20 == 0:
            print(f"Worker {worker_id}: {request_count} requests, latency: {latency:.3f}s, status: {response.status_code}")
        return True
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        return False

def worker(worker_id, requests_per_second=2):
    interval = 1.0 / requests_per_second
    request_count = 0
    
    while running:
        success = make_request(worker_id, request_count)
        request_count += 1
        sleep_time = interval + random.uniform(-0.1, 0.1)
        time.sleep(max(0.01, sleep_time))

if __name__ == "__main__":
    print("Starting continuous load test (Ctrl+C to stop)...")
    print("Generating ~10 requests per second...")
    
    workers = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i, 2))
        t.daemon = True
        t.start()
        workers.append(t)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        print("\nLoad test stopped.")
