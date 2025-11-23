"""
Locust load testing configuration for Plant Disease Detection API
"""
from locust import HttpUser, task, between
import random
from io import BytesIO
from PIL import Image
import json

class PlantDiseaseUser(HttpUser):
    """Simulates a user interacting with the Plant Disease Detection API"""
    
    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Check if API is available
        response = self.client.get("/health")
        if response.status_code != 200:
            print("API is not healthy!")
    
    def create_test_image(self, size=(224, 224)):
        """Create a test image for prediction"""
        # Create random image
        img = Image.new('RGB', size, color=(
            random.randint(0, 255),
            random.randint(100, 200),
            random.randint(0, 150)
        ))
        
        # Convert to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes
    
    @task(10)
    def predict_single_image(self):
        """
        Task: Predict disease for a single image
        Weight: 10 (most common operation)
        """
        img_bytes = self.create_test_image()
        
        files = {
            'file': ('test_image.jpg', img_bytes, 'image/jpeg')
        }
        
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if 'predicted_class' in result and 'confidence' in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    def predict_batch_images(self):
        """
        Task: Predict disease for multiple images
        Weight: 3 (less common)
        """
        num_images = random.randint(2, 5)
        files = []
        
        for i in range(num_images):
            img_bytes = self.create_test_image()
            files.append(
                ('files', (f'test_image_{i}.jpg', img_bytes, 'image/jpeg'))
            )
        
        with self.client.post(
            "/predict-batch",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if 'results' in result and len(result['results']) == num_images:
                    response.success()
                else:
                    response.failure("Invalid batch response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(5)
    def check_status(self):
        """
        Task: Check model status
        Weight: 5 (monitoring operation)
        """
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if 'status' in result and 'total_requests' in result:
                    response.success()
                else:
                    response.failure("Invalid status response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def get_model_info(self):
        """
        Task: Get model information
        Weight: 2 (occasional operation)
        """
        with self.client.get("/model-info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """
        Task: Health check
        Weight: 1 (periodic monitoring)
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class AdminUser(HttpUser):
    """Simulates an admin user with different behavior patterns"""
    
    wait_time = between(5, 10)
    
    @task(1)
    def check_retrain_status(self):
        """Check retraining status"""
        with self.client.get("/retrain-status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def monitor_system(self):
        """Monitor system metrics"""
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


# Custom load shape for gradual ramp-up
from locust import LoadTestShape

class StagesShape(LoadTestShape):
    """
    A custom load test shape that gradually increases load
    
    Stages:
    1. 10 users for 60s
    2. 50 users for 120s
    3. 100 users for 120s
    4. 50 users for 60s (ramp down)
    """
    
    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 2},
        {"duration": 180, "users": 50, "spawn_rate": 5},
        {"duration": 300, "users": 100, "spawn_rate": 10},
        {"duration": 360, "users": 50, "spawn_rate": 5},
        {"duration": 420, "users": 10, "spawn_rate": 2}
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data
        
        return None


"""
USAGE INSTRUCTIONS:

1. Basic load test:
   locust -f locustfile.py --host=http://localhost:8000

2. Headless mode with specific parameters:
   locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 5m --headless

3. With custom load shape:
   locust -f locustfile.py --host=http://localhost:8000 --headless

4. Open Web UI:
   Open browser to http://localhost:8089
   Configure:
   - Number of users: 100
   - Spawn rate: 10
   - Host: http://localhost:8000

5. Test different container configurations:
   
   # 1 container
   docker-compose up --scale api=1
   locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 10 --run-time 3m --headless --csv=results_1_container
   
   # 2 containers
   docker-compose up --scale api=2
   locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 20 --run-time 3m --headless --csv=results_2_containers
   
   # 3 containers
   docker-compose up --scale api=3
   locust -f locustfile.py --host=http://localhost:8000 --users 150 --spawn-rate 30 --run-time 3m --headless --csv=results_3_containers

EXPECTED RESULTS:

1 Container:
- RPS: ~20-30
- P95 Latency: 800-1200ms
- Failures: <1%

2 Containers:
- RPS: ~40-60
- P95 Latency: 500-800ms
- Failures: <1%

3 Containers:
- RPS: ~60-90
- P95 Latency: 350-600ms
- Failures: <1%
"""