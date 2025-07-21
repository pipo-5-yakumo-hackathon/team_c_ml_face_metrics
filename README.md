# Face Analyzer

A FastAPI service for facial photo analysis. Supports age, gender, and skin condition estimation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-analyzer.git
   cd face-analyzer
   ```

2. Build the Docker image:
   ```bash
   docker build -t face-analyzer .
   ```

3. Run the container:
   ```bash
   docker run -p 8000:8000 face-analyzer
   ```

Or run locally in development mode:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

### `POST /analyze`

Upload a face photo and receive analysis.

#### Form parameters:

| Field | Type | Required | Description                 |
|-------|------|----------|-----------------------------|
| image | file | ✅ Yes   | Photo of a person (JPG/PNG) |

#### Example request:

```bash
curl -X POST -F "image=@photo.jpg" http://localhost:8000/analyze
```

#### Example response (`application/json`):

```json
{
  "emotions": {
    "angry": 0.000670479261316359,
    "disgust": 1.1489969864442173e-8,
    "fear": 0.006800354458391666,
    "happy": 7.611642837524414,
    "sad": 6.984113693237305,
    "surprise": 0.00004583375630318187,
    "neutral": 85.396728515625
  },
  "metrics": {
    "age": {
      "value": 29,
      "status": "ok"
    },
    "skin_redness_rgb": {
      "value": 149.3,
      "status": "normal"
    },
    "brightness_l": {
      "value": 51,
      "status": "normal"
    },
    "red_green_a": {
      "value": 11.2,
      "status": "normal"
    },
    "blue_yellow_b": {
      "value": 14,
      "status": "normal"
    },
    "contrast": {
      "value": 61.5,
      "status": "normal"
    },
    "hydration": {
      "value": 281.5,
      "status": "normal"
    }
  }
}
```

### `GET /health`

Service health check.

```bash
curl http://localhost:8000/health
```

#### Response:

```json
{"status": "ok"}
```

## 🧪 Typical Usage Flow

1. Take a well-lit photo of a face.
2. Send the file via `curl` or any HTTP client.
3. Receive a JSON response with analysis results.


## Author

Ilia Vorobev, 2025  
"""