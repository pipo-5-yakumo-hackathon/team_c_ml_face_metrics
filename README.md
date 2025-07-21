readme_en = """
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
  "age": 28,
  "gender": "male",
  "skin_condition": {
    "acne": "mild",
    "dark_spots": "none",
    "wrinkles": "moderate"
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