# AI Image Analysis Service (ImageCrawler Sub-Service)

An asynchronous microservice built with **FastAPI**, **Hugging Face Transformers (BLIP)**, and **HTTPX**. This service acts as a specialized backend utility for a larger university project ecosystem (**Image Crawler**). It dynamically fetches high-quality media assets from external stock registries and uses computer vision models to generate automated semantic captions and descriptive tags.

---

## University Project Context
> **Note:** This repository represents a specific sub-service (`python-service`) designed to plug into a broader **Image Crawler** application developed for an academic course assignment. Its core focus is handling asynchronous downstream external API communication and running local Machine Learning inference workloads.

---

## Key Features

- **Asynchronous External Fetching:** Leverages `httpx` to handle non-blocking concurrent requests to the Pexels API registry.
- **On-the-Fly Image Analysis:** Integrates Salesforce's `BLIP` (Bootstrapped Language-Image Pre-training) model natively to generate contextual text descriptions from raw visual matrices.
- **Auto-Tag Generation:** Evaluates generated image captions, strips down filler text or stop-words, and isolates unique keyword tags dynamically.
- **Local & Remote Pipeline:** Built to cleanly handle both remote HTTP image resources and local disk paths (`uploads/`).

---

## Project Structure

Your repository follows this exact alignment:

```text
ImageCrawler/
│
├── python-service/
│   ├── main.py              # FastAPI core endpoints, AI pipeline, and routing
│   └── requirements.txt     # Python framework and deep learning dependencies
│
├── .gitignore               # Excludes python virtual environments & IDE caches
└── README.md                # Project documentation

```

---

## Architecture & Data Workflow

```text
 [Client/Main App] ──────>  GET /fetch-images?query=...  ──────>  [python-service]
                                                                        │
   ┌────────────────────────────────────────────────────────────────────┘
   ▼
1. Fetch 20 Media References ──> ( Pexels API )
2. Process/Analyze Images    ──> ( PIL Image Parsing )
3. Local ML Inference Engine ──> ( Salesforce BLIP Base Model ) ──> Return AI Captions & Tags

```

---

## Installation & Local Setup

### 1. Prerequisites

* Python 3.9 or higher is recommended due to torch/transformers dependency requirements.

### 2. Move Into Service Directory

```bash
cd python-service

```

### 3. Create & Activate Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

```

### 4. Install Dependencies

```bash
pip install -r requirements.txt

```

*Note: The environment requires `fastapi`, `uvicorn`, `httpx`, `pillow`, `transformers`, and a deep-learning backend like `torch`.*

---

## Running the Application

Launch the development server via Uvicorn from inside the `python-service` directory:

```bash
uvicorn main:app --reload

```

The server will boot up natively at **`http://127.0.0.1:8000`**.

* **Interactive Swagger UI:** Visit `http://127.0.0.1:8000/docs` to test endpoints manually through your web browser.

---

## 📡 API Endpoints Summary

### 1. Fetch Images by Query

* **Endpoint:** `GET /fetch-images`
* **Query Parameters:** `query` (string)
* **Description:** Hits the Pexels registry for curated photos matching your search terms.

### 2. Analyze Single Image

* **Endpoint:** `GET /analyze-image`
* **Query Parameters:** `image_url` (string)
* **Description:** Accepts a web address or local relative system path, passes it through the transformers engine, and extracts text context.
* **Response Payload Structure:**

```json
{
  "ai_description": "A close up photo of a mechanical keyboard on a wooden desk",
  "tags": ["close", "mechanical", "keyboard", "wooden", "desk"]
}

```

---

## Future Improvements (Roadmap)

* Uncomment and stabilize the `asyncio.gather(*tasks)` block within the `/fetch-images` endpoint to execute parallel batch AI processing across all 20 retrieved assets concurrently.
* Shift hardcoded API authorization keys to a secure configuration management format or system environment variable (`.env`).

```

```
