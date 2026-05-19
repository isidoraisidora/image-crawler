# AI Image Analysis Studio (ImageCrawler Sub-Service)

An asynchronous microservice built with **FastAPI**, **Hugging Face Transformers (BLIP)**, and **HTTPX**. This service acts as an AI utility for a larger university project ecosystem (**Image Crawler**). It dynamically fetches media assets from the Pexels API registry and runs local deep-learning inference to generate automated captions and tags.

The application features an ultra-modern, bright, responsive split-screen dashboard workspace built with HTML5, JavaScript, and Bootstrap 5.

---

## University Project Context
> **Note:** This repository represents a specific sub-service (`python-service`) designed to plug into a broader **Image Crawler** application developed for an academic course assignment.

---

## Project Directory Layout

The workspace is organized with the frontend assets decoupled from the backend pipeline environment:

```text
ImageCrawler/
│
├── frontend/
│   └── index.html           # Bright, interactive split-screen dashboard layout
│
├── python-service/
│   ├── .env                 # Local environment file containing your Pexels API key
│   ├── main.py              # FastAPI core routing, ML pipeline, and template engine
│   └── requirements.txt     # Framework and machine learning dependencies
│
└── .gitignore               # Excludes virtual environments and hidden system files
🔧 Installation & Local Setup
1. Open your Terminal inside PyCharm
Ensure your terminal session path is pointing to the root workspace directory (ImageCrawler).

2. Create & Activate your Virtual Environment
Bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Move Into the Service Directory and Install Dependencies
Bash
cd python-service
pip install -r requirements.txt
Note: Dependencies include fastapi, uvicorn, httpx, pillow, transformers, torch, and python-dotenv.

4. Configure your API Credentials
Inside the python-service/ directory, verify that your .env file contains your valid endpoint authentication key without spaces or quotation marks:

Plaintext
PEXELS_API_KEY=MUMbVhWI0H3O66g6210EDbCRL9bpvUqkarsWjoMm3yD39oPK4ZbN5O3x
 How to Run the Project Safely
 CRITICAL REQUIREMENT: Never double-click index.html or use PyCharm's built-in floating browser preview tools. If you open the file directly, the frontend will fail to communicate with the Python ML models and will throw 404 Not Found or SyntaxError script execution blocks.

Step 1: Start the FastAPI Server Engine
From inside the python-service folder directory, spin up your Uvicorn host:

Bash
uvicorn main:app --reload
Your terminal console will display confirmation logs that look like this:
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

(If you are working on your main project simultaneously and need to free up network port conflicts, you can forcefully map this service to port 8001 instead using: uvicorn main:app --reload --port 8001)

Step 2: Access the Application via your Web Browser
Open a brand-new tab in your web browser (Chrome, Edge, Firefox, etc.) and type the Uvicorn address manually into the top URL bar:

 http://127.0.0.1:8000/

FastAPI acts as your production web host. It will read the code from frontend/index.html and serve it to your browser directly on port 8000.

 Core API Summary
GET / (HTMLResponse): Reads and compiles your bright HTML interface dashboard into the client viewport dynamically.

GET /fetch-images?query={search_term} (JSON): Queries Pexels for high-quality photos, passes those image vectors concurrently into Salesforce's BLIP-base vision pipeline, and builds data response blocks matching the following payload contract structure:

JSON
[
  {
    "url": "[https://images.pexels.com/photos/](https://images.pexels.com/photos/)...",
    "ai_description": "A close up photo of a mechanical keyboard on a wooden desk",
    "tags": ["close", "mechanical", "keyboard", "wooden", "desk"]
  }
]