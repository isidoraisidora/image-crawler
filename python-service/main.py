import asyncio
from fastapi import FastAPI, Query
import httpx, os
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse

app = FastAPI()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")
STOP_WORDS = {"and", "the", "with", "for", "this", "that", "from"}


@app.get("/", response_class=HTMLResponse)
async def read_frontend():
    html_file = BASE_DIR.parent / "frontend" / "index.html"

    if not html_file.exists():
        return f"<h1>Configuration Error</h1><p>Backend could not find the HTML file at: {html_file}</p>"

    return html_file.read_text(encoding="utf-8")


async def local_analyze_pipeline(image_url: str):
    try:
        if image_url.startswith("uploads") or os.path.exists(image_url):
            img = Image.open(image_url).convert("RGB")
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url, timeout=15.0)
                img = Image.open(BytesIO(response.content)).convert("RGB")

        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=20)
        raw_caption = processor.decode(out[0], skip_special_tokens=True)

        formatted_caption = raw_caption.capitalize()
        tags = list(set([
            word.lower() for word in raw_caption.split(' ')
            if len(word) > 2 and word not in STOP_WORDS
        ]))

        return {"ai_description": formatted_caption, "tags": tags}
    except Exception as e:
        print(f"AI Error: {e}")
        return {"ai_description": "Description unavailable", "tags": []}


@app.get("/analyze-image")
async def generate_ai_data(image_url: str):
    return await local_analyze_pipeline(image_url)


@app.get("/fetch-images")
async def fetch_images(query: str = Query(...)):
    if not PEXELS_API_KEY:
        return {"error": "Incorrect or missing API key in .env file"}

    # Keeping it at 6 images so your laptop CPU runs smoothly!
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=6"
    headers = {"Authorization": PEXELS_API_KEY}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return {"error": f"Pexels API error: Status {resp.status_code}"}
        data = resp.json()

    if "photos" not in data:
        return {"error": "Invalid response from Pexels structure"}

    # Process all images concurrently with the AI pipeline
    tasks = [local_analyze_pipeline(photo['src']['medium']) for photo in data['photos']]
    ai_results = await asyncio.gather(*tasks)

    final_results = []
    for i, photo in enumerate(data['photos']):
        final_results.append({
            "url": photo['src']['medium'],
            "ai_description": ai_results[i]["ai_description"],
            "tags": ai_results[i]["tags"]
        })

    return final_results