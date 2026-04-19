import asyncio
from fastapi import FastAPI, Query
import httpx
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# Load model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

PEXELS_API_KEY = "MUMbVhWI0H3O66g6210EDbCRL9bpvUqkarsWjoMm3yD39oPK4ZbN5O3x"
STOP_WORDS = {"and", "the", "with", "for", "this", "that", "from"}


async def generate_ai_data(image_url: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=10.0)
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
        # If an image fails, return a fallback instead of crashing
        return {"ai_description": "Description unavailable", "tags": []}


@app.get("/fetch-images")
async def fetch_images(query: str = Query(...)):
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=5"
    headers = {"Authorization": PEXELS_API_KEY}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        data = resp.json()

    # Step 1: Create a list of "tasks" for the AI to do
    tasks = [generate_ai_data(photo['src']['medium']) for photo in data['photos']]

    # Step 2: Run all AI tasks at the same time!
    ai_results = await asyncio.gather(*tasks)

    # Step 3: Combine the Pexels data with your AI results
    final_results = []
    for i, photo in enumerate(data['photos']):
        final_results.append({
            "url": photo['src']['medium'],
            "ai_description": ai_results[i]["ai_description"],
            "tags": ai_results[i]["tags"]
        })

    return final_results