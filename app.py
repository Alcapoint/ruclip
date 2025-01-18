from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import duckdb
import shutil
import os
import torch
import numpy as np
from PIL import Image
import clip

app = FastAPI()

db_path = "photo_gallery.duckdb"
conn = duckdb.connect(db_path)
conn.execute("""
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER,
    filename TEXT,
    vector BLOB
)
""")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload/")
async def upload_photo(file: UploadFile = File(...)):
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = preprocess(Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).float().cpu().numpy()

    result = conn.execute("SELECT MAX(id) FROM photos").fetchone()
    new_id = (result[0] or 0) + 1

    conn.execute("INSERT INTO photos (id, filename, vector) VALUES (?, ?, ?)",
                 (new_id, file.filename, image_features.tobytes()))

    return RedirectResponse(url="/gallery/", status_code=303)


@app.get("/gallery/")
async def gallery_page(request: Request, query: str = None):
    photos = conn.execute("SELECT id, filename, vector FROM photos").fetchall()

    if query:
        text_tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        image_vectors = []
        photo_info = []
        for photo_id, filename, image_vector in photos:
            vector = np.frombuffer(image_vector, dtype=np.float32)
            image_vectors.append(vector)
            photo_info.append((photo_id, filename))

        image_features = torch.tensor(np.stack(image_vectors)).to(device)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarities = image_features @ text_features.T
        similarities = similarities.squeeze(1).cpu().numpy()

        results = [
            (photo_info[i][0], photo_info[i][1], similarities[i])
            for i in range(len(photo_info))
        ]
        photos = sorted(results, key=lambda x: x[2], reverse=True)
    else:
        photos = [(photo[0], photo[1], None) for photo in photos]

    return templates.TemplateResponse("gallery.html", {"request": request, "photos": photos, "query": query or ""})
