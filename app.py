from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import duckdb
import os
import shutil
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

uploads_dir = "uploads"
os.makedirs(uploads_dir, exist_ok=True)


@app.post("/image/process")
async def process_image(file: UploadFile = File(...)):
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

    return JSONResponse({"message": "Image uploaded", "id": new_id})


@app.get("/image/textSearch")
async def search_images(
    q: str = Query(...),
    sim: float = Query(0.2),
    sort: str = Query("DESC"),
    limit: int = Query(10),
    page: int = Query(1)
):
    text_tokens = clip.tokenize([q]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    photos = conn.execute("SELECT id, filename, vector FROM photos").fetchall()

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
        {"img": photo_info[i][1], "similarity": float(similarities[i])}
        for i in range(len(photo_info))
        if similarities[i] >= sim
    ]

    results = sorted(results, key=lambda x: x["similarity"], reverse=(sort == "DESC"))
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_results = results[start_index:end_index]

    return JSONResponse({"results": paginated_results})


@app.post("/image/imageSearch")
async def search_by_image(
    file: UploadFile = File(...),
    sim: float = Query(0.2),
    sort: str = Query("DESC"),
    limit: int = Query(10),
    page: int = Query(1)
):
    uploads_dir = "uploads/temp"
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = os.path.join(uploads_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = preprocess(Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = model.encode_image(image).float()
        query_features /= query_features.norm(dim=-1, keepdim=True)

    photos = conn.execute("SELECT id, filename, vector FROM photos").fetchall()

    image_vectors = []
    photo_info = []
    for photo_id, filename, image_vector in photos:
        vector = np.frombuffer(image_vector, dtype=np.float32)
        image_vectors.append(vector)
        photo_info.append((photo_id, filename))

    image_features = torch.tensor(np.stack(image_vectors)).to(device)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarities = image_features @ query_features.T
    similarities = similarities.squeeze(1).cpu().numpy()

    results = [
        {"img": photo_info[i][1], "similarity": float(similarities[i])}
        for i in range(len(photo_info))
        if similarities[i] >= sim
    ]
    results = sorted(results, key=lambda x: x["similarity"], reverse=(sort == "DESC"))
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_results = results[start_index:end_index]

    return JSONResponse({"results": paginated_results})
