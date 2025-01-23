from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import duckdb
import torch
from PIL import Image
import io
import clip
# flake8: noqa: E501

app = FastAPI()

db_path = "photo_gallery.duckdb"
conn = duckdb.connect(db_path)
conn.execute("""
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER,
    filename TEXT,
    image_data BLOB,
    vector DOUBLE[]
)
""")

conn.execute("INSTALL 'vss'")
conn.execute("LOAD 'vss'")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


@app.post("/image/process")
async def process_image(file: UploadFile = File(...)):
    file_data = await file.read()

    image = preprocess(Image.open(io.BytesIO(file_data)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).float().cpu().numpy()
        image_features_list = image_features[0].tolist()

    result = conn.execute("SELECT MAX(id) FROM photos").fetchone()
    new_id = (result[0] or 0) + 1

    conn.execute("INSERT INTO photos (id, filename, image_data, vector) VALUES (?, ?, ?, ?)",
                 (new_id, file.filename, file_data, image_features_list))

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
        text_features = model.encode_text(text_tokens).double().cpu().numpy()

    offset = (page - 1) * limit
    query = f"""
        SELECT id, filename,
               list_cosine_similarity(vector, ?) AS similarity
        FROM photos
        WHERE list_cosine_similarity(vector, ?) >= ?
        ORDER BY similarity {sort}
        LIMIT ? OFFSET ?
    """
    results = conn.execute(query, (text_features[0], text_features[0], sim, limit, offset)).fetchall()

    return JSONResponse([{"img": r[1], "similarity": r[2]} for r in results])


@app.post("/image/imageSearch")
async def search_by_image(
    file: UploadFile = File(...),
    sim: float = Query(0.2),
    sort: str = Query("DESC"),
    limit: int = Query(10),
    page: int = Query(1)
):
    file_data = await file.read()

    image = preprocess(Image.open(io.BytesIO(file_data)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = model.encode_image(image).float().cpu().numpy()

    offset = (page - 1) * limit
    query = f"""
        SELECT id, filename,
               list_cosine_similarity(vector, ?) AS similarity
        FROM photos
        WHERE list_cosine_similarity(vector, ?) >= ?
        ORDER BY similarity {sort}
        LIMIT ? OFFSET ?
    """
    results = conn.execute(query, (query_features[0], query_features[0], sim, limit, offset)).fetchall()

    return JSONResponse([{"img": r[1], "similarity": r[2]} for r in results])
