from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
import duckdb
import torch
from PIL import Image
import io
import clip


app = FastAPI()

db_path = "photo_gallery.duckdb"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with duckdb.connect(db_path) as conn:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS photos (
        id INTEGER PRIMARY KEY,
        filename TEXT,
        image_data BLOB,
        vector DOUBLE[]
    )
    """)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_photoid START 1")
    conn.execute("INSTALL 'vss'")
    conn.execute("LOAD 'vss'")


@app.post("/image/process")
async def process_image(file: UploadFile = File(...)):
    file_data = await file.read()

    image = preprocess(Image.open(io.BytesIO(file_data)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).float().cpu().numpy()
        image_features_list = image_features[0].tolist()

    with duckdb.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO photos (id, filename, image_data, vector)
            VALUES (nextval('seq_photoid'), ?, ?, ?)
        """, (file.filename, file_data, image_features_list))

    return JSONResponse({"message": "Image uploaded"})


def validate_params(sim, limit, page, sort):
    if not (0.0 <= sim <= 1.0):
        raise HTTPException(status_code=400, detail="0.0 <= sim <= 1.0")
    if not (1 <= limit <= 100):
        raise HTTPException(status_code=400, detail="1 <= limit <= 100")
    if page < 1:
        raise HTTPException(status_code=400, detail="page >= 1")
    if sort not in ("ASC", "DESC"):
        raise HTTPException(status_code=400, detail="sort == (ASC, DESC)")


def search_photos(query_features, sim, sort, limit, page):
    offset = (page - 1) * limit
    query = f"""
        SELECT id, filename,
               list_cosine_similarity(vector, ?) AS similarity
        FROM photos
        WHERE list_cosine_similarity(vector, ?) >= ?
        ORDER BY similarity {sort}
        LIMIT ? OFFSET ?
    """

    with duckdb.connect(db_path) as conn:
        results = conn.execute(query, (query_features, query_features, sim, limit, offset)).fetchall()

    return [{"img": r[1], "similarity": r[2]} for r in results]


@app.get("/image/textSearch")
async def search_images(
    q: str = Query(...),
    sim: float = Query(0.2),
    sort: str = Query("DESC"),
    limit: int = Query(10),
    page: int = Query(1)
):
    validate_params(sim, limit, page, sort)

    text_tokens = clip.tokenize([q]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).double().cpu().numpy()[0]

    results = search_photos(text_features, sim, sort, limit, page)
    return JSONResponse(results)


@app.post("/image/imageSearch")
async def search_by_image(
    file: UploadFile = File(...),
    sim: float = Query(0.2),
    sort: str = Query("DESC"),
    limit: int = Query(10),
    page: int = Query(1)
):
    validate_params(sim, limit, page, sort)

    file_data = await file.read()
    image = preprocess(Image.open(io.BytesIO(file_data)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = model.encode_image(image).float().cpu().numpy()[0]

    results = search_photos(query_features, sim, sort, limit, page)
    return JSONResponse(results)
