"""Patch main.py to insert Redis cache check and move model_key earlier."""
import re

with open("backend/main.py", "r", encoding="utf-8") as f:
    src = f.read()

# 1. Replace the try block opener to add hash, model_key, cache check
old_block = '''    try:
        # ── Read uploaded bytes → both grayscale and BGR ──────────────────────
        contents  = await file.read()
        nparr     = np.frombuffer(contents, np.uint8)
        img_gray  = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # BGR — for color saturation check

        if img_gray is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")'''

new_block = '''    try:
        # ── Read uploaded bytes ───────────────────────────────────────────────
        contents  = await file.read()
        img_hash  = hashlib.sha256(contents).hexdigest()
        model_key = model.lower().strip()

        # ── Redis cache check ─────────────────────────────────────────────────
        cache_key = f"qucardio:{model_key}:{img_hash}"
        if redis_client is not None:
            _cached = redis_client.get(cache_key)
            if _cached:
                print(f"  [predict] CACHE HIT model={model_key}")
                return json.loads(_cached)

        # ── Decode image ──────────────────────────────────────────────────────
        nparr     = np.frombuffer(contents, np.uint8)
        img_gray  = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_gray is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")'''

if old_block in src:
    src = src.replace(old_block, new_block, 1)
    print("✅ Chunk 1 replaced (try block + cache check)")
else:
    print("❌ Chunk 1 NOT found — searching for partial match:")
    # Debug: print the first 300 chars of what we expect
    idx = src.find("contents  = await file.read()")
    print(f"  'contents = await file.read()' found at char {idx}")
    print(repr(src[idx-200:idx+100]))

# 2. Remove duplicate `model_key = model.lower().strip()` inside Step 4
old_step4 = '''        # ── Step 4: Classification ────────────────────────────────────────────
        t_clf = time.perf_counter()
        model_key = model.lower().strip()'''

new_step4 = '''        # ── Step 4: Classification ────────────────────────────────────────────
        t_clf = time.perf_counter()'''

if old_step4 in src:
    src = src.replace(old_step4, new_step4, 1)
    print("✅ Chunk 2 replaced (removed duplicate model_key)")
else:
    print("❌ Chunk 2 NOT found — checking if already fixed:")
    idx = src.find("t_clf = time.perf_counter()")
    print(repr(src[idx:idx+120]))

with open("backend/main.py", "w", encoding="utf-8") as f:
    f.write(src)

print("Done.")
