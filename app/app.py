import streamlit as st
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np, json, base64
from io import BytesIO
from pathlib import Path
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Hirfatuna", page_icon="🏺", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"]{background:#f8f4ef!important;font-family:'Inter',sans-serif;color:#1a1208;}
[data-testid="stAppViewContainer"]>.main{padding:0!important;}
[data-testid="stHeader"]{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
[data-testid="stVerticalBlock"]{gap:0!important;}
#MainMenu,footer,.stDeployButton{display:none!important;}
div[data-testid="column"]{padding:0!important;}
[data-testid="stHorizontalBlock"]{gap:0!important;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-thumb{background:#c8965a;border-radius:3px;}
[data-testid="stFileUploaderDropzone"]{background:white!important;border:2px dashed #d4a96a!important;border-radius:20px!important;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#c8965a!important;background:#fef9f3!important;}
[data-testid="stFileUploaderDropzone"] *{color:#8a6a45!important;}
.stSpinner>div{border-top-color:#c8965a!important;}
iframe{border-radius:16px!important;border:none!important;}
[data-testid="stTextInput"] input{
  background:white!important;border:1.5px solid #e8d0b0!important;
  border-radius:50px!important;padding:12px 20px!important;
  font-size:14px!important;color:#1a1208!important;}
[data-testid="stTextInput"] input:focus{border-color:#c8965a!important;outline:none!important;}
.stButton>button{
  background:linear-gradient(135deg,#c8965a,#a8743a)!important;
  color:white!important;border:none!important;border-radius:50px!important;
  font-weight:500!important;padding:10px 24px!important;transition:all .2s!important;
  white-space:nowrap!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(200,150,90,.35)!important;}
</style>
""", unsafe_allow_html=True)

KB_PATH = Path(__file__).parent / "craft_knowledge.json"
with open(KB_PATH, encoding="utf-8") as f:
    KB = json.load(f)
CRAFTS = {c["id"]: c for c in KB["crafts"]}
REGIONS = {r["name"]: r for r in KB["regions"]}

CLASS_LABELS = {
    "bijoux_berberes":         {"en":"Berber Jewelry",      "ar":"مجوهرات أمازيغية", "emoji":"💎","color":"#8B5E3C"},
    "bois_sculpte":            {"en":"Carved Wood",         "ar":"نحت على الخشب",    "emoji":"🪵","color":"#6B4423"},
    "broderie_tunisienne":     {"en":"Tunisian Embroidery", "ar":"التطريز التونسي",   "emoji":"🪡","color":"#B5451B"},
    "fer_forge":               {"en":"Wrought Iron",        "ar":"الحديد المطروق",    "emoji":"⚒️","color":"#2C3E50"},
    "maroquinerie_tunisienne": {"en":"Leather Craft",       "ar":"صناعة الجلود",      "emoji":"👜","color":"#8B4513"},
    "poterie_nabeul":          {"en":"Nabeul Pottery",      "ar":"فخار نابل",         "emoji":"🏺","color":"#1B6CA8"},
    "tapis_kairouan":          {"en":"Kairouan Carpet",     "ar":"سجاد القيروان",     "emoji":"🧶","color":"#7B2D8B"},
    "verre_souffle":           {"en":"Blown Glass",         "ar":"الزجاج المنفوخ",    "emoji":"🫧","color":"#1A7A4A"},
    "djebba":                  {"en":"Traditional Djebba",  "ar":"الجبة التونسية",    "emoji":"🧵","color":"#C8A44A"},
    "cuivre":                  {"en":"Copperware",          "ar":"صناعة النحاس",      "emoji":"🍯","color":"#B5651D"},
}
DEFAULT_LABEL = {"en":"Tunisian Craft","ar":"","emoji":"🏺","color":"#c8965a"}

def label_for(cid):
    return CLASS_LABELS.get(cid, DEFAULT_LABEL)

REGION_MAP_DATA = {
    "poterie_nabeul":"Nabeul","tapis_kairouan":"Kairouan",
    "broderie_tunisienne":"Monastir","bijoux_berberes":"Djerba",
    "maroquinerie_tunisienne":"Tunis","bois_sculpte":"Tunis",
    "verre_souffle":"Beja","fer_forge":"Kairouan",
    "djebba":"Tunis","cuivre":"Tunis",
}

META_PATH  = Path(__file__).parent.parent / "models" / "metadata.json"
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.pth"

@st.cache_resource
def load_model():
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, meta["num_classes"]))
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    return m, meta

tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def predict(img, model):
    t = tfm(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(t), dim=1)[0]
    i = p.argmax().item()
    return i, p.numpy(), t

def gradcam(img, tensor, model):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        np_img = np.array(img.resize((224,224))).astype(np.float32)/255.
        cam = GradCAM(model=model, target_layers=[model.features[-1]])
        return show_cam_on_image(np_img, cam(input_tensor=tensor)[0], use_rgb=True)
    except Exception:
        return None

def b64(img, size=(700,700)):
    i=img.copy(); i.thumbnail(size,Image.LANCZOS)
    buf=BytesIO(); i.save(buf,"JPEG",quality=90)
    return base64.b64encode(buf.getvalue()).decode()

try:
    model, meta = load_model()
    CLASS_NAMES = meta["class_names"]
    RMAP = meta["region_map"]
    model_ok = True
except Exception:
    model_ok = False
    CLASS_NAMES = [k for k in CLASS_LABELS if k not in ("djebba","cuivre")]
    RMAP = REGION_MAP_DATA

@st.cache_resource
def build_rag():
    try:
        from sentence_transformers import SentenceTransformer
        import faiss as faiss_lib

        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        docs, meta_list = [], []
        for cid, c in CRAFTS.items():
            text = (f"{c['name_en']} is a craft from {c['region']}, Tunisia. "
                    f"Category: {c['category']}. "
                    f"Materials: {', '.join(c['materials'])}. "
                    f"History: {c['history']} "
                    f"Types: {', '.join(c.get('types', []))}. "
                    f"Techniques: {', '.join(c.get('techniques', []))}. "
                    f"Fun fact: {c['fun_fact']} "
                    f"Time to create: {c['time_to_create']}.")
            docs.append(text)
            meta_list.append({"id": cid, "name": c["name_en"], "region": c["region"]})

        embs = embedder.encode(docs, convert_to_numpy=True).astype("float32")
        faiss_lib.normalize_L2(embs)
        index = faiss_lib.IndexFlatIP(embs.shape[1])
        index.add(embs)
        return embedder, index, docs, meta_list
    except Exception:
        return None, None, [], []

rag_embedder, rag_index, rag_docs, rag_meta = build_rag()

def _build_keyword_index():
    idx = {}
    for cid, c in CRAFTS.items():
        keys = set()
        keys.add(c["name_en"].lower())
        keys.update(c["name_en"].lower().split())
        keys.add(c["region"].lower())
        keys.add(c["category"].lower())
        keys.update(m.lower() for m in c.get("materials", []))
        keys.update(t.lower() for t in c.get("tags", []))
        for k in keys:
            idx.setdefault(k, set()).add(cid)
    return idx

KEYWORD_INDEX = _build_keyword_index()

INTENT_PATTERNS = {
    "list_by_region": ["which crafts", "what crafts", "crafts come from", "crafts from", "craft come from"],
    "materials":  ["material", "made of", "made from", "what is it made", "ingredient", "fabric", "metal", "wood used"],
    "techniques": ["technique", "how is it made", "how do they make", "how it's made", "process", "method", "craft process"],
    "time":       ["how long", "time to make", "how much time", "duration", "days", "weeks", "months to"],
    "types":      ["types", "kinds", "varieties", "different types", "categories of", "what kind"],
    "fact":       ["fun fact", "interesting", "did you know", "fact about", "surprising"],
    "region":     ["where is", "which region", "what region", "located", "from which city"],
    "history":    ["history", "origin", "where did", "old", "ancient", "story", "started", "began", "century"],
}

def _detect_intent(q):
    q = q.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        if any(p in q for p in patterns):
            return intent
    return None

def _find_crafts_in_question(question):
    q_lower = question.lower()
    matches = {}
    for word, cids in KEYWORD_INDEX.items():
        if len(word) > 2 and word in q_lower:
            for cid in cids:
                matches[cid] = matches.get(cid, 0) + len(word)
    if matches:
        ranked = sorted(matches.items(), key=lambda x: -x[1])
        return [cid for cid, _ in ranked[:2]]
    return []

def _semantic_fallback(question, k=2):
    if rag_embedder is None:
        return []
    import faiss as faiss_lib
    q_emb = rag_embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss_lib.normalize_L2(q_emb)
    _, idxs = rag_index.search(q_emb, k=k)
    return [rag_meta[i]["id"] for i in idxs[0] if 0 <= i < len(rag_meta)]

def _format_answer(cid, intent):
    c = CRAFTS.get(cid)
    if not c:
        return None
    name = c["name_en"]

    if intent == "materials":
        return f"{name} is made using: {', '.join(c['materials'])}."
    if intent == "time":
        return f"{name} takes {c['time_to_create']} to complete, depending on complexity."
    if intent == "techniques":
        return f"{name} is crafted using these techniques: {', '.join(c['techniques'])}."
    if intent == "types":
        types = c.get("types", [])
        if types:
            bullet_list = "\n".join(f"- {t}" for t in types)
            return f"{name} comes in several types:\n\n{bullet_list}"
        return f"I don't have a detailed list of types for {name} yet."
    if intent == "fact":
        return f" About {name}: {c['fun_fact']}"
    if intent == "region":
        return f"{name} comes from {c['region']}, Tunisia. {REGIONS.get(c['region'], {}).get('specialty','')}"
    if intent == "history":
        return f"{name} — {c['history']}"

    return (f"{name} ({c['region']})\n\n"
            f"{c['history']}\n"
            f" Techniques: {', '.join(c['techniques'][:3])}\n"
            f" Did you know? {c['fun_fact']}")

def answer_with_rag(question):
    q_clean = question.strip()
    if not q_clean:
        return "Please ask a question about a Tunisian craft."

    intent = _detect_intent(q_clean)

    if intent == "list_by_region":
        q_lower = q_clean.lower()
        for rname in REGIONS:
            if rname.lower() in q_lower:
                crafts_here = [c["name_en"] for cid, c in CRAFTS.items() if c["region"] == rname]
                if crafts_here:
                    bullet_list = "\n".join(f"- {name}" for name in crafts_here)
                    return f"Crafts from {rname}:\n\n{bullet_list}"

    craft_ids = _find_crafts_in_question(q_clean)

    if not craft_ids:
        craft_ids = _semantic_fallback(q_clean, k=2)

    if not craft_ids:
        all_names = ", ".join(label_for(cid)["en"] for cid in list(CRAFTS.keys())[:6])
        return f"I couldn't match that to a specific craft. Try asking about one of these: {all_names}…"

    if len(craft_ids) >= 2 and any(w in q_clean.lower() for w in ["difference", "compare", "versus", " vs "]):
        parts = []
        for cid in craft_ids[:2]:
            c = CRAFTS[cid]
            parts.append(f"{c['name_en']} ({c['region']}) — {c['history'][:200]}…")
        return "\n\n".join(parts)

    answer = _format_answer(craft_ids[0], intent)
    if answer:
        return answer
    return "I found a related craft, but couldn't compose a precise answer. Try rephrasing your question."

for key, val in [("page","scan"),("history",[]),
                 ("chat_messages",[]),("filter_cat","All")]:
    if key not in st.session_state:
        st.session_state[key] = val

st.markdown("""
<div style="background:white;border-bottom:1px solid #f0e4d0;
     padding:18px 40px;display:flex;align-items:center;
     justify-content:space-between;flex-wrap:wrap;gap:16px;
     position:sticky;top:0;z-index:999;
     box-shadow:0 2px 16px rgba(200,150,90,.08);">
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:26px;">🏺</span>
    <span style="font-family:'Playfair Display',serif;font-size:19px;
          font-weight:600;color:#1a1208;">Hirfatuna</span>
  </div>
  <span style="font-family:'Playfair Display',serif;font-size:16px;
        color:#c8b89a;font-style:italic;">حرفتنا</span>
</div>
""", unsafe_allow_html=True)

PAGES = ["Scan","Explore","Map","AI Chat","History"]

st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] div[data-testid="column"]{
  padding:0 10px!important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div style="height:48px;"></div>', unsafe_allow_html=True)


_, nav_area, _ = st.columns([1, 6, 1])

with nav_area:
    widths = []
    for i in range(len(PAGES)):
        widths.append(1)
        if i < len(PAGES) - 1:
            widths.append(0.1)

    cols = st.columns(widths)

    for i, label in enumerate(PAGES):
        with cols[i * 2]:
            if st.button(label, key=f"nav_{label}", use_container_width=True):
                st.session_state.page = label.lower()
                st.rerun()

st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)

PAGE = st.session_state.page

if PAGE == "scan":

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1208 0%,#2d1a08 50%,#0d2a1a 100%);
         padding:64px 40px;text-align:center;position:relative;overflow:hidden;">
      <div style="position:absolute;top:-80px;left:-80px;width:350px;height:350px;
           background:radial-gradient(circle,rgba(200,150,90,.12),transparent);border-radius:50%;"></div>
      <div style="position:absolute;bottom:-60px;right:-60px;width:280px;height:280px;
           background:radial-gradient(circle,rgba(26,122,74,.1),transparent);border-radius:50%;"></div>
      <h1 style="font-family:'Playfair Display',serif;
         font-size:clamp(34px,6vw,62px);font-weight:700;color:white;
         line-height:1.1;margin-bottom:18px;">
        Discover the Story Behind<br>
        <span style="background:linear-gradient(90deg,#c8965a,#e8c87a);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;">Every Tunisian Craft</span>
      </h1>
      <p style="font-size:15px;letter-spacing:2px;color:#c8965a;font-weight:500;margin-bottom:14px;">Upload a photo of any Tunisian craft and let AI reveal its story</p>
                
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:720px;margin:48px auto 0;padding:0 24px;text-align:center;">
      <h2 style="font-family:'Playfair Display',serif;font-size:28px;font-weight:600;
         color:#1a1208;margin-bottom:8px;">Upload Your Photo</h2> 
    </div>
    """, unsafe_allow_html=True)

    _,upc,_ = st.columns([1,3,1])
    with upc:
        uploaded = st.file_uploader(type=["jpg","jpeg","png"], label_visibility="visible")

    if not uploaded:
        st.markdown('<div style="height:80px;"></div>', unsafe_allow_html=True)

    if uploaded and model_ok:
        img_pil = Image.open(uploaded).convert("RGB")

        with st.spinner("Analyzing your craft object…"):
            idx, probs, tensor = predict(img_pil, model)
            pred_key = CLASS_NAMES[idx]
            conf = float(probs[idx])
            linfo = label_for(pred_key)
            region = RMAP.get(pred_key,"Tunisia")
            craft_data = CRAFTS.get(pred_key,{})
            cam = gradcam(img_pil, tensor, model)
            b64_orig = b64(img_pil)
            b64_cam = b64(Image.fromarray(cam)) if cam is not None else None
            accent = linfo.get("color","#c8965a")
            rinfo = REGIONS.get(region,{"color":"#c8965a","coords":[33.8,9.5]})

            st.session_state.history.insert(0,{"b64":b64(img_pil,(240,240)),"label":linfo["en"], "emoji":linfo["emoji"],"region":region, "confidence":conf,"key":pred_key})
            if len(st.session_state.history)>16:
                st.session_state.history = st.session_state.history[:16]

        st.markdown(f"""
        <div style="max-width:1100px;margin:40px auto 0;padding:0 32px;">
          <div style="background:linear-gradient(135deg,{accent}1a,{accent}08);
               border:2px solid {accent}33;border-radius:24px;
               padding:32px 36px;display:flex;align-items:center;
               gap:20px;flex-wrap:wrap;justify-content:center;">
            <div style="width:76px;height:76px;
                 background:linear-gradient(135deg,{accent},{accent}99);
                 border-radius:18px;display:flex;align-items:center;
                 justify-content:center;font-size:36px;flex-shrink:0;">{linfo['emoji']}</div>
            <div>
              <p style="font-size:11px;letter-spacing:1px;
                 color:{accent};font-weight:500;margin-bottom:6px;">Identified as</p>
              <h2 style="font-family:'Playfair Display',serif;
                 font-size:clamp(24px,4vw,42px);font-weight:700;
                 color:#1a1208;margin:0;line-height:1.1;">{linfo['en']}</h2>
              <p style="font-size:18px;color:{accent};margin:5px 0 0;font-style:italic;">{linfo['ar']}</p>
            </div>
            <div style="text-align:center;background:white;border-radius:14px;
                 padding:14px 22px;border:1px solid #e8d0b0;">
              <p style="font-size:10px;color:#8a6a45;margin:0 0 3px;
                 letter-spacing:1px;">Origin</p>
              <p style="font-size:20px;font-weight:600;color:#1a1208;margin:0;">📍 {region}</p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="max-width:1100px;margin:58px auto 0;padding:0 52px;">', unsafe_allow_html=True)
        c1,c2 = st.columns([1,1], gap="large")
        with c1:
            st.markdown(f"""
            <p style="font-size:11px;font-weight:600;color:#8a6a45;
               text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Your Photo</p>
            <div style="border-radius:18px;overflow:hidden;border:1px solid #e8d0b0;
                 box-shadow:0 8px 32px rgba(200,150,90,.1);">
              <img src="data:image/jpeg;base64,{b64_orig}"
                   style="width:100%;display:block;object-fit:cover;max-height:420px;">
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <p style="font-size:11px;font-weight:600;color:#8a6a45;
               text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Geographic Origin</p>
            <div style="background:white;border-radius:14px;padding:14px 18px;
                 border:1px solid #e8d0b0;margin-bottom:14px;
                 display:flex;align-items:center;gap:12px;">
              <div style="width:42px;height:42px;background:{rinfo['color']}18;
                   border-radius:10px;display:flex;align-items:center;
                   justify-content:center;font-size:20px;">📍</div>
              <div>
                <p style="font-size:17px;font-weight:600;color:#1a1208;margin:0;">{region}</p>
                <p style="font-size:12px;color:#8a6a45;margin:2px 0 0;">{rinfo.get('specialty','')}</p>
              </div>
            </div>
            """, unsafe_allow_html=True)
            mmap = folium.Map(location=[33.8,9.5], zoom_start=6, tiles="CartoDB positron")
            rc = rinfo.get("coords",[33.8,9.5])
            folium.CircleMarker(rc,radius=22,color=rinfo["color"],fill=True, fill_color=rinfo["color"],fill_opacity=.18,weight=2).add_to(mmap)
            folium.CircleMarker(rc,radius=7,color=rinfo["color"],fill=True, fill_color=rinfo["color"],fill_opacity=1,weight=0).add_to(mmap)
            st_folium(mmap, width=None, height=300, returned_objects=[])
        st.markdown('</div>', unsafe_allow_html=True)

        history_txt = craft_data.get("history","—")
        techniques = ", ".join(craft_data.get("techniques",["—"])[:4])
        fun_fact = craft_data.get("fun_fact","—")
        time_create = craft_data.get("time_to_create","—")
        materials = ", ".join(craft_data.get("materials",["—"]))
        types_list = craft_data.get("types",[])
        tags = craft_data.get("tags",[])

        tags_html = "".join(
            f'<span style="font-size:11px;background:{accent}1a;color:{accent};'
            f'padding:3px 10px;border-radius:20px;font-weight:500;margin:2px;'
            f'display:inline-block;">{tg}</span>' for tg in tags
        )
        types_html = "".join(
            f'<p style="font-size:12px;color:#6a5a45;margin:3px 0;font-weight:300;">• {tp}</p>'
            for tp in types_list[:4]
        )

        st.markdown(f"""
        <div style="max-width:1100px;margin:32px auto 0;padding:0 32px;">
          <h3 style="font-family:'Playfair Display',serif;font-size:24px;
             font-weight:600;color:#1a1208;margin-bottom:20px;">History &amp; Craftsmanship</h3>
        </div>
        """, unsafe_allow_html=True)

        _, story_area, _ = st.columns([0.04, 1.92, 0.04])
        with story_area:
            sc1, sc2, sc3 = st.columns(3, gap="medium")

            with sc1:
                st.markdown(f"""<div style="background:white;border-radius:18px;padding:24px;
                     border:1px solid #f0e4d0;box-shadow:0 4px 20px rgba(200,150,90,.06);height:100%;">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
                    <div style="width:32px;height:32px;background:{accent}18;border-radius:8px;
                         display:flex;align-items:center;justify-content:center;font-size:16px;">📖</div>
                    <p style="font-size:11px;font-weight:600;color:{accent};
                       text-transform:uppercase;letter-spacing:1px;margin:0;">History</p>
                  </div>
                  <p style="font-size:13.5px;color:#4a3a28;line-height:1.75;margin:0;font-weight:300;">{history_txt}</p>
                </div>""", unsafe_allow_html=True)

            with sc2:
                st.markdown(f"""<div style="background:white;border-radius:18px;padding:24px;
                     border:1px solid #f0e4d0;box-shadow:0 4px 20px rgba(200,150,90,.06);height:100%;">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
                    <div style="width:32px;height:32px;background:{accent}18;border-radius:8px;
                         display:flex;align-items:center;justify-content:center;font-size:16px;">🛠️</div>
                    <p style="font-size:11px;font-weight:600;color:{accent};
                       text-transform:uppercase;letter-spacing:1px;margin:0;">Techniques</p>
                  </div>
                  <p style="font-size:13.5px;color:#4a3a28;line-height:1.75;margin:0 0 16px;font-weight:300;">{techniques}</p>
                  <div style="background:{accent}0d;border-radius:10px;padding:12px 14px;">
                    <p style="font-size:10px;font-weight:600;color:{accent};
                       text-transform:uppercase;letter-spacing:1px;margin:0 0 5px;">Materials</p>
                    <p style="font-size:12px;color:#6a5a45;margin:0 0 10px;font-weight:300;">{materials}</p>
                    <p style="font-size:10px;font-weight:600;color:{accent};
                       text-transform:uppercase;letter-spacing:1px;margin:0 0 5px;">Time to Create</p>
                    <p style="font-size:12px;color:#6a5a45;margin:0;font-weight:300;">{time_create}</p>
                  </div>
                </div>""", unsafe_allow_html=True)

            with sc3:
                st.markdown(f"""<div style="background:linear-gradient(135deg,{accent}14,{accent}06);
                     border-radius:18px;padding:24px;border:1.5px solid {accent}22;
                     box-shadow:0 4px 20px rgba(200,150,90,.08);height:100%;">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
                    <div style="width:32px;height:32px;background:{accent}22;border-radius:8px;
                         display:flex;align-items:center;justify-content:center;font-size:16px;">✨</div>
                    <p style="font-size:11px;font-weight:600;color:{accent};
                       text-transform:uppercase;letter-spacing:1px;margin:0;">Did You Know?</p>
                  </div>
                  <p style="font-size:15px;font-family:'Playfair Display',serif;font-style:italic;
                     color:#1a1208;line-height:1.7;margin:0 0 16px;">&ldquo;{fun_fact}&rdquo;</p>
                  <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">{tags_html}</div>
                  {types_html}
                </div>""", unsafe_allow_html=True)

        if b64_cam:
            st.markdown(f"""
            <div style="max-width:1100px;margin:32px auto 0;padding:0 32px;">
              <h3 style="font-family:'Playfair Display',serif;font-size:24px;
                 font-weight:600;color:#1a1208;margin-bottom:8px;">How the AI Sees It</h3>
              <p style="font-size:13px;color:#8a6a45;margin:0 0 20px;font-weight:300;">
                Red zones show which parts of the image the AI focused on most
              </p>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;max-width:820px;">
                <div>
                  <p style="font-size:11px;font-weight:600;color:#8a6a45;margin:0 0 10px;
                     text-transform:uppercase;letter-spacing:1px;">Original</p>
                  <div style="border-radius:14px;overflow:hidden;border:1px solid #e8d0b0;">
                    <img src="data:image/jpeg;base64,{b64_orig}"
                         style="width:100%;display:block;object-fit:cover;max-height:280px;">
                  </div>
                </div>
                <div>
                  <p style="font-size:11px;font-weight:600;color:{accent};margin:0 0 10px;
                     text-transform:uppercase;letter-spacing:1px;">AI Focus Map</p>
                  <div style="border-radius:14px;overflow:hidden;border:1.5px solid {accent}55;">
                    <img src="data:image/jpeg;base64,{b64_cam}"
                         style="width:100%;display:block;object-fit:cover;max-height:280px;">
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        similar = CRAFTS.get(pred_key,{}).get("similar",[])
        valid_similar = [sid for sid in similar if sid in CRAFTS]
        if valid_similar:
            st.markdown(f"""
            <div style="max-width:1100px;margin:32px auto 0;padding:0 32px;">
              <h3 style="font-family:'Playfair Display',serif;font-size:24px;
                 font-weight:600;color:#1a1208;margin-bottom:8px;">You Might Also Like</h3>
            </div>
            """, unsafe_allow_html=True)

            _, rec_area, _ = st.columns([1,6,1])
            with rec_area:
                rec_cols = st.columns(len(valid_similar), gap="small")
                for col, sid in zip(rec_cols, valid_similar):
                    sc = CRAFTS.get(sid,{})
                    sl = label_for(sid)
                    with col:
                        st.markdown(f"""
                        <div style="background:white;border-radius:16px;padding:18px 20px;
                             border:1px solid #f0e4d0;text-align:center;
                             box-shadow:0 4px 16px rgba(200,150,90,.06);">
                          <div style="font-size:28px;margin-bottom:10px;">{sl['emoji']}</div>
                          <p style="font-size:14px;font-weight:600;color:#1a1208;margin:0 0 4px;">{sl['en']}</p>
                          <p style="font-size:11px;color:{sl['color']};margin:0 0 10px;font-style:italic;">{sl.get('ar','')}</p>
                          <p style="font-size:12px;color:#8a6a45;margin:0;font-weight:300;">📍 {sc.get('region','')}</p>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown('<div style="height:60px;"></div>', unsafe_allow_html=True)

    elif uploaded and not model_ok:
        st.error("Model files not found. Please make sure `models/best_model.pth` and `models/metadata.json` exist.")

elif PAGE == "explore":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1208,#2d1a08);padding:56px 40px;text-align:center;">
      <p style="font-size:14px;letter-spacing:2px;color:#c8965a;font-weight:500;margin-bottom:14px;">Cultural Heritage</p>
      <h1 style="font-family:'Playfair Display',serif;font-size:clamp(30px,5vw,52px);font-weight:700;color:white;margin:0 0 14px;">
        Explore Tunisian Crafts
      </h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="max-width:1100px;margin:52px auto 0;padding:0 52px;">', unsafe_allow_html=True)
    categories = sorted(set(c["category"] for c in CRAFTS.values()))
    nb_buttons = len(categories) + 1 

    widths = []
    for i in range(nb_buttons):
        widths.append(1)
        if i < nb_buttons - 1:
            widths.append(0.1)  

    filter_cols = st.columns(widths)

    with filter_cols[0]:
        if st.button("All", key="filt_all", use_container_width=True):
            st.session_state.filter_cat = "All"
            st.rerun()

    for i, cat in enumerate(categories):
        with filter_cols[(i + 1) * 2]:
            if st.button(cat, key=f"filt_{cat}", use_container_width=True):
                st.session_state.filter_cat = cat
                st.rerun()

    selected_cat = st.session_state.filter_cat

    for cid, c in CRAFTS.items():
        if selected_cat != "All" and c["category"] != selected_cat:
            continue
        cl = label_for(cid)
        acc = cl.get("color","#c8965a")
        types_html = "".join(f"<li style='font-size:13px;color:#4a3a28;margin-bottom:4px;font-weight:300;'>{tp}</li>" for tp in c.get("types",[]))
        mats_html = " · ".join(c.get("materials",[]))

        st.markdown(f"""
        <div style="background:white;border-radius:22px;overflow:hidden;
             border:1px solid #f0e4d0;box-shadow:0 4px 24px rgba(200,150,90,.07);
             margin-bottom:24px;margin-top:30px;">
          <div style="background:linear-gradient(135deg,{acc}1c,{acc}08);
               padding:24px 28px;border-bottom:1px solid {acc}1e;
               display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
            <div style="width:56px;height:56px;background:linear-gradient(135deg,{acc},{acc}99);
                 border-radius:14px;display:flex;align-items:center;
                 justify-content:center;font-size:26px;flex-shrink:0;">{cl['emoji']}</div>
            <div style="flex:1;">
              <h2 style="font-family:'Playfair Display',serif;font-size:22px;
                 font-weight:700;color:#1a1208;margin:0 0 3px;">{cl['en']}</h2>
              <p style="font-size:15px;color:{acc};margin:0;font-style:italic;">{cl['ar']}</p>
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
              <span style="background:{acc}18;color:{acc};font-size:12px;font-weight:600;
                    padding:6px 14px;border-radius:50px;border:1.5px solid {acc}33;">
                📍 {c['region']}
              </span>
              <span style="background:#f5ede0;color:#8a6a45;font-size:12px;font-weight:500;
                    padding:6px 14px;border-radius:50px;">
                {c['category']}
              </span>
            </div>
          </div>
          <div style="padding:24px 28px;">
            <div style="display:grid;grid-template-columns:2fr 1fr 1fr;gap:22px;">
              <div>
                <p style="font-size:10px;font-weight:600;color:{acc};text-transform:uppercase;
                   letter-spacing:1px;margin:0 0 10px;">History</p>
                <p style="font-size:13.5px;color:#4a3a28;line-height:1.75;margin:0 0 16px;font-weight:300;">{c['history']}</p>
                <div style="background:{acc}0c;border-radius:10px;padding:12px 14px;border-left:3px solid {acc};">
                  <p style="font-size:10px;font-weight:600;color:{acc};text-transform:uppercase;letter-spacing:1px;margin:0 0 5px;"> Did You Know?</p>
                  <p style="font-size:13px;font-family:'Playfair Display',serif;font-style:italic;color:#1a1208;line-height:1.6;margin:0;">&ldquo;{c['fun_fact']}&rdquo;</p>
                </div>
              </div>
              <div>
                <p style="font-size:10px;font-weight:600;color:{acc};text-transform:uppercase;letter-spacing:1px;margin:0 0 10px;">Types</p>
                <ul style="padding-left:16px;margin:0 0 16px;">{types_html}</ul>
                <p style="font-size:10px;font-weight:600;color:{acc};text-transform:uppercase;letter-spacing:1px;margin:0 0 6px;">Time to Create</p>
                <p style="font-size:12px;color:#6a5a45;font-weight:300;">{c['time_to_create']}</p>
              </div>
              <div>
                <p style="font-size:10px;font-weight:600;color:{acc};text-transform:uppercase;letter-spacing:1px;margin:0 0 10px;">Materials</p>
                <p style="font-size:12px;color:#6a5a45;line-height:1.7;margin:0 0 14px;font-weight:300;">{mats_html}</p>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div><div style="height:60px;"></div>', unsafe_allow_html=True)

elif PAGE == "map":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d2a1a,#1a1208);padding:56px 40px;text-align:center;">
      <h1 style="font-family:'Playfair Display',serif;font-size:clamp(30px,5vw,52px);font-weight:700;color:white;margin:0 0 14px;">
        Craft Regions of Tunisia
      </h1>
      <p style="font-size:14px;letter-spacing:2px;color:#c8965a;font-weight:500;margin-bottom:14px;">Click on any region to discover its craft traditions. Every dot is a living heritage.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="max-width:1100px;margin:36px auto 0;padding:0 32px;">', unsafe_allow_html=True)

    fmap = folium.Map(location=[33.8,9.5], zoom_start=6, tiles="CartoDB positron", attr="CartoDB")

    for cid,c in CRAFTS.items():
        cl = label_for(cid)
        color = cl.get("color","#c8965a")
        coords= c.get("coords",[33.8,9.5])
        popup = f"""
        <div style='font-family:Inter,sans-serif;min-width:220px;padding:6px;'>
          <h3 style='color:{color};font-size:15px;margin:0 0 4px;'>{cl['emoji']} {cl['en']}</h3>
          <p style='font-size:11px;color:#666;margin:0 0 8px;font-style:italic;'>{cl.get('ar','')}</p>
          <p style='font-size:11px;background:{color}18;color:{color};
             padding:3px 8px;border-radius:8px;display:inline-block;margin:0 0 8px;'>
             📍 {c['region']}
          </p>
          <p style='font-size:12px;color:#333;line-height:1.5;margin:0;'>
            {c['fun_fact'][:120]}...
          </p>
        </div>"""
        folium.CircleMarker(coords,radius=22,color=color,fill=True, fill_color=color,fill_opacity=.15,weight=2, popup=folium.Popup(popup,max_width=260)).add_to(fmap)
        folium.CircleMarker(coords,radius=7,color=color,fill=True, fill_color=color,fill_opacity=1,weight=0, tooltip=f"{cl['emoji']} {cl['en']}").add_to(fmap)

    st_folium(fmap, width=None, height=540, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="max-width:1100px;margin:32px auto 0;padding:0 32px;">', unsafe_allow_html=True)
    st.markdown("""
    <h2 style="font-family:'Playfair Display',serif;font-size:24px;font-weight:600;
       color:#1a1208;margin-bottom:20px;">Regions &amp; Specialties</h2>
    """, unsafe_allow_html=True)

    rcols = st.columns(4, gap="small")
    for i,(rname,rdata) in enumerate(REGIONS.items()):
        color = rdata.get("color","#c8965a")
        crafts_in_region = [label_for(cid)["en"] for cid in CRAFTS if CRAFTS.get(cid,{}).get("region")==rname]
        with rcols[i%4]:
            crafts_html = "".join(f'<p style="font-size:12px;color:#4a3a28;margin:3px 0;font-weight:300;">• {cr}</p>' for cr in crafts_in_region)
            st.markdown(f"""
            <div style="background:white;border-radius:16px;padding:20px;
                 border:1px solid #f0e4d0;margin:0 6px 16px;
                 box-shadow:0 4px 16px rgba(200,150,90,.06);">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
                <div style="width:10px;height:10px;background:{color};border-radius:50%;flex-shrink:0;"></div>
                <h3 style="font-size:15px;font-weight:600;color:#1a1208;margin:0;">{rname}</h3>
              </div>
              <p style="font-size:11px;color:#8a6a45;margin:0 0 12px;">{rdata.get('specialty','')}</p>
              <div style="border-top:1px solid #f0e4d0;padding-top:10px;">{crafts_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div><div style="height:60px;"></div>', unsafe_allow_html=True)

elif PAGE == "ai chat":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1208,#2a1a08);padding:56px 40px;text-align:center;">
      <h1 style="font-family:'Playfair Display',serif;font-size:clamp(30px,5vw,52px);font-weight:700;color:white;margin:0 0 14px;">
        Ask the Craft Expert
      </h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:820px;margin:49px auto 0;padding:0 49px;">
      <p style="font-size:12px;font-weight:600;color:#8a6a45;text-transform:uppercase;
         letter-spacing:1px;margin-bottom:28px;margin-left:300px;">Suggested questions</p>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "What are the types of Kairouan carpets?",
        "Tell me about Nabeul pottery history",
        "What materials are used in Berber jewelry?",
        "How long does it take to make a djebba?",
        "What is the difference between copper and iron crafts?",
        "Which crafts come from Djerba?",
    ]
    st.markdown('<div style="max-width:820px;margin:0 auto;padding:0 32px;">', unsafe_allow_html=True)
    sg_row1 = st.columns([1, 0.1, 1, 0.1, 1])
    st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True)
    sg_row2 = st.columns([1, 0.1, 1, 0.1, 1])
    sg_cols = [sg_row1[0], sg_row1[2], sg_row1[4], sg_row2[0], sg_row2[2], sg_row2[4]]
    for i,s in enumerate(suggestions):
        with sg_cols[i]:
            if st.button(s, key=f"sg_{i}", use_container_width=True):
                st.session_state.chat_messages.append({"role":"user","content":s})
                with st.spinner():
                    ans = answer_with_rag(s)
                st.session_state.chat_messages.append({"role":"assistant","content":ans})
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="max-width:820px;margin:38px auto 0;padding:0 38px;">', unsafe_allow_html=True)

    msgs = st.session_state.chat_messages
    if not msgs:
        st.markdown("""
        <div style="background:white;border-radius:18px;padding:32px;
             border:1px solid #f0e4d0;text-align:center;
             box-shadow:0 4px 20px rgba(200,150,90,.06);">
          <div style="font-size:48px;margin-bottom:14px;">💬</div>
          <p style="font-size:16px;color:#1a1208;font-weight:500;margin:0 0 8px;">
            Ask me anything about Tunisian crafts
          </p>
          <p style="font-size:13px;color:#8a6a45;font-weight:300;margin:0;">
            I know about carpets, pottery, jewelry, djebba, carved wood, copper, ironwork and more
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in msgs:
            is_user = msg["role"] == "user"
            align = "flex-end" if is_user else "flex-start"
            bg = "#c8965a" if is_user else "white"
            tc = "white"   if is_user else "#1a1208"
            border = "none"    if is_user else "1px solid #f0e4d0"
            icon = "👤"      if is_user else "🏺"
            content = msg["content"].replace("\n","<br>")
            st.markdown(f"""
            <div style="display:flex;justify-content:{align};margin-bottom:14px;gap:8px;
                 align-items:flex-end;">
              {'<span style="font-size:20px;">'+icon+'</span>' if not is_user else ''}
              <div style="max-width:82%;background:{bg};color:{tc};
                   border-radius:{'18px 18px 4px 18px' if is_user else '18px 18px 18px 4px'};
                   padding:14px 18px;border:{border};font-size:14px;line-height:1.65;
                   box-shadow:0 2px 12px rgba(0,0,0,.06);">
                {content}
              </div>
              {'<span style="font-size:20px;">👤</span>' if is_user else ''}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:20px;">', unsafe_allow_html=True)
    inp_col, btn_col = st.columns([5,1], gap="small")
    with inp_col:
        user_input = st.text_input("", placeholder="Ask about any Tunisian craft…", label_visibility="collapsed", key="chat_input")
    with btn_col:
        send = st.button("Send →", use_container_width=True)

    if send and user_input.strip():
        st.session_state.chat_messages.append({"role":"user","content":user_input})
        with st.spinner("Searching…"):
            ans = answer_with_rag(user_input)
        st.session_state.chat_messages.append({"role":"assistant","content":ans})
        st.rerun()

    if msgs:
        if st.button("🗑 Clear chat", key="clear_chat"):
            st.session_state.chat_messages = []; st.rerun()

    st.markdown('</div></div><div style="height:60px;"></div>', unsafe_allow_html=True)

elif PAGE == "history":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1208,#2d1a08);padding:56px 40px;text-align:center;">
      <h1 style="font-family:'Playfair Display',serif;font-size:clamp(30px,5vw,52px);font-weight:700;color:white;margin:0 0 14px;">
        Analysis History
      </h1>
      <p style="font-size:14px;letter-spacing:2px;color:#c8965a;font-weight:500;margin-bottom:14px;">Every craft object you've analyzed</p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.history
    st.markdown('<div style="max-width:1100px;margin:36px auto 0;padding:0 32px;">', unsafe_allow_html=True)

    if not history:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;">
          <div style="font-size:56px;margin-bottom:16px;">🔍</div>
          <h2 style="font-family:'Playfair Display',serif;font-size:24px;font-weight:400;color:#1a1208;margin-bottom:10px;">No analyses yet</h2>
          <p style="font-size:14px;color:#8a6a45;font-weight:300;">Go to Scan and upload your first craft image</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("→ Go to Scan"):
            st.session_state.page="scan"; st.rerun()
    else:
        st.markdown(f'<p style="font-size:14px;color:#8a6a45;margin-bottom:24px;">{len(history)} object{"s" if len(history)>1 else ""} analyzed this session</p>', unsafe_allow_html=True)
        cols = st.columns(4, gap="small")
        for i,e in enumerate(history):
            li = CLASS_LABELS.get(e["key"], DEFAULT_LABEL)
            ac = li.get("color","#c8965a")
            with cols[i%4]:
                st.markdown(f"""
                <div style="background:white;border-radius:18px;overflow:hidden;
                     border:1px solid #f0e4d0;box-shadow:0 4px 20px rgba(200,150,90,.06);
                     margin-bottom:20px;">
                  <div style="position:relative;">
                    <img src="data:image/jpeg;base64,{e['b64']}"
                         style="width:100%;height:150px;object-fit:cover;display:block;">
                  </div>
                  <div style="padding:14px;">
                    <div style="display:flex;align-items:center;gap:7px;margin-bottom:5px;">
                      <span style="font-size:16px;">{e['emoji']}</span>
                      <p style="font-size:13px;font-weight:600;color:#1a1208;margin:0;">{e['label']}</p>
                    </div>
                    <p style="font-size:11px;color:#8a6a45;margin:0;">📍 {e['region']}</p>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        _,clr,_ = st.columns([2,1,2])
        with clr:
            if st.button("🗑 Clear History"):
                st.session_state.history=[]; st.rerun()

    st.markdown('</div><div style="height:60px;"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="background:#1a1208;padding:36px 40px;text-align:center;
     border-top:1px solid rgba(200,150,90,.12);">
  <p style="font-family:'Playfair Display',serif;font-size:20px;font-weight:400;
     color:rgba(255,255,255,.6);margin:0 0 8px;font-style:italic;">
    حرفتنا · Hirfatuna
  </p>
  <p style="font-size:14px;letter-spacing:2px;color:#c8965a;font-weight:500;margin-bottom:14px;">Preserving Tunisian Craft Heritage through AI</p>
</div>
""", unsafe_allow_html=True)