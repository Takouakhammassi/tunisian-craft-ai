import base64
from io import BytesIO

from openai import OpenAI
from PIL import Image

# Qwen's vlm hosted via Hugging Face Inference Providers
VLM_MODEL_ID = "Qwen/Qwen3.5-122B-A10B:deepinfra"

GUARD_PROMPT = (
    "Does this image show a handcrafted artisanal object — pottery, a "
    "carpet, jewelry, leatherwork, carved wood, copper or wrought iron "
    "metalwork, embroidery, blown glass, or a traditional garment such "
    "as a djebba? "
    "The object is still a valid match even if a person is wearing, "
    "holding, or modeling it (for example, a model wearing an embroidered "
    "traditional dress counts as showing the craft) — focus on whether "
    "the craft item itself is clearly visible, not on whether a person "
    "is also in the frame. "
    "Do not list or evaluate each category. Do not explain your reasoning. "
    "Respond with only one line, nothing else: 'VERDICT: YES' or 'VERDICT: NO'."
)

NOT_A_CRAFT_MESSAGE = "I can't identify this image, this is not a craft photo."


def _pil_image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_verdict(raw_text: str) -> str:
    #Extract a YES/NO verdict from the model's response
    text_upper = raw_text.upper()

    last_verdict_yes = text_upper.rfind("VERDICT: YES")
    last_verdict_no = text_upper.rfind("VERDICT: NO")

    if last_verdict_yes != -1 or last_verdict_no != -1:
        return "YES" if last_verdict_yes > last_verdict_no else "NO"

    last_lines = "\n".join(raw_text.strip().splitlines()[-3:]).upper()
    last_yes = last_lines.rfind("YES")
    last_no = last_lines.rfind("NO")
    if last_yes == -1 and last_no == -1:
        return "UNKNOWN"
    return "YES" if last_yes > last_no else "NO"


def is_craft_image(image: Image.Image, hf_token: str) -> tuple[bool, bool]:
    #Checks whether the VLM guard judges the image to show a handcrafted artisanal object

    try:
        client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
        response = client.chat.completions.create(
            model=VLM_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _pil_image_to_data_url(image)}},
                        {"type": "text", "text": GUARD_PROMPT},
                    ],
                }
            ],
            # to ensure the final VERDICT line is never cut off mid-generation
            max_tokens=600,
        )
        raw_text = response.choices[0].message.content.strip()
        return _extract_verdict(raw_text) == "YES", True
    except Exception:
        # Fail open
        return True, False