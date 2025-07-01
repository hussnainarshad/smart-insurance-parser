import os
import base64
import numpy as np
import cv2
from fastapi import Depends, HTTPException ,FastAPI
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn
import io
from PIL import Image

load_dotenv()

app = FastAPI()

# Authentication
BEARER_TOKEN = os.getenv('Bearer_Token')

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing token")

# PaddleOCR initialization
model_base_path = os.path.join(os.getcwd(), 'models')

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Request model
class imageBase64Request(BaseModel):
    base64_img: str

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Perform OCR on an image file (in-memory bytes).

    Args:
        image_bytes (bytes): The image file content in bytes (e.g., JPEG, PNG).

    Returns:
        str: Extracted text from the image.
    """
    try:
        # Load image using PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert to NumPy array and then to OpenCV format (BGR)
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Run OCR
        results = ocr.ocr(image_cv2, cls=True)

        # Extract and concatenate text
        if results and results[0]:
            text = "\n".join([line[1][0] for line in results[0]])
            return text
        else:
            return ""
    except Exception as e:
        raise RuntimeError(f"OCR image processing failed: {e}")



def insurance_details(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
You are a highly accurate and intelligent parser specialized in extracting structured insurance information from OCR-processed images of insurance cards. Your task is to analyze the provided OCR text and return all relevant member and plan details in JSON format.

### Instructions:
- Extract **only** relevant insurance-related information from the OCR text.
- The final output should be a JSON object with a single top-level key: `"insuranceInfo"`.
- If any field is missing or unclear in the text, use `null` for that field.
- All dates should be formatted as `YYYY`.

### Extract the following fields:
- `memberName`: Full name of the insured member.
- `address`: Member's residential address, if available.
- `memberId`: Unique member identification number.
- `yearOfInsurance`: Year the insurance coverage is valid for (typically based on start or issue date).
- `companyName`: Name of the insurance company.
- `plan`: Name or type of the insurance plan.

### OCR Text:
{text}

### Expected Output Format:
Return a valid JSON object with the following structure:

```json
{{
  "insuranceInfo": {{
    "memberName": "John Doe",
    "address": "1234 Elm Street, Springfield, IL 62704",
    "memberId": "A123456789",
    "yearOfInsurance": "2025",
    "companyName": "HealthSecure Inc.",
    "plan": "Platinum PPO"
  }}
}}
""",
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content


@app.post("/extract-insurance", dependencies=[Depends(verify_token)])
async def extract_insurance_info(request: imageBase64Request):
    try:
        # Decode image from base64
        image_data = base64.b64decode(request.base64_img)

        # Perform OCR on the image
        ocr_text = extract_text_from_image(image_data)

        # Extract structured insurance details using GPT
        result = insurance_details(ocr_text)  # Consider renaming this to reflect insurance extraction
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
