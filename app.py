from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.exceptions import HTTPException
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline, DiffusionPipeline, StableDiffusionControlNetImg2ImgPipeline
from diffusers import ControlNetModel, UniPCMultistepScheduler
from transformers import pipeline
from starlette.responses import StreamingResponse
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
import io
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import requests
import cv2
import numpy as np

HF_TOKEN = os.getenv('HF_TOKEN')
app = FastAPI()

uploaded_image = None
uploaded_logo = None
generated_image = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


if torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE: ",device)

pipe = None
if device == "cpu":
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion",torch_dtype=torch.float32, use_auth_token=HF_TOKEN).to(device)
else:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion",torch_dtype=torch.float16, use_auth_token=HF_TOKEN).to(device)
#pipe1 = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map
    
@app.get("/")
async def root():
    return {"message": "Welcome to the Creating Ad Template With Stable Diffusion API!"}
    
@app.post("/uploadImage", response_class=JSONResponse)
async def upload_image(image_file:UploadFile):
    global uploaded_image

    if not image_file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file format")
        
    image_bytes = await image_file.read()
    uploaded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    return JSONResponse(content={"message": "Image uploaded successfully"})
    
@app.get("/get_image")
async def get_image(response_class=StreamingResponse):
    if uploaded_image is not None:
        # Return the uploaded image as a streaming response
        image_bytes = io.BytesIO()
        uploaded_image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        return StreamingResponse(image_bytes, media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="No image uploaded")
    
@app.post("/uploadLogo", response_class=JSONResponse)
async def upload_logo(logo_file:UploadFile):
    global uploaded_logo

    if not logo_file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file format")
        
    logo_bytes = await logo_file.read()
    uploaded_logo = Image.open(io.BytesIO(logo_bytes)).convert("RGB")
    return JSONResponse(content={"message": "Logo uploaded successfully"})
    
@app.get("/get_logo")
async def get_logo(response_class=StreamingResponse):
    if uploaded_logo is not None:
        # Return the uploaded image as a streaming response
        logo_bytes = io.BytesIO()
        uploaded_logo.save(logo_bytes, format="PNG")
        logo_bytes.seek(0)
    
        return StreamingResponse(logo_bytes, media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="No logo uploaded")
    
@app.get("/generate_new_img",response_class=StreamingResponse)
async def generate_new_img(hex_code: str, prompt: str = Query(..., description="Text prompt for image generation")):
    if uploaded_image is not None:
        try:
            
            print("Image is creating....")
            # Generate the image using the text-to-image model
            ad_prompt = f"""Your system prompt is this: {prompt} Consider your system prompt first. 
                            Then from the initial image create a new image that will attract customers to put in an (ad template) 
                            Also, use this RGB color {hex_to_rgb(hex_code)} as a tone in the image while image is still recognized as it is original."""
            print(f"uploaded image type: {type(uploaded_image)}")
            depth_estimator = pipeline("depth-estimation")
            depth_map = None
            if device=="cpu":
                depth_map = get_depth_map(uploaded_image, depth_estimator).unsqueeze(0).to(device)
                controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float32)
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
                ).to(device)

            else:
                depth_map = get_depth_map(uploaded_image, depth_estimator).unsqueeze(0).half().to(device)
                controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
                ).to(device)
                pipe.enable_model_cpu_offload()
                
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            
            image_bytes = io.BytesIO()
            uploaded_image.save(image_bytes, format="PNG")
            image_bytes.seek(0)
            init_image = Image.open(image_bytes).convert("RGB")
            
            print(f"Image bytes length: {len(image_bytes.getvalue())}")
            print(f"init_image type: {type(init_image)}")

            image = pipe(
                ad_prompt, image=init_image, control_image=depth_map
            ).images[0]
            print(f"image type: {type(image)}")
            
            print("Image created")
            image_data = io.BytesIO()
            image.save(image_data, format="PNG")
            image_data.seek(0)

            global generated_image
            generated_image = image
            return StreamingResponse(image_data, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return PlainTextResponse("You have not uploaded the image!")

@app.get("/create_ad_template",response_class=StreamingResponse)
async def create_ad_template(punchline: str, punchline_color:str, button_text:str,button_color:str):
    
    if uploaded_logo is not None and generated_image is not None:
        # Create LOGO
        print("Drawing")
        #logo_width, logo_height = logo_image.size
        logo_width, logo_height = 100, 100  # Desired size for the logo
        logo_image_resized = uploaded_logo.resize((logo_width, logo_height))
        logo_image_location = ((800 - logo_width) // 2,10)
        # Create Generataed Image
        # Oku ve okuduktan sonra Image ile oku
        generated_image_width, generated_image_height = 350, 350  # Desired size for the generated image
        generated_image_resized = generated_image.resize((generated_image_width, generated_image_height))
        generated_image_center_x = (800 - generated_image_width) // 2
        generated_image_center_y = (600 - generated_image_height) // 2
        generated_image_location = (generated_image_center_x, generated_image_center_y)
        
        # Create a blank canvas for the ad template
        ad_template = Image.new("RGB", (800, 600), "#FFFFFF")
        print("Template created")
        # Add logo
        ad_template.paste(logo_image_resized, logo_image_location)
        print("Logo added")
        # Add generated image
        ad_template.paste(generated_image_resized, generated_image_location)
        print("Image added")
        # Add the text at the bottom of the ad template
        draw = ImageDraw.Draw(ad_template)
        font = ImageFont.load_default()
        text_width, text_height = draw.textsize(punchline, font=font)
        text_position = (400 - text_width / 2, 500 - text_height / 2)
        draw.text(text_position, punchline, font=font, fill=punchline_color)
        print("Punchline added")
        # Add the button at the bottom of the ad template
        button_width, button_height = 200, 50
        button_position = ((400 - button_width / 2), (550 - button_height / 2))
        button_positions = [button_position[0], button_position[1], button_position[0] + button_width, button_position[1] + button_height]
        draw.rectangle(button_positions, fill=button_color)
        text_width, text_height = draw.textsize(button_text, font=font)
        text_position = (button_position[0] + (button_width - text_width) / 2, button_position[1] + (button_height - text_height) / 2)
        rect_text_color = (255, 255, 255)  # Text color within the rectangle
        draw.text(text_position, button_text, fill=rect_text_color, font=font)
        
        border_width = 2  # Border width
        border_positions = [0, 0, 800, 600]
        draw.rectangle(border_positions, outline="#D60D0D", width=border_width)
        print("Button added")
        ad_template_data = io.BytesIO()
        ad_template.save(ad_template_data, format="PNG")
        ad_template_data.seek(0)
        print("Template finished")
        return StreamingResponse(ad_template_data, media_type="image/png")
        
    else:
        StreamingResponse("If you did not generate the new image or if you did not upload logo image. You should check and try again...", media_type="text/plain")
