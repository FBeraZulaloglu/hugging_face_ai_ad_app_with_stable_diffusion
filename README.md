---
title: Create Ai Ad
emoji: 🐢
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Creating Ad Template with Stable Diffusion

[Project Description: This project is about creating a dynamic ad template with stable diffusion's Img2Img algorithm.]

## Table of Contents
- [Overview]
- [How it Works]
- [Installation]
- [Error Notes]
- [What have I learned]
- [Sources]

## Overview
[Provide a brief overview of your project. What does it do? Why is it useful?]
This project makes people who creates ad templates in photoshop or from another softwares life easier. If you want to create uncomplex but creative and goodlooking
template then you shouldn't miss this application. You can easily use your FastAPI interface and create the template that you want.

## How it Works

* First go to this link: https://farukbera-create-ai-ad.hf.space/docs#/
* Then use uploadImage post function to upload an base image to create a new image with stable diffusion.
* You have to use uploadIcon function to put at the beginning in your last ad template. Otherwise you can not create that template.
* The main part is generate_new_img function which is a get function when you run that function it will take time and return you with a new image.
  To run that function you should give your idea about the image in the prompt section and give hex code of color that you would like to be in that image.
  When you run that function stable diffusion model will use your idea in your prompt and the color that you gave while creating and image.

## Installation
git clone https://huggingface.co/spaces/farukbera/create_ai_ad<br>

## Error Notes
Error1: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
Solution: add requriments.txt opencv-headless-python instead of opencv<br>
Error2: I Run stable diffusion,It`s wrong RuntimeError: "LayerNormKernelImpl" not implemented for 'Half',
Solution: make torch_dtype=torch.float16 to torch_dtype=torch.float32 if you are using CPU. If you use cuda then float16 is ok!
Solution: cuda version of a spesific function should be this: get_depth_map(image, depth_estimator).unsqueeze(0).half().to("cuda")
However, if you are using cpu then you should update like this: get_depth_map(image, depth_estimator).unsqueeze(0).to("cpu")<br>
Error3: Sphinx raises 'ImageDraw' object has no attribute 'textsize' error
Solution: Change Pillow to Pillow==9.5.0 in requirements file then use this code to make sure it works: font = ImageFont.load_default()

## What have I learned
* How to use docker fastAPI and huggingFace all together.
* POST function and GET function difference usage. For POST I am just uploading data but for GET also I am fetching the data.
* Difference between StableDiffusionImg2ImgPipeline and StableDiffusionControlNetPipeline. For the base stable diffusion I tried many prompts to
create the image that I want. But, mostly the model did not listen me. Then I learned there is a more suitiable way to use prompt with controlNet.
In this way I have more control over image generation with my prompt.
* I learned another prompt engineering trick which is instead of using HEX to set the color for the image I have used RGB version of the HEX color.
In this way I observed diffusion model's is more accurate and is more likely to success as I want.

## Sources
https://stackoverflow.com/questions/75641074/i-run-stable-diffusion-its-wrong-runtimeerror-layernormkernelimpl-not-implem
https://github.com/pytorch/pytorch/issues/52291
https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
https://review.opendev.org/c/openstack/tacker-specs/+/887772
https://huggingface.co/docs/diffusers/using-diffusers/controlnet
https://huggingface.co/docs/diffusers/using-diffusers/img2img
https://github.com/katanaml/sparrow/blob/main/sparrow-ml/api/endpoints.py
https://www.youtube.com/watch?v=0v9ZsleUuEg<br>
Lastly of course: ChatGPT :)