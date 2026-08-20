import csv
import os
import random

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

MODEL_PATH = "./models/stable-diffusion-v1-5"
INPUT_IMAGE = "./dataset/references/class_B/B.jpg"
OUTPUT_DIR = "./dataset/synthetic/class_B/B"

COUNT = 500
STRENGTH = 0.20

AGES = [
    "young adult appearance",
    "slightly younger adult appearance",
    "current age appearance",
    "slightly older adult appearance",
    "middle-aged appearance",
    "older adult appearance",
]

BUILDS = [
    "slender natural build",
    "average natural build",
    "slightly broader build",
    "slightly heavier natural build",
]

SCALES = [
    "close-up portrait framing",
    "medium portrait framing",
    "farther portrait framing",
    "head and shoulders framing",
]

LIGHTING = [
    "natural morning daylight",
    "soft afternoon daylight",
    "bright indoor lighting",
    "warm indoor lighting",
    "cool indoor lighting",
    "soft window lighting",
    "dim indoor lighting",
    "overcast outdoor lighting",
]

POSES = [
    "front-facing pose",
    "slightly turned to the left",
    "slightly turned to the right",
    "slightly looking upward",
    "slightly looking downward",
    "mild three-quarter view",
]

BACKGROUNDS = [
    "plain indoor wall",
    "subtle office background",
    "classroom background",
    "library background",
    "corridor background",
    "softly blurred indoor background",
    "natural outdoor background",
]

EXPRESSIONS = [
    "neutral relaxed expression",
    "subtle natural smile",
    "calm expression",
    "slightly serious expression",
]

NEGATIVE_PROMPT = (
    "different person, another person, multiple people, "
    "deformed face, distorted face, malformed eyes, malformed mouth, "
    "extra face, duplicate person, asymmetrical face, "
    "blurry face, cartoon, illustration, painting, "
    "unrealistic skin, extreme pose, extreme aging, low quality"
)

device = "cuda"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    local_files_only=True,
)

pipe = pipe.to(device)
pipe.enable_attention_slicing()

os.makedirs(OUTPUT_DIR, exist_ok=True)

input_image = Image.open(INPUT_IMAGE).convert("RGB")
input_image = input_image.resize((512, 512))

metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")

with open(metadata_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "image",
        "identity",
        "age",
        "build",
        "scale",
        "lighting",
        "pose",
        "background",
        "expression",
        "seed",
    ])

    for i in range(COUNT):

        age = random.choice(AGES)
        build = random.choice(BUILDS)
        scale = random.choice(SCALES)
        lighting = random.choice(LIGHTING)
        pose = random.choice(POSES)
        background = random.choice(BACKGROUNDS)
        expression = random.choice(EXPRESSIONS)

        prompt = (
            "photorealistic portrait photograph of the same person, "
            f"{age}, {build}, {scale}, {lighting}, {pose}, "
            f"{background}, {expression}, "
            "realistic facial features, natural skin texture, "
            "realistic photography, high quality"
        )

        seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=input_image,
            strength=STRENGTH,
            guidance_scale=7.0,
            num_inference_steps=30,
            generator=generator,
        ).images[0]

        filename = f"B_{i + 1:04d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        result.save(filepath)

        writer.writerow([
            filename,
            "B",
            age,
            build,
            scale,
            lighting,
            pose,
            background,
            expression,
            seed,
        ])

        print(f"Generated {i + 1}/{COUNT}: {filename}")

print("Class B generation complete.")