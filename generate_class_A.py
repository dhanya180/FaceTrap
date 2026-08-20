import csv
import os
import random
import sys

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

MODEL_PATH = "./models/stable-diffusion-v1-5"

CONFIG = {
    "A_1": {
        "input": "./dataset/references/class_A/A_1.jpg",
        "output": "./dataset/synthetic/class_A/A_1",
        "count": 500,
        "use_aging": True
    },
    "A_2": {
        "input": "./dataset/references/class_A/A_2.jpg",
        "output": "./dataset/synthetic/class_A/A_2",
        "count": 500,
        "use_aging": False
    },
}

AGES = [
    "younger adult appearance",
    "slightly younger adult appearance",
    "current age appearance",
    "slightly older adult appearance",
    "older adult appearance",
]

SCALES = [
    "close-up portrait framing",
    "medium portrait framing",
    "slightly farther camera framing",
]

LIGHTING = [
    "natural daylight",
    "soft daylight",
    "bright indoor lighting",
    "dim indoor lighting",
    "warm indoor lighting",
]

POSES = [
    "frontal face",
    "slightly turned to the left",
    "slightly turned to the right",
    "slightly looking upward",
    "slightly looking downward",
]

BACKGROUNDS = [
    "simple neutral indoor background",
    "subtle office background",
    "plain wall background",
    "softly blurred indoor background",
    "natural outdoor background",
]

EXPRESSIONS = [
    "neutral relaxed expression",
    "subtle natural smile",
    "calm expression",
]

NEGATIVE_PROMPT = (
    "deformed face, distorted face, malformed eyes, malformed mouth, "
    "extra face, multiple people, duplicate person, blurry face, "
    "cartoon, illustration, painting, unrealistic skin, "
    "extreme pose, extreme aging, low quality"
)

device = "cuda"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    local_files_only=True,
)

pipe = pipe.to(device)
pipe.enable_attention_slicing()

identity = sys.argv[1]
config = CONFIG[identity]

os.makedirs(config["output"], exist_ok=True)

input_image = Image.open(config["input"]).convert("RGB")
input_image = input_image.resize((512, 512))

metadata_path = os.path.join(config["output"], "metadata.csv")

with open(metadata_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "image",
        "identity",
        "age",
        "scale",
        "lighting",
        "pose",
        "background",
        "expression",
        "seed",
    ])

    for i in range(config["count"]):

        if config["use_aging"]:
            age = random.choice(AGES)
        else:
            age = "current age appearance"

        scale = random.choice(SCALES)
        lighting = random.choice(LIGHTING)
        pose = random.choice(POSES)
        background = random.choice(BACKGROUNDS)
        expression = random.choice(EXPRESSIONS)

        prompt = (
            f"realistic photograph of the same person, "
            f"{age}, {scale}, {lighting}, {pose}, "
            f"{background}, {expression}, "
            f"natural skin texture, realistic facial details, "
            f"high quality portrait photograph"
        )

        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=input_image,
            strength=0.15,
            guidance_scale=7.0,
            num_inference_steps=30,
            generator=generator,
        ).images[0]

        filename = f"{identity}_{i + 1:04d}.png"
        filepath = os.path.join(config["output"], filename)

        result.save(filepath)

        writer.writerow([
            filename,
            identity,
            age,
            scale,
            lighting,
            pose,
            background,
            expression,
            seed,
        ])

        print(f"[{identity}] Generated {i + 1}/{config['count']}: {filename}")

print("Generation complete.")
