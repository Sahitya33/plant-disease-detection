"""
Image Preprocessing Playground
================================
How to run this: open this file in VS Code with the Python extension
installed. You'll see "Run Cell" / "Run Below" links appear above each
block that starts with "# %%". Click one (or put your cursor in a cell
and press Shift+Enter) and VS Code opens an Interactive Window showing
the output and any plots, right next to your code.

If you don't have a leaf photo handy yet, this generates a simple
synthetic placeholder image automatically, so every cell still runs.
To use a real photo instead, drop a file named sample_leaf.jpg into this
same folder.

Install what this needs (separate from the main app's requirements,
since this is just a learning tool):
    pip install matplotlib pillow numpy
"""

# %%
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_leaf.jpg")


def load_or_generate_sample():
    if os.path.exists(SAMPLE_PATH):
        return Image.open(SAMPLE_PATH).convert("RGB")
    # Synthetic stand-in: green leaf shape with a darker blotch, so the
    # later "what does augmentation/contrast do" cells have something
    # visually obvious to act on.
    img = Image.new("RGB", (400, 300), (235, 240, 230))
    draw = ImageDraw.Draw(img)
    draw.ellipse((90, 60, 310, 240), fill=(58, 125, 58))     # leaf body
    draw.ellipse((170, 110, 230, 160), fill=(133, 92, 41))    # "disease" blotch
    return img


original = load_or_generate_sample()
print("Image mode:", original.mode, "| size (w, h):", original.size)

plt.figure(figsize=(4, 3))
plt.imshow(original)
plt.title("Original")
plt.axis("off")
plt.show()

# %% [markdown]
# ## 1. Resizing
# Models need every input image to be the exact same shape. But "resize"
# can mean two different things, and they give different results.

# %%
naive = original.resize((224, 224))  # stretches to fit, ignores aspect ratio


def resize_preserve_aspect(img, size=224):
    """Scale so the short side = size, then crop the center square.
    Avoids the stretching/distortion you get from a naive resize."""
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h))
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


cropped = resize_preserve_aspect(original)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for ax, img, title in zip(
    axes,
    [original, naive, cropped],
    ["Original", "Naive resize\n(stretched)", "Resize + center crop\n(no distortion)"],
):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

# Note: this project's train.py uses image_dataset_from_directory(image_size=...),
# which does the naive stretch. Fine for PlantVillage (leaves are roughly
# centered, framing is consistent) — but for messier datasets,
# resize_preserve_aspect usually keeps more real signal.

# %% [markdown]
# ## 2. Color / format normalization
# Real image folders are messy: some files are grayscale, some have a
# transparency channel (RGBA), some are CMYK. Mix those into training
# unconverted and your code crashes partway through on whatever file
# happens to be different.

# %%
grayscale = original.convert("L")     # 1 channel
rgba = original.convert("RGBA")       # 4 channels (RGB + alpha/transparency)
back_to_rgb = rgba.convert("RGB")     # always normalize back to 3-channel RGB

print("grayscale array shape:", np.array(grayscale).shape)
print("RGBA array shape:     ", np.array(rgba).shape)
print("RGB array shape:      ", np.array(back_to_rgb).shape)

# This is exactly why app/ml_model.py always does:
#     Image.open(path).convert("RGB")
# No matter what format gets uploaded, it's forced into a consistent
# 3-channel shape before it goes anywhere near the model.

# %% [markdown]
# ## 3. Pixel value scaling
# Raw pixels are integers 0-255. Models train better on small, centered
# numbers. There's no single "correct" scaling — it depends on what the
# specific model expects.

# %%
arr = np.array(original).astype("float32")

raw = arr                          # 0 to 255   (what you start with)
zero_one = arr / 255.0             # 0 to 1     (generic normalization)
neg_one_to_one = (arr / 127.5) - 1.0  # -1 to 1 (what MobileNetV2 expects)

print(f"raw range:        {raw.min():.1f} to {raw.max():.1f}")
print(f"[0, 1] range:      {zero_one.min():.2f} to {zero_one.max():.2f}")
print(f"[-1, 1] range:     {neg_one_to_one.min():.2f} to {neg_one_to_one.max():.2f}")

# This is exactly the bug fixed earlier in this project: train.py and
# ml_model.py must use the SAME formula here, matched to whatever
# pretrained model you're using. (mobilenet_v2.preprocess_input does
# this [-1, 1] math for you, which is what both files now call.)

# %% [markdown]
# ## 4. Data augmentation
# Random, realistic variations applied ONLY during training (never at
# prediction time). The point is to stop the model from memorizing exact
# pixel positions/lighting, so it's forced to learn the actual disease
# pattern instead of incidental details.

# %%
flipped = original.transpose(Image.FLIP_LEFT_RIGHT)
rotated = original.rotate(15)
brighter = ImageEnhance.Brightness(original).enhance(1.4)
higher_contrast = ImageEnhance.Contrast(original).enhance(1.4)
zoomed_in = resize_preserve_aspect(original.crop((40, 30, 360, 270)))

fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for ax, img, title in zip(
    axes,
    [flipped, rotated, brighter, higher_contrast, zoomed_in],
    ["Flipped", "Rotated 15°", "Brighter", "More contrast", "Zoomed in"],
):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

# train.py does the equivalent of all five of these (and more)
# automatically via tf.keras.layers.RandomFlip / RandomRotation /
# RandomZoom / RandomContrast — but it's worth seeing what each one
# actually does to a real image at least once.

# %% [markdown]
# ## 5. Putting it all together
# This mirrors exactly what app/ml_model.py does to every uploaded image
# right before prediction.

# %%
def preprocess_for_model(pil_image, size=(224, 224)):
    img = pil_image.convert("RGB").resize(size)        # step 1 + 2: shape + format
    arr = np.array(img).astype("float32")
    arr = (arr / 127.5) - 1.0                            # step 3: scaling
    arr = np.expand_dims(arr, axis=0)                     # add batch dimension
    return arr


batch = preprocess_for_model(original)
print("Final shape going into the model:", batch.shape)     # (1, 224, 224, 3)
print(f"Final value range: {batch.min():.2f} to {batch.max():.2f}")

# %% [markdown]
# ## Try it yourself
# 1. Drop a real leaf photo in as sample_leaf.jpg and re-run from the top.
# 2. Change `resize_preserve_aspect`'s crop to keep the LEFT side instead
#    of the center — see how much of the subject you can accidentally lose.
# 3. Try `ImageEnhance.Color(original).enhance(0.0)` — what does it do, and
#    why might desaturating be a bad augmentation for a *disease* classifier
#    specifically (hint: think about what color tells you here)?
# 4. Write a `preprocess_for_model` variant that uses [0, 1] scaling instead
#    of [-1, 1], and explain in a comment why you would never mix this
#    version with a model trained using the other one.
