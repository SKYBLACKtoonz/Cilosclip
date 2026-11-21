
from PIL import Image
from pathlib import Path

icon_folder = Path("icon")

png_files = list(icon_folder.glob("*.png"))

if not png_files:
    raise FileNotFoundError("No PNG found in the icon folder!")

image = Image.open(png_files[0]).convert("RGBA")

image.save(
    icon_folder / "clipforge.ico",
    format="ICO",
    sizes=[
        (256, 256),
        (128, 128),
        (64, 64),
        (48, 48),
        (32, 32),
        (16, 16)
    ]
)

print("SUCCESS! ClipForge official icon created!")
