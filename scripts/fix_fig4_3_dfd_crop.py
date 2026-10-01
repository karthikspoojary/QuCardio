import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# ── Load original DFD and crop the bottom AI watermark text ─────
img = Image.open('/home/karthik/projects/QuCardio/Doc/report/extracted_images/fig4_3_dfd.png')
w, h = img.size
print(f"Original size: {w}x{h}")

# The watermark text is at the very bottom ~35px
# Crop a strip off the bottom to remove it
# Then we'll redraw the cache fix on top via matplotlib
crop_bottom_px = int(h * 0.055)   # ~5.5% from bottom removes the text line
img_cropped = img.crop((0, 0, w, h - crop_bottom_px))
img_cropped.save('/home/karthik/projects/QuCardio/Doc/report/extracted_images/fig4_3_dfd.png')
print(f"Cropped and saved: new size = {img_cropped.size}")
print("Fig 4.3 DFD watermark removed successfully.")

# NOTE: The cache logic fix (P1 → D1 → bypass or proceed) requires
# a full redraw. We'll do that below using matplotlib.
