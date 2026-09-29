"""Side-by-side contact sheets (references | ours) and the reference board, for judging the resemblance.

    python3 demo/images/redesign/contact.py
      -> $DEMO_OUT/images_redesign/contact_{A1_flow,A2_stack,B1_grid,B2_list}.png, $DEMO_OUT/refs_images/board.png
"""
import glob
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
R, O, OURS = OUT / "refs_images", OUT / "images_redesign", Path(__file__).resolve().parent
SHEETS = {
    "A1_flow": ["arch_01_nvidia_nemotron3_diarization_pipeline.png", "arch_05_meta_llama4_moe.png"],
    "A2_stack": ["arch_10_alammar_deepseek_r1_layers.png", "arch_05_meta_llama4_moe.png"],
    "B1_grid": ["res_02_anthropic_claude4_swebench.png", "res_04_deepmind_gemini_live_tau3_light.png"],
    "B2_list": ["res_01_nvidia_nemotron3_diarization_bars.png", "guide_02_datawrapper_title_subtitle.png"],
}
FONT = "/System/Library/Fonts/Helvetica.ttc"


def fit(im, w, h):
    im = im.convert("RGB"); im.thumbnail((w, h), Image.LANCZOS)
    c = Image.new("RGB", (w, h), "white"); c.paste(im, ((w - im.width) // 2, (h - im.height) // 2)); return c


def main():
    f = ImageFont.truetype(FONT, 26)
    for name, refs in SHEETS.items():
        sh = Image.new("RGB", (2000 + 1920 + 60, 1080 * 2 + 120), "#E9EBEE"); d = ImageDraw.Draw(sh)
        for i, r in enumerate(refs):
            sh.paste(fit(Image.open(R / r), 2000, 1080), (20, 60 + i * 1100))
        d.text((20, 18), "references: " + " | ".join(refs), fill="#222", font=f)
        sh.paste(fit(Image.open(OURS / f"{name}.png"), 1920, 1080), (2040, 60))
        sh.paste(fit(Image.open(OURS / f"{name}_square.png"), 1920, 1080), (2040, 1160))
        d.text((2040, 18), f"ours: {name} (wide, square)", fill="#222", font=f)
        sh.save(O / f"contact_{name}.png", optimize=True)
    files = [p for k in ("arch", "res", "guide") for p in sorted(glob.glob(str(R / f"{k}_*")))]
    cw, ch, cols = 640, 380, 6; rows = -(-len(files) // cols)
    b = Image.new("RGB", (cw * cols, (ch + 36) * rows), "white"); d = ImageDraw.Draw(b); f2 = ImageFont.truetype(FONT, 18)
    for i, p in enumerate(files):
        x, y = (i % cols) * cw, (i // cols) * (ch + 36)
        b.paste(fit(Image.open(p), cw - 10, ch - 10), (x + 5, y + 5)); d.text((x + 6, y + ch), os.path.basename(p)[:60], fill="#333", font=f2)
    b.save(R / "board.png", optimize=True)
    print(len(files), "references;", ", ".join(f"contact_{n}.png" for n in SHEETS))


if __name__ == "__main__":
    main()
