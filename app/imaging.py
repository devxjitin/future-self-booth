from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat


def fit(image: Image.Image, max_side: int) -> Image.Image:
    """Apply EXIF rotation, then downscale so the longest side is <= max_side and both sides are multiples of 16."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    w, h = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
    return image.resize((w, h), Image.LANCZOS)


def _transfer(src: Image.Image, ref: Image.Image, strength: float, mask: Image.Image | None = None,
              match_contrast: bool = True) -> Image.Image:
    """Reinhard colour transfer in LAB: move src's mean (and contrast) toward ref, measured inside mask."""
    src_lab, ref_lab = src.convert("LAB"), ref.convert("LAB")
    s_stat, r_stat = ImageStat.Stat(src_lab, mask), ImageStat.Stat(ref_lab, mask)
    bands = []
    for i, band in enumerate(src_lab.split()):
        scale = r_stat.stddev[i] / s_stat.stddev[i] if match_contrast and s_stat.stddev[i] > 1e-3 else 1.0
        scale = min(1.6, max(0.6, scale))
        lut = [max(0, min(255, round(v + strength * ((v - s_stat.mean[i]) * scale + r_stat.mean[i] - v))))
               for v in range(256)]
        bands.append(band.point(lut))
    return Image.merge("LAB", bands).convert("RGB")


def _face_mask(size, shrink: float = 1.0, blur: float = 0.0) -> Image.Image:
    """Ellipse around where a photo-booth face sits (centre, slightly above middle)."""
    w, h = size
    rx, ry, cx, cy = 0.22 * w * shrink, 0.24 * h * shrink, 0.5 * w, 0.47 * h
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    return m.filter(ImageFilter.GaussianBlur(blur)) if blur else m


def match_colors(image: Image.Image, reference: Image.Image, same_scene: bool = True) -> Image.Image:
    """Bring back the input photo's lighting and skin tone.

    Image models tend to relight and lighten skin. When the edit keeps the same scene, first match the whole
    frame's colour statistics. Then match the face area (measured in the centre of the frame, where the
    camera puts the face) and blend that in with a soft mask. With same_scene=False (a new background on
    purpose) only the face is matched.
    """
    image = image.convert("RGB")
    ref = reference.convert("RGB").resize(image.size)
    out = _transfer(image, ref, 0.6) if same_scene else image
    face = _transfer(out, ref, 0.9, _face_mask(image.size, 0.8), match_contrast=False)
    return Image.composite(face, out, _face_mask(image.size, 1.35, blur=image.width * 0.06))
