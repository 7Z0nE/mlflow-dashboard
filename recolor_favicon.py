from PIL import Image
import colorsys

def shift_hue(image, hue_shift):
    img = image.convert('RGBA')
    pixels = img.load()
    for i in range(img.width):
        for j in range(img.height):
            r, g, b, a = pixels[i, j]
            if a == 0:
                continue
            h, l, s = colorsys.rgb_to_hls(r/255., g/255., b/255.)
            h = (h + hue_shift) % 1.0
            r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
            pixels[i, j] = (int(r_new*255), int(g_new*255), int(b_new*255), a)
    return img

original = Image.open('static/original_favicon.ico')
icons = []
try:
    while True:
        # Shift hue backwards by ~30 degrees (0.08 in 0..1) to go from Blue to Turquoise
        shifted = shift_hue(original, -0.08)
        icons.append(shifted)
        original.seek(original.tell() + 1)
except EOFError:
    pass

if icons:
    sizes = [(img.width, img.height) for img in icons]
    icons[0].save('static/favicon.ico', format='ICO', sizes=sizes, append_images=icons[1:])
    print("Saved static/favicon.ico with turquoise colors.")
