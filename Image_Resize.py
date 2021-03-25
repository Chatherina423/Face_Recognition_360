from PIL import Image

img = Image.open('/Users/suhyeonyoo/Downloads/archive/PNGImages/FudanPed00001.png')

img_resize = img.resize((int(img.width / 2), int(img.height / 2)))
img_resize.save('example.jpg')