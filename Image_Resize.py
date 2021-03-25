from PIL import Image
import glob
import os

# img = Image.open('/Users/suhyeonyoo/Downloads/archive/PNGImages/FudanPed00001.png')
# # #
# # # img_resize = img.resize((300,300), Image.BICUBIC)
# # # img_resize.save('example.jpg')


files = glob.glob('/Users/suhyeonyoo/Downloads/archive/PNGImages/*.png')
print(files)

for f in files:
    img = Image.open(f)
    img_resize = img.resize((300,300), Image.BICUBIC)
    directory, name = os.path.split(f)
    print(directory, name)
    img_resize.save('/Users/suhyeonyoo/Desktop/Face_Recognition_360/Resized_300/'+ name)