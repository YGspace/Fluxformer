from PIL import Image

img1 = Image.open('/home/kimminju/workspace/gmflow/demo/kinetic400/00514.png')
img2 = Image.open('/home/kimminju/workspace/gmflow/demo/kinetic400/00515.png')

img1_resized = img1.resize((224, 224))
img2_resized = img2.resize((224, 224))

img1_resized.save('/home/kimminju/workspace/gmflow/demo/kinetic400/00001.jpg')
img2_resized.save('/home/kimminju/workspace/gmflow/demo/kinetic400/00002.jpg')


