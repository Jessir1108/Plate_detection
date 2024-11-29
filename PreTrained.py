import matplotlib.pyplot as plt

image_path = 'plate_image.png'
image = plt.imread(image_path)
plt.imshow(image)
plt.show()