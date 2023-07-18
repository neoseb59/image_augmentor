import Augmentor
from pdf2image import convert_from_path


# Store Pdf with convert_from_path function
images = convert_from_path("test.pdf")

for i in range(len(images)):
    # Save pages as images in the pdf
    images[i].save("page" + str(i) + ".jpg", "JPEG")
p = Augmentor.Pipeline("./")
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
p.sample(200)
