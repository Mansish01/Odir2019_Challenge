from torchvision import transforms as t
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def crop_nonzero(image):
    bbox = image.getbbox()
    cropped_image = image.crop(bbox)
    return cropped_image

class CropNonzero(object):
    def __call__(self, image):
        return crop_nonzero(image)

# Define transformations
transforms = t.Compose([
    CropNonzero(),
    t.Resize(size=(256,256)),
    t.ToTensor(),
    t.Normalize(mean=[0.0177, 0.0195, 0.0210], std=[0.2271,0.2271, 0.2271]),
])

if __name__ == '__main__':
    img_path = r"Data/ODIR-5K_Testing_Images/967_left.jpg"
    img = Image.open(img_path)

    transformed_img = transforms(img)
    img_np = transformed_img.numpy()

    plt.imshow(transformed_img.permute(1,2,0))
    plt.show()









