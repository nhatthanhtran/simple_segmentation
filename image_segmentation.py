from ultralytics import YOLO
from PIL import Image
import pillow_heif
import os
import matplotlib.pyplot as plt
# load pretrained segmentation model
# model = YOLO("yolov8n-seg.pt")  # nano = fastest

# data_path = "./data/"
# save_path = "./results/"

# # process HEIC image
# # pillow_heif.register_heif_opener()
# img_name = "example3.jpg"
# img = Image.open(f"{data_path}{img_name}")
# img_res = 640
# img = img.resize((img_res, img_res))  # resize to model input size
# # img.save(f"{data_path}{img_name.replace('.HEIC', '.jpg')}")
# # run on image
# # import pdb; pdb.set_trace()
# results = model(img)




def segmentation(model, data_path, save_path, img_res=640):

    # getting all images in the data path
    for img_name in os.listdir(data_path):
        if img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png"):
            img = Image.open(f"{data_path}{img_name}")
            img = img.resize((img_res, img_res))  # resize to model input size
            
            results = model(img)
    
            fig, ax = plt.subplots()
            ax.imshow(results[0].plot())  # Returns the overlay image as NumPy array
            plt.axis('off')
            plt.savefig(os.path.join(save_path, img_name.replace('.jpg', '_output.jpg')), bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    return None

def convert_heic_to_jpeg(data_path):
    pillow_heif.register_heif_opener()
    # Convert HEIC to JPEG
    for file in os.listdir(data_path):
        if file.endswith(".HEIC"):
            img = Image.open(f"{data_path}{file}")
            img.save(f"{data_path}{file.replace('.HEIC', '.jpg')}")
    print(f"Converted HEIC images to JPEG in {data_path}")

if __name__ == "__main__":
    
    data_path = "./data/img/"
    save_path = "./results/img/"
    model = YOLO("yolov8n-seg.pt")  # nano = fastest
    # convert HEIC images to JPEG
    convert_heic_to_jpeg(data_path)
    # run segmentation on images
    segmentation(model, data_path, save_path)
