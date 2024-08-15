import os
import cv2
import gradio as gr
from PIL import Image
from model.classify import PassportClassifier


MODEL_WEIGHTS_GDRIVE_ID = 'https://drive.google.com/drive/folders/1M5_YKX08uozRpQWUN_-O1jp0O87F9npF'
TEST_IMAGES_FOLDER = 'test_images'

css = """
h1 {
    text-align: center;
    display:block;
}
"""

# Initialize the classifier
classifier = PassportClassifier(pth_device='cpu',
                                gdrive_id=MODEL_WEIGHTS_GDRIVE_ID)

# Gradio app function
def classify_image(image):
    # Classify the image
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    label = classifier.classify(image)
    return label


# Gradio app function to load images from 'test' folder
def load_gallery_images():
    gallery_images = []

    # Iterate through image files in the test folder
    for img_file in sorted(os.listdir(TEST_IMAGES_FOLDER)):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(TEST_IMAGES_FOLDER, img_file)
            gallery_images.append(Image.open(img_path))

    return gallery_images

def load_selected_image(image_path, sd: gr.SelectData):
    return Image.open(image_path[sd.index][0])

# Create Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Passport Classification")
    gr.Markdown("Please upload an image to run the model. This is a transformer-based model that classifies whether the uploaded image is a valid passport image or not. This is useful for KYC applications. (Note: Wait for the image to completely finish uploading before clicking on the 'Submit', otherwise you may get an error.)\n\nThis demo is built by NextFlows. Our goal is to make AI cost-effective and accessible for organisations. For commercial inquiries, reach out to nextflows.ai@gmail.com. We can assist with finetuning models for specific use cases, multilingual support, enhancing performance, production deployment, and more.")

    # Image classification interface
    with gr.Row():
        input_image = gr.Image(label="Upload Image")
        output_label = gr.Label(label="Label")

    classify_button = gr.Button("Classify")

    # Link the classify_image function to the button
    classify_button.click(fn=classify_image, inputs=input_image, outputs=output_label)

    # Gallery of images from the 'test' folder
    with gr.Row():
        gr.Markdown("Or select one of these images")
    with gr.Row():
        image_gallery = gr.Gallery(label="Sample images",
                                   show_label=False,
                                   elem_id="gallery",
                                   columns=[5],
                                   rows=[1],
                                   object_fit="contain",
                                   height="auto")

    # Load the gallery images when the interface starts
    demo.load(fn=load_gallery_images, outputs=image_gallery)

    # When the user clicks on an image in the gallery, populate the input box
    image_gallery.select(fn=load_selected_image, inputs=image_gallery, outputs=input_image)

# Launch the app
demo.launch(debug=True, server_port=3103, server_name="0.0.0.0")




## Create Gradio interface
#interface = gr.Interface(
#    fn=classify_image,
#    inputs=gr.Image(),
#    outputs='label',
#    title='Passport Classification',
#    description="""Please upload an image to run the model. This is a transformer-based model that classifies whether the uploaded image is a valid passport image or not. This is useful for KYC applications. (Note: Wait for the image to completely finish uploading before clicking on the 'Submit', otherwise you may get an error.)\n\nThis demo is built by NextFlows. Our goal is to make AI cost-effective and accessible for organisations. For commercial inquiries, reach out to nextflows.ai@gmail.com. We can assist with finetuning models for specific use cases, multilingual support, enhancing performance, production deployment, and more.""",
#)
#
## Launch the app
#interface.launch(share=True)
