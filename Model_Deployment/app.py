from flask import Flask, request, render_template
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Even if you detect multiple versions of the OpenMP library loaded (like libiomp5md.dll), just allow it — don’t crash.
import numpy as np
import torch
import segmentation_models_pytorch as smp
import tifffile as tiff
import io
import base64
import matplotlib
matplotlib.use('Agg') # Tells matplotlib to use the 'Agg' backend, which is for off-screen (non-GUI) image rendering.
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')
device = torch.device("cpu")

model = smp.Unet(
    encoder_name="densenet169",
    encoder_weights="imagenet",
    in_channels=12,
    classes=1
)
model.load_state_dict(torch.load("best_model_transfer.pth", map_location=device))
model.eval()

@app.route('/')
def home():
    return render_template("app.html")

@app.route('/file_upload', methods=['POST'])
def file_upload():
    file = request.files['file']
    filename = file.filename
    name, _ = os.path.splitext(filename)
    file_bytes = file.read()
    img = tiff.imread(io.BytesIO(file_bytes))

    # img = np.transpose(img, (2, 0, 1))  # C, H, W
    img = img / img.max()
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # 1, C, H, W

    # selected_bands = [1, 4, 5, 6, 7, 9]
    # if img.shape[1] <= max(selected_bands):
    #     return "Image does not have enough bands.", 400

    # img_selected = img[:, selected_bands, :, :].to(device)

    with torch.no_grad():
        output = model(img)
        pred = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

        fig, ax = plt.subplots()
        ax.imshow(pred, cmap='gray')
        ax.axis('off')
        buf = io.BytesIO() # Creates an in-memory buffer (like a virtual file in RAM) — no file will be saved to disk.
                           # This is useful for sending the image directly to the browser.
                           
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0) # Resets the pointer in the buffer to the beginning so it can be read from the start.
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8') # Reads the image data from the buffer.
                                                                  # Encodes the binary data as Base64 (a format suitable for embedding in HTML).
                                                                  # Then decodes it to a UTF-8 string.
        buf.close()

    return render_template("app.html", pred=img_base64, or_label = name)

if __name__ == '__main__':
    app.run(debug=True)
