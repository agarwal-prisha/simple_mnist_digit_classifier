import cv2
import torch
import numpy as np

from model import Net

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# LOAD DATA
# ==============================

def load_images(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data[16:].reshape(-1, 28, 28)

def load_labels(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data[8:]

test_data = load_images("t10k-images.idx3-ubyte")
test_targets = load_labels("t10k-labels.idx1-ubyte")

# ==============================
# LOAD MODEL
# ==============================

network = Net().to(device)
network.load_state_dict(torch.load("Models/model.pt", map_location=device))
network.eval()

# ==============================
# INFERENCE
# ==============================

for test_image, test_target in zip(test_data, test_targets):

    inference_image = torch.from_numpy(test_image).float() / 255.0
    inference_image = inference_image.unsqueeze(0).unsqueeze(0).to(device)

    output = network(inference_image)
    pred = output.argmax(dim=1)

    prediction = str(pred.item())

    test_image = cv2.resize(test_image, (400, 400))
    cv2.imshow(prediction, test_image)

    key = cv2.waitKey(0)
    if key == ord("q"):
        break

cv2.destroyAllWindows()