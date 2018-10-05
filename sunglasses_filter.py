import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import fld_utils

# Load original image
face_img_path = "input/picard.png"
orig_img = cv2.imread(face_img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
orig_size_x, orig_size_y = orig_img.shape[0], orig_img.shape[1]

# Prepare input image
img = cv2.imread(face_img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_AREA)
img = np.expand_dims(img, axis=2)
img = img / 255
img = img.astype("float32")

# Predict landmarks
model = fld_utils.load_model("cnn")
y_pred = model.predict(np.expand_dims(img, axis=0))[0]
landmarks = fld_utils.extract_landmarks(y_pred, orig_size_x, orig_size_y)

# Save original image with landmarks on top
fld_utils.save_img_with_landmarks(orig_img, landmarks, "test_img_prediction.png")

# Extract x and y values from landmarks of interest
left_eye_center_x = int(landmarks[0][0])
left_eye_center_y = int(landmarks[0][1])
right_eye_center_x = int(landmarks[1][0])
right_eye_center_y = int(landmarks[1][1])
left_eye_outer_x = int(landmarks[3][0])
right_eye_outer_x = int(landmarks[5][0])

# Load images using PIL
# PIL has better functions for rotating and pasting compared to cv2
face_img = Image.open(face_img_path)
sunglasses_img = Image.open("input/sunglasses.png")

# Resize sunglasses
sunglasses_width = int((left_eye_outer_x - right_eye_outer_x) * 1.4)
sunglasses_height = int(sunglasses_img.size[1] * (sunglasses_width / sunglasses_img.size[0]))
sunglasses_resized = sunglasses_img.resize((sunglasses_width, sunglasses_height))

# Rotate sunglasses
eye_angle_radians = np.arctan((right_eye_center_y - left_eye_center_y) / (left_eye_center_x - right_eye_center_x))
sunglasses_rotated = sunglasses_resized.rotate(np.degrees(eye_angle_radians), expand=True, resample=Image.BICUBIC)

# Compute positions such that the center of the sunglasses is
# positioned at the center point between the eyes
x_offset = int(sunglasses_width * 0.5)
y_offset = int(sunglasses_height * 0.5)
pos_x = int((left_eye_center_x + right_eye_center_x) / 2) - x_offset
pos_y = int((left_eye_center_y + right_eye_center_y) / 2) - y_offset

# Paste sunglasses on face image
face_img.paste(sunglasses_rotated, (pos_x, pos_y), sunglasses_rotated)
face_img.save("output/test_img_sunglasses.png")
