import numpy as np
import fld_utils

X_train, Y_train, X_val, Y_val = fld_utils.load_data(validation_split=0.2)
img = X_val[0]
img_size_x, img_size_y = img.shape[0], img.shape[1]

# Plot correct landmarks
landmarks = fld_utils.extract_landmarks(Y_val[0], img.shape[0], img.shape[1])
fld_utils.save_img_with_landmarks(img, landmarks, "data_visual.png", gray_scale=True)

# Baseline model
model = fld_utils.load_model("baseline")
y_pred = model.predict(np.expand_dims(img, axis=0))[0]
landmarks = fld_utils.extract_landmarks(y_pred, img_size_x, img_size_y)
fld_utils.save_img_with_landmarks(img, landmarks, "baseline_prediction.png", gray_scale=True)

# CNN model
model = fld_utils.load_model("cnn")
y_pred = model.predict(np.expand_dims(img, axis=0))[0]
landmarks = fld_utils.extract_landmarks(y_pred, img_size_x, img_size_y)
fld_utils.save_img_with_landmarks(img, landmarks, "cnn_prediction.png", gray_scale=True)
