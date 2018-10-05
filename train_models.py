import fld_utils
import model_factory

X_train, Y_train, X_val, Y_val = fld_utils.load_data(validation_split=0.2)
NUM_EPOCHS = 100

# Run models
models = []
model_names = []
models.append(model_factory.create_baseline_model())
model_names.append("baseline")
models.append(model_factory.create_cnn_model())
model_names.append("cnn")
fld_utils.run_models(X_train, Y_train, X_val, Y_val, models, model_names, NUM_EPOCHS)
