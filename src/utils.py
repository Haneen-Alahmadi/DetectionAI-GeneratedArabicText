#Import Necessary Libraries
import joblib
import os


# Save Models
def save_model(model, name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{name}.pkl"
    joblib.dump(model, path)
    print(f"Model saved successfully {path}")