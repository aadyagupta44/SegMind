import pickle
import pathlib

current_path = pathlib.Path(__file__).resolve().parent
pickle_file = current_path / 'model.pkl'

with pickle_file.open('rb') as model_file:
    segmentation_model = pickle.load(model_file)
