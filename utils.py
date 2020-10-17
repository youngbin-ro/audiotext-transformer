import pickle


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
