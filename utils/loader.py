import pandas as pd

def load_data(path):
    df = pd.read_excel(path)
    return df