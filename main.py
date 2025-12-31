from src.predict import input_from_terminal
from src.wrapper_classes import InferenceDelay

if __name__ == "__main__":
    df = input_from_terminal()
    prediction = InferenceDelay.predict_delay(df)
    print("Predicted delay:", int(prediction.iloc[0]))
