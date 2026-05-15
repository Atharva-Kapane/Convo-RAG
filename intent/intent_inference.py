import joblib
import os


class IntentClassifier:

    def __init__(self):

        model_path = os.path.join(
            "models",
            "intent_classifier.joblib"
        )

        self.model = joblib.load(model_path)

    def predict(self, text: str):

        # predict label
        prediction = self.model.predict([text])[0]

        # predict confidence
        probabilities = self.model.predict_proba([text])[0]

        confidence = max(probabilities)

        return {
            "intent": prediction,
            "confidence": round(float(confidence), 4)
        }