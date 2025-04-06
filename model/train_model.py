# model/train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_hybrid_model(X_train, y_train, X_test, y_test):
    print(" Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(" Evaluating model...")
    y_pred = model.predict(X_test)

    print("\n Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, "model/signature_classifier.pkl")
    print(" Model saved to model/signature_classifier.pkl")

    return model
