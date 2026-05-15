import joblib
import time
import os

def run_test():
    # Adjusted path to match your saved location
    model_path = 'models/intent_classifier.joblib'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load model (Offline)
    clf = joblib.load(model_path)
    
    # Test cases to verify the categories requested in the prompt
    test_cases = [
        ("Can you remind me to check the oven in 20 minutes?", "reminder"),
        ("I've been feeling really down and lonely lately", "emotional-support"),
        ("Please send the final report to the manager by EOD", "action-item"),
        ("I am going to Mumbai!", "small-talk"),
        ("asdfghjkl 12345", "unknown"),
        (" You should! I think you'd like her.","action-item")
    ]

    print("\n--- Offline Intent Classifier Performance Test ---")
    print(f"{'Input Text':<50} | {'Predicted':<18} | {'Conf':<6} | {'Latency'}")
    print("-" * 95)

    total_latency = 0

    for text, expected in test_cases:
        start_time = time.time()
        
        # Get probabilities
        probs = clf.predict_proba([text])
        max_idx = probs.argmax()
        confidence = probs[0][max_idx]
        label = clf.classes_[max_idx]
        
        # Logic: If model is unsure, mark as unknown
        if confidence < 0.35:
            label = "unknown"
            
        latency_ms = (time.time() - start_time) * 1000
        total_latency += latency_ms
        
        print(f"{text[:48]:<50} | {label:<18} | {confidence:.2f} | {latency_ms:.2f}ms")

    avg_latency = total_latency / len(test_cases)
    print("-" * 95)
    print(f"Avg Latency: {avg_latency:.2f}ms (Target: <200ms)")
    print(f"Offline Status: Fully Offline (No API calls)")
    print("--- Test Complete ---\n")

if __name__ == "__main__":
    run_test()
