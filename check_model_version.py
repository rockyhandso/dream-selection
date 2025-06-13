import joblib

# Encoder file ka path
encoders_path = 'model/label_encoders.joblib'

# Load label encoders dictionary
label_encoders = joblib.load(encoders_path)

# Venue ke liye encoder check karte hain
if 'venue' in label_encoders:
    venue_encoder = label_encoders['venue']
    venues = list(venue_encoder.classes_)
    print("Venues trained in encoder:")
    print(venues)
    
    # Check if Wankkade Stadium present hai
    if 'wankkade stadium' in [v.lower() for v in venues]:
        print("Yes, Wankkade Stadium is present in venue labels.")
    else:
        print("No, Wankkade Stadium is NOT present in venue labels.")
else:
    print("No 'venue' encoder found in label encoders.")
