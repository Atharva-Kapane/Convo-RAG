import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import joblib
import os

# --- 1. DATASET ---
# synthetic data
data = [
    # --- SMALL-TALK (40) ---
    ("Hi! How are you doing today?", "small-talk"),
    ("I'm doing pretty well, thanks for asking.", "small-talk"),
    ("I just got back from a long hike in the mountains, it was beautiful!", "small-talk"),
    ("I'm a full-time student studying biology at the local university.", "small-talk"),
    ("I love to cook Italian food on the weekends, especially lasagna.", "small-talk"),
    ("I'm originally from Seattle but I've lived in Austin for three years now.", "small-talk"),
    ("It's such a beautiful day outside, the sun is finally out!", "small-talk"),
    ("I've been reading a really fascinating book about space lately.", "small-talk"),
    ("I have two cats and they are honestly my best friends.", "small-talk"),
    ("I'm thinking about starting a small herb garden on my balcony.", "small-talk"),
    ("I'm moving to Portland soon to pursue my dream of being a chef!", "small-talk"),
    ("I work as a children's librarian and I absolutely love my job.", "small-talk"),
    ("I'm a big fan of punk music, I even play in a small local band.", "small-talk"),
    ("I just finished a yoga class and I feel so much better now.", "small-talk"),
    ("I've been working on restoring a classic 1964 Impala in my garage.", "small-talk"),
    ("I love taking my kids to the park to play soccer on Saturdays.", "small-talk"),
    ("I'm training for a marathon, so I've been running quite a lot lately.", "small-talk"),
    ("Have you ever been to Powell's Books in Portland? It's amazing.", "small-talk"),
    ("I'm an avid fisherman and spend most of my weekends at the lake.", "small-talk"),
    ("I'm a nurse at the city hospital, it's a very busy but rewarding job.", "small-talk"),
    ("I love going to see the fall colors in the mountains every year.", "small-talk"),
    ("I'm learning how to play the piano, though I'm still a beginner.", "small-talk"),
    ("I'm a software engineer, so I spend a lot of time at my computer.", "small-talk"),
    ("I'm really excited because I just got a new job as a firefighter!", "small-talk"),
    ("I enjoy singing karaoke, even if I'm not the best at it.", "small-talk"),
    ("I'm getting ready to go out with some friends for dinner tonight.", "small-talk"),
    ("I just woke up and I'm about to have my first cup of coffee.", "small-talk"),
    ("I grew up in a very small town and then moved to the big city.", "small-talk"),
    ("I'm a single parent, so life stays pretty busy for me.", "small-talk"),
    ("I love to make chocolate chip cookies for my coworkers.", "small-talk"),
    ("I'm a big fan of the Harry Potter series, I've read them all twice.", "small-talk"),
    ("I'm a bit nervous about moving away from my family and friends.", "small-talk"),
    ("I'm a student studying radiology, it's a lot of hard work.", "small-talk"),
    ("I've been meaning to try archery, it seems like a cool hobby.", "small-talk"),
    ("I love meeting new people and hearing their stories.", "small-talk"),
    ("I'm just hanging out today, not doing much of anything.", "small-talk"),
    ("I'm a huge fan of classic cars, especially those from the 60s.", "small-talk"),
    ("I'm trying to stay positive and look forward to the future.", "small-talk"),
    ("I'm going to start working on a new paint job for my car this weekend.", "small-talk"),
    ("I love the freedom of being a student and learning new things.", "small-talk"),

    # --- EMOTIONAL-SUPPORT (40) ---
    ("I've been feeling a bit overwhelmed with work lately.", "emotional-support"),
    ("I really miss my family since I moved across the country.", "emotional-support"),
    ("I'm so sad that my favorite local bookstore is closing down.", "emotional-support"),
    ("I've been feeling pretty lonely this weekend and just wanted to talk.", "emotional-support"),
    ("I'm really nervous about my presentation at the college tomorrow.", "emotional-support"),
    ("I just need some comfort right now, it's been a hard day.", "emotional-support"),
    ("I'm so happy that everything finally worked out with the move!", "emotional-support"),
    ("I've been feeling a lot of anxiety about what the future holds.", "emotional-support"),
    ("I'm so proud of myself for finishing that project on time.", "emotional-support"),
    ("I'm feeling a bit down today because it's been raining non-stop.", "emotional-support"),
    ("I'm really struggling with being away from my support system.", "emotional-support"),
    ("I feel so stressed about these finals, I don't know if I can do it.", "emotional-support"),
    ("I'm still grieving the loss of my dog, he was such a good companion.", "emotional-support"),
    ("I'm feeling discouraged because my car keeps breaking down.", "emotional-support"),
    ("I'm so excited but also terrified to start my new career.", "emotional-support"),
    ("I feel like I'm not making enough time for the things I love.", "emotional-support"),
    ("I'm feeling really grateful for my friends today.", "emotional-support"),
    ("I've been feeling a bit lost lately, like I'm just going through the motions.", "emotional-support"),
    ("I'm so frustrated that I can't seem to get this piano piece right.", "emotional-support"),
    ("I'm feeling optimistic for the first time in a long while.", "emotional-support"),
    ("I'm really upset about how that conversation went earlier.", "emotional-support"),
    ("I've been feeling very tired and drained from the work week.", "emotional-support"),
    ("I'm so happy to be pursuing my culinary dreams at last.", "emotional-support"),
    ("I'm feeling a little insecure about starting this new job.", "emotional-support"),
    ("I just feel like everything is changing too fast for me.", "emotional-support"),
    ("I'm really sad to leave my old city and all my memories there.", "emotional-support"),
    ("I'm feeling much more confident after that yoga session.", "emotional-support"),
    ("I'm so worried about my kids and their transition to the new school.", "emotional-support"),
    ("I've been feeling a bit angry about the situation at work.", "emotional-support"),
    ("I'm so glad I have someone to talk to about my feelings.", "emotional-support"),
    ("I'm feeling overwhelmed by all the packing I still have to do.", "emotional-support"),
    ("I'm really proud of how far I've come this year.", "emotional-support"),
    ("I'm feeling a bit hopeless today, but I'm trying to stay strong.", "emotional-support"),
    ("I'm so excited to finally see my parents after so long.", "emotional-support"),
    ("I'm feeling a bit burnt out from studying all the time.", "emotional-support"),
    ("I'm really disappointed that I didn't get the promotion.", "emotional-support"),
    ("I'm feeling peaceful for the first time in weeks.", "emotional-support"),
    ("I'm so nervous about meeting new people in Portland.", "emotional-support"),
    ("I'm feeling a bit lonely even though I'm surrounded by people.", "emotional-support"),
    ("I'm so happy I took the kids to the mountains, they loved it.", "emotional-support"),

    # --- REMINDER (40) ---
    ("Remind me to buy some more coffee beans tomorrow morning.", "reminder"),
    ("Can you remember that we have dinner reservations at 7?", "reminder"),
    ("Don't forget to take the trash out before you leave tonight.", "reminder"),
    ("Make sure to remind me about the team meeting at 3 PM.", "reminder"),
    ("Remind me to call my mom for her birthday this Sunday.", "reminder"),
    ("Can you set a reminder for my dentist appointment on Tuesday?", "reminder"),
    ("Don't let me forget to pack my gym shoes for tomorrow.", "reminder"),
    ("Remind me to check the mailbox as soon as we get home.", "reminder"),
    ("I need to remember to pay the electricity bill later today.", "reminder"),
    ("Please remind me to water the indoor plants tomorrow.", "reminder"),
    ("Remind me to pick up a birthday card for Sarah.", "reminder"),
    ("Don't forget that we need to drop off the library books.", "reminder"),
    ("Can you remind me to check the oil in the car this weekend?", "reminder"),
    ("Make a note for me to bring my umbrella tomorrow morning.", "reminder"),
    ("Remind me to send that email to the landlord tonight.", "reminder"),
    ("Don't forget to take your vitamins after breakfast.", "reminder"),
    ("Remind me to book the hotel for our trip to Portland.", "reminder"),
    ("Can you remember to buy milk on your way home?", "reminder"),
    ("Remind me to prep the chicken for dinner later.", "reminder"),
    ("I need to remember to call the insurance company tomorrow.", "reminder"),
    ("Please remind me to charge my laptop before the presentation.", "reminder"),
    ("Remind me to sign the kids' permission slips tonight.", "reminder"),
    ("Don't forget to renew the car registration this month.", "reminder"),
    ("Can you remind me to look up the museum hours?", "reminder"),
    ("Remind me to stop by the pharmacy on the way back.", "reminder"),
    ("Make sure I don't forget to bring the desert for the party.", "reminder"),
    ("Remind me to ask the vet about the new cat food.", "reminder"),
    ("Don't forget to turn off the oven when you're done.", "reminder"),
    ("Remind me to wake up early for the hike on Saturday.", "reminder"),
    ("Can you remember to print out the tickets for the show?", "reminder"),
    ("Remind me to text my brother back this evening.", "reminder"),
    ("I have to remember to pack the cooler for the beach.", "reminder"),
    ("Please remind me to set my alarm for 6 AM tomorrow.", "reminder"),
    ("Remind me to check the flight status before we leave.", "reminder"),
    ("Don't forget to grab the keys from the kitchen counter.", "reminder"),
    ("Remind me to bring the recipe for the chicken parmesan.", "reminder"),
    ("Can you remember to buy more stamps at the post office?", "reminder"),
    ("Remind me to update my resume for the new job application.", "reminder"),
    ("Don't let me forget to call the gardener on Monday.", "reminder"),
    ("Remind me to put the laundry in the dryer before bed.", "reminder"),

    # --- ACTION-ITEM (40) ---
    ("Can you send me the address of that new restaurant?", "action-item"),
    ("Let's plan to meet at the library around noon tomorrow.", "action-item"),
    ("Please email me the final report once you've finished it.", "action-item"),
    ("We should book the flights for our cross-country trip soon.", "action-item"),
    ("Can you pick up some bread and eggs on your way back?", "action-item"),
    ("Let's organize a small get-together for next weekend.", "action-item"),
    ("Put those research files in the shared folder for me, please.", "action-item"),
    ("We need to decide on a color for the living room paint.", "action-item"),
    ("Can you look up the hours for the Everglades park for me?", "action-item"),
    ("Let's finish this group project by Friday afternoon.", "action-item"),
    ("Please send me the link to that punk band's website.", "action-item"),
    ("Can you check if there are any yoga classes on Sunday?", "action-item"),
    ("Let's go to Powell's Books when we visit Portland.", "action-item"),
    ("Please take the dog for a walk while I'm at work.", "action-item"),
    ("Can you help me move these heavy boxes into the truck?", "action-item"),
    ("Let's look at the map and plan our hiking route.", "action-item"),
    ("Please share the recipe for those chocolate chip cookies.", "action-item"),
    ("Can you call the restaurant and change our reservation?", "action-item"),
    ("Let's start working on the car's engine this Saturday.", "action-item"),
    ("Please send me a list of the books we need for class.", "action-item"),
    ("Can you find a good hotel near the culinary school?", "action-item"),
    ("Let's buy the soccer gear for the kids this evening.", "action-item"),
    ("Please sign up for the book club before the deadline.", "action-item"),
    ("Can you check the weather forecast for our mountain trip?", "action-item"),
    ("Let's schedule a call to discuss the project details.", "action-item"),
    ("Please download the new app for the radiology course.", "action-item"),
    ("Can you buy the tickets for the karaoke night?", "action-item"),
    ("Let's make a list of everything we need for the move.", "action-item"),
    ("Please print the directions to the hospital for me.", "action-item"),
    ("Can you send a text to the group about the hike?", "action-item"),
    ("Let's meet at the coffee shop at 9 AM sharp.", "action-item"),
    ("Please check if the library has the new Harry Potter book.", "action-item"),
    ("Can you help me find my keys? I think I lost them.", "action-item"),
    ("Let's go see the fall colors this weekend before they're gone.", "action-item"),
    ("Please order the pizza for the kids' soccer party.", "action-item"),
    ("Can you look for a local band that's playing tonight?", "action-item"),
    ("Let's sit down and figure out our budget for the month.", "action-item"),
    ("Please send me the photos from our trip to the Everglades.", "action-item"),
    ("Can you check the oil levels in the Impala today?", "action-item"),
    ("Let's try that new Italian place for dinner tonight.", "action-item"),

    # --- UNKNOWN (40) ---
    ("asdfghjkl", "unknown"),
    ("982374928374", "unknown"),
    ("!@#$%^&*()", "unknown"),
    ("lorem ipsum dolor sit amet", "unknown"),
    ("skdjfhksjdhf", "unknown"),
    ("000000000", "unknown"),
    ("sdf", "unknown"),
    ("....................", "unknown"),
    ("qwert yuiop", "unknown"),
    ("banana phone refrigerator 42", "unknown"),
    ("zzzzzzzzzzzzzz", "unknown"),
    ("the the the the the", "unknown"),
    ("123 abc 789 xyz", "unknown"),
    ("??? ??? ???", "unknown"),
    ("error 404 text not found", "unknown"),
    ("---", "unknown"),
    ("random letters here", "unknown"),
    ("poiuytrewq", "unknown"),
    ("1a2b3c4d5e", "unknown"),
    ("not a sentence at all", "unknown"),
    ("mnbvcxz", "unknown"),
    ("!!!!!!!!!!", "unknown"),
    ("1. 2. 3. 4. 5.", "unknown"),
    ("testing 1 2 3 testing", "unknown"),
    ("a b c d e f g", "unknown"),
    ("~~~~~~~~~~~~~~~~", "unknown"),
    ("filler text goes here", "unknown"),
    ("jklöäü", "unknown"),
    ("........................", "unknown"),
    ("010101010101", "unknown"),
    ("void", "unknown"),
    ("null null null", "unknown"),
    ("x x x x x", "unknown"),
    ("--- --- ---", "unknown"),
    ("asdf", "unknown"),
    ("lkj hg fd sa", "unknown"),
    ("9999999999", "unknown"),
    ("!!!!!!!!!!!!!!!!!!!!!!!!!", "unknown"),
    ("qwerty", "unknown"),
    ("z x c v b n m", "unknown")
]

def train_intent_model():
    print("Starting model training...")
    
    # Create directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['text', 'label'])

    # Split data to validate performance (80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # --- 2. THE PIPELINE ---
    # TfidfVectorizer: Converts text to numbers, using single words and pairs (bigrams)
    # CalibratedClassifierCV: Wraps the SVM to provide probability/confidence scores
    # LinearSVC: The core "brain" that is extremely fast and lightweight
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), 
            stop_words='english',
            max_features=5000
        )),
        ('clf', CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', C=1.0, max_iter=2000),
            cv=3
        ))
    ])

    # --- 3. TRAINING & VALIDATION ---
    pipeline.fit(train_df['text'], train_df['label'])
    
    accuracy = pipeline.score(test_df['text'], test_df['label'])
    print(f"Model Training Complete!")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # --- 4. EXPORT ---
    model_path = 'models/intent_classifier.joblib'
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    train_intent_model()
