from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
import csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the NLP recommender
from utils.nlp_recommender import NLPRecommender

app = Flask(__name__)
app.secret_key = 'mexico_city_recommender_2025'  # For session management

# --- Path Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Initialize recommenders
restaurant_recommender = None
activity_recommender = None

# Load data - now optional since we use Pinecone metadata
def load_data():
    """
    Load local JSON data if available, otherwise return empty lists.
    With full Pinecone integration, this is only used for setup/migration.
    """
    try:
        with open(os.path.join(BASE_DIR, 'data', 'activities.json'), 'r', encoding='utf-8') as f:
            activities = json.load(f)
        with open(os.path.join(BASE_DIR, 'data', 'restaurants.json'), 'r', encoding='utf-8') as f:
            restaurants = json.load(f)
        print("Loaded local JSON data files")
        return activities, restaurants
    except FileNotFoundError as e:
        print(f"Local JSON files not found - using Pinecone metadata only: {e}")
        return [], []


initial_questions = [
    {
        "question": "What's your budget preference?",
        "choices": [
            {"text": "Budget-friendly", "tags": ["price:low", "street food", "tacos", "antojitos"]},
            {"text": "Mid-range", "tags": ["price:medium", "casual dining", "fonda style"]},
            {"text": "High-end", "tags": ["price:high", "fine dining", "upscale", "tasting menu"]}
        ]
    },
    {
        "question": "What type of cuisine interests you most?",
        "choices": [
            {"text": "Traditional Mexican", "tags": ["traditional mexican", "tacos", "mexican", "mole"]},
            {"text": "Modern/Contemporary", "tags": ["modern mexican", "contemporary", "fusion"]},
            {"text": "International", "tags": ["japanese", "italian", "french", "korean", "thai"]}
        ]
    },
    {
        "question": "What kind of dining experience do you prefer?",
        "choices": [
            {"text": "Quick & Casual", "tags": ["street food", "tacos", "market stall", "casual"]},
            {"text": "Bar & Social", "tags": ["bar", "cantina", "nightlife", "cocktails"]},
            {"text": "Special Occasion", "tags": ["fine dining", "upscale", "tasting menu", "historic"]}
        ]
    },
    {
        "question": "What type of activities are you looking for?",
        "choices": [
            {"text": "Cultural & Museums", "tags": ["museum", "culture", "history", "art"]},
            {"text": "Outdoor & Nature", "tags": ["park", "hike", "nature", "outdoor"]},
            {"text": "Urban Exploration", "tags": ["neighborhood", "market", "landmark", "plaza"]}
        ]
    }
]

def initialize_recommenders():
    """Initialize the NLP recommenders with Pinecone indexes"""
    global restaurant_recommender, activity_recommender
    
    # Load data if not already loaded
    activities, restaurants = load_data()
    
    # Initialize restaurant recommender if needed
    if restaurant_recommender is None:
        print("Initializing restaurant recommender with Pinecone...")
        restaurant_recommender = NLPRecommender(index_name='restaurants')
        
        # Check if we need to populate the index
        if restaurant_recommender.index:
            try:
                # get the index statistics
                stats = restaurant_recommender.index.describe_index_stats()
                if stats['total_vector_count'] == 0:
                    print("Populating restaurant index...")
                    restaurant_recommender.populate_index(restaurants)
                else:
                    print(f"Restaurant index already contains {stats['total_vector_count']} vectors")
            except Exception as e:
                print(f"Error checking restaurant index stats: {e}")
    
    # Initialize activity recommender if needed
    if activity_recommender is None:
        print("Initializing activity recommender with Pinecone...")
        activity_recommender = NLPRecommender(index_name='activities')
        
        # Check if we need to populate the index
        if activity_recommender.index:
            try:
                # Try to get index stats
                stats = activity_recommender.index.describe_index_stats()
                if stats['total_vector_count'] == 0:
                    print("Populating activity index...")
                    activity_recommender.populate_index(activities)
                else:
                    print(f"Activity index already contains {stats['total_vector_count']} vectors")
            except Exception as e:
                print(f"Error checking activity index stats: {e}")
    
    return restaurant_recommender, activity_recommender

def get_recommendations(preferences, item_type, count=5):
    """Get recommendations using the Pinecone-backed NLP recommender. 
    Needed to switch to Pinecone to cut down Docker Image size!"""
    # Initialize recommenders if needed
    restaurant_rec, activity_rec = initialize_recommenders()
    
    # Select the appropriate recommender
    recommender = restaurant_rec if item_type == 'restaurant' else activity_rec
    
    # Make sure the user embedding is initialized with preferences
    if recommender.user_embedding is None:
        preference_text = " ".join(preferences)
        print(f"Initializing {item_type} embedding with: {preference_text[:100]}...")
        recommender.exponential_moving_average_embedding_update(preference_text)
    
    # Get recs and metadata
    print(f"Getting {count} {item_type} recommendations from Pinecone...")
    recommendations = recommender.get_recommendations(k=count)
    
    # Return (idx, item_metadata) pairs so no more JSONs.
    result = []
    for idx, score, metadata in recommendations:
        # Convert metadata back to item format
        item = {
            'name': metadata.get('name', ''),
            'tags': metadata.get('tags', []),
            'description': metadata.get('description', ''),
            'price': metadata.get('price', ''),
            'location': metadata.get('location', ''),
            'address': metadata.get('address', ''),
            'cuisine': metadata.get('cuisine', ''),
            'rating': metadata.get('rating', ''),
            'hours': metadata.get('hours', ''),
            'phone': metadata.get('phone', ''),
            'website': metadata.get('website', ''),
        }
        # Remove empty fields
        item = {k: v for k, v in item.items() if v}
        result.append((idx, item))
    
    print(f"Returning {len(result)} {item_type} recommendations")
    return result

# Main page
@app.route('/')
def index():
    # Clear any existing session data when visiting the home page
    print("Clearing session on index page visit")
    session.clear()
    session['user_id'] = str(uuid4())
    responses_path = os.path.join(BASE_DIR, 'data', 'responses.csv')
    if not os.path.exists(responses_path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(responses_path), exist_ok=True)
        with open(responses_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id','selected_items','ratings','score'])
    return render_template('index.html')

# Questionnaire page
@app.route('/questionnaire')
def questionnaire():
    if 'max_selection' not in session:
        return render_template('set_max.html')
    # Reset the session if we're starting fresh
    if 'question_index' not in session:
        print("Initializing questionnaire session...")
        session['question_index'] = 0
        session['preferences'] = []
        session['selected_items'] = []
    
    # Debug session info
    print(f"Session contents: {session}")
    print(f"Current question index: {session.get('question_index')}")
    print(f"Total questions: {len(initial_questions)}")
    
    # Check if we've completed all questions
    if session.get('question_index', 0) >= len(initial_questions):
        print("All questions completed, redirecting to recommendations")
        return redirect(url_for('recommendations'))
    
    # Get the current question
    current_question = initial_questions[session['question_index']]
    progress = int((session['question_index']/len(initial_questions))*100)
    
    # Debug question info
    print(f"Rendering question: {current_question['question']}")
    print(f"Progress: {progress}%")
    print(f"Choices: {[c['text'] for c in current_question['choices']]}")
    
    return render_template('questionnaire.html', 
                          question=current_question, 
                          progress=progress)

# Handle questionnaire answers
@app.route('/answer', methods=['POST'])
def answer():
    print(f"Form data received: {request.form}")
    
    if 'choice' not in request.form:
        print("No choice submitted, redirecting back to questionnaire")
        return redirect(url_for('questionnaire'))
    
    # Check if we've completed all questions before processing
    current_question_index = session.get('question_index', 0)
    if current_question_index >= len(initial_questions):
        print("All questions completed, redirecting to recommendations")
        return redirect(url_for('recommendations'))
        
    choice_index = int(request.form.get('choice'))
    current_question = initial_questions[current_question_index]
    selected_choice = current_question['choices'][choice_index]
    
    print(f"Selected choice: {selected_choice['text']}")
    print(f"Selected tags: {selected_choice['tags']}")
    
    # Store the selected tags
    if 'preferences' not in session:
        session['preferences'] = []
    session['preferences'].extend(selected_choice['tags'])
    print(f"Updated preferences: {session['preferences']}")
    
    # Initialize recommenders and update with this choice
    restaurant_rec, activity_rec = initialize_recommenders()
    
    # Update both recommenders with the selected choice
    choice_text = " ".join(selected_choice['tags'])
    print(f"Updating recommender embeddings with: {choice_text}")
    # restaurant_rec.update_user_embedding(choice_text)
    # activity_rec.update_user_embedding(choice_text)
    restaurant_rec.exponential_moving_average_embedding_update(choice_text)
    activity_rec.exponential_moving_average_embedding_update(choice_text)
    
    # Move to the next question
    session['question_index'] = session.get('question_index', 0) + 1
    print(f"New question index: {session['question_index']}")
    
    return redirect(url_for('questionnaire'))

# Show recommendations page
@app.route('/recommendations')
def recommendations():
    print("Rendering recommendations page")
    
    # Get preferences from the session
    preferences = session.get('preferences', [])
    selected_items = session.get('selected_items', [])
    
    print(f"Current preferences: {preferences}")
    print(f"Selected items: {selected_items}")
    
    # Initialize recommenders if needed
    restaurant_rec, activity_rec = initialize_recommenders()
    
    # Update recommender with selected items
    for idx, item_type in selected_items:
        if item_type == 'restaurant':
            restaurant_rec.mark_as_recommended(idx)
        else:
            activity_rec.mark_as_recommended(idx)
    
    # Get recommendations
    restaurant_recs = get_recommendations(preferences, 'restaurant')
    activity_recs = get_recommendations(preferences, 'activity')
    
    print(f"Found {len(restaurant_recs)} restaurant recommendations")
    print(f"Found {len(activity_recs)} activity recommendations")
    
    # Get detailed information about selected items - skip if no local data
    selected_items_info = []
    activities, restaurants = load_data()
    
    # Only process selected items if we have local data, otherwise skip
    if activities or restaurants:
        for idx, item_type in selected_items:
            try:
                if item_type == 'restaurant' and idx < len(restaurants):
                    item = restaurants[idx]
                    selected_items_info.append({
                        'name': item['name'],
                        'tags': item.get('tags', []),
                        'type': 'restaurant'
                    })
                elif item_type == 'activity' and idx < len(activities):
                    item = activities[idx]
                    selected_items_info.append({
                        'name': item['name'],
                        'tags': item.get('tags', []),
                        'type': 'activity'
                    })
            except (IndexError, KeyError) as e:
                # Skip items that don't exist
                print(f"Error accessing item {idx} of type {item_type}: {e}")
    else:
        print("No local JSON data available - skipping selected items display")
    
    print(f"Selected items info: {selected_items_info}")
    
    return render_template('recommendations.html', 
                           restaurant_recommendations=restaurant_recs,
                           activity_recommendations=activity_recs,
                           selected_items_info=selected_items_info)

# Handle selection of an item to refine recommendations
@app.route('/select/<item_type>/<int:item_id>', methods=['POST'])
def select_item(item_type, item_id):
    print(f"Selected {item_type} with ID {item_id}")
    
    # Ensure we have a list of selected items
    if 'selected_items' not in session:
        session['selected_items'] = []

    # Only add item if not already in list
    if not any(idx == item_id and t == item_type for idx, t in session['selected_items']):
        session['selected_items'].append((item_id, item_type))
        session.modified = True
        print(f"Updated selected items: {session['selected_items']}")
    
    # Check limit IMMEDIATELY after adding the item
    max_sel = session.get('max_selection', 0)
    current_count = len(session['selected_items'])
    if current_count >= max_sel:
        print(f"Reached selection limit ({max_sel}). Redirecting to /rate now.")
        return redirect(url_for('rate'))
    
    # If we haven’t reached the limit, do the usual embedding logic
    restaurant_rec, activity_rec = initialize_recommenders()
    activities, restaurants = load_data()
    selected_item = None
    
    if item_type == 'restaurant' and item_id < len(restaurants):
        selected_item = restaurants[item_id]
    elif item_type == 'activity' and item_id < len(activities):
        selected_item = activities[item_id]
    
    # Update embeddings
    if selected_item:
        item_text = " ".join(selected_item.get('tags', []))
        print(f"Updating {item_type} recommender with tags: {item_text}")
    else:
        item_text = item_type
        print(f"No data found; using generic tag: {item_text}")

    if item_type == 'restaurant':
        restaurant_rec.mark_as_recommended(item_id)
        restaurant_rec.exponential_moving_average_embedding_update(item_text)
    else:
        activity_rec.mark_as_recommended(item_id)
        activity_rec.exponential_moving_average_embedding_update(item_text)
    
    # Update preferences
    if 'preferences' not in session:
        session['preferences'] = []
    if selected_item:
        session['preferences'].extend(selected_item.get('tags', []))
        print(f"Updated preferences: {session['preferences']}")
    else:
        print("No item data available to update preferences")
    
    return redirect(url_for('recommendations'))

# Reset the recommendation session
@app.route('/reset', methods=['POST'])
def reset():
    print("Resetting application state")
    global restaurant_recommender, activity_recommender
    session.clear()
    
    # Reset the recommenders
    if restaurant_recommender is not None:
        print("Resetting restaurant recommender")
        restaurant_recommender.user_embedding = None
        restaurant_recommender.recommended_items = set()
    
    if activity_recommender is not None:
        print("Resetting activity recommender")
        activity_recommender.user_embedding = None
        activity_recommender.recommended_items = set()
    
    return redirect(url_for('index'))

@app.route('/set_max', methods=['POST'])
def set_max():
    # Save the user's choice of max recommendations
    session['max_selection'] = int(request.form['max_selection'])
    # Initialize selected_items if not already
    session.setdefault('selected_items', [])
    return redirect(url_for('questionnaire'))

@app.route('/rate')
def rate():
    """
    Show the rating page and display human-readable names
    fetched directly from Pinecone metadata.
    """
    restaurant_rec, activity_rec = initialize_recommenders()
    selected_items_info = []

    print(f"DEBUG: Selected items from session: {session.get('selected_items', [])}")

    for idx, item_type in session.get('selected_items', []):
        pinecone_index = (restaurant_rec.index
                          if item_type == 'restaurant' else
                          activity_rec.index)

        print(f"DEBUG: Trying to fetch {item_type} with ID: {idx}")
        
        try:
            # Fix for newer Pinecone SDK - correct parameter name
            resp = pinecone_index.fetch(
                ids=[str(idx)],
                include_metadata=True  # This should work with newer SDK
            )
            
            print(f"DEBUG: Pinecone response for {item_type} {idx}: {resp}")
            
            # The response structure might be different in newer SDK
            if 'vectors' in resp:
                meta = resp['vectors'].get(str(idx), {}).get('metadata', {})
            else:
                # Try alternative response structure
                meta = resp.get(str(idx), {}).get('metadata', {})
            
            print(f"DEBUG: Metadata extracted: {meta}")
            
            name = meta.get('name') if meta.get('name') else f"{item_type.capitalize()} #{idx}"
            
            print(f"DEBUG: Final name for {item_type} {idx}: {name}")
            
        except Exception as e:
            print(f"Error fetching {item_type} {idx} from Pinecone: {e}")
            
            # Let's try an alternative approach - query instead of fetch
            try:
                print(f"DEBUG: Trying query approach for {item_type} {idx}")
                # Create a dummy query vector (all zeros) to get the item by filtering
                dummy_vector = [0.0] * 384  # Match your embedding dimension
                
                query_resp = pinecone_index.query(
                    vector=dummy_vector,
                    filter={"original_index": idx},  # Filter by the original index
                    top_k=1,
                    include_metadata=True
                )
                
                print(f"DEBUG: Query response: {query_resp}")
                
                if query_resp.get('matches') and len(query_resp['matches']) > 0:
                    meta = query_resp['matches'][0].get('metadata', {})
                    name = meta.get('name', f"{item_type.capitalize()} #{idx}")
                    print(f"DEBUG: Got name from query: {name}")
                else:
                    name = f"{item_type.capitalize()} #{idx}"
                    
            except Exception as e2:
                print(f"Query approach also failed: {e2}")
                name = f"{item_type.capitalize()} #{idx}"

        selected_items_info.append(
            (idx, {'name': name, 'type': item_type})
        )

    print(f"DEBUG: Final selected_items_info: {selected_items_info}")

    return render_template(
        'rate.html',
        selected_items_info=selected_items_info
    )



@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    user_id = session['user_id']
    # Build ratings dict and compute score (e.g. average)
    ratings = []
    for idx, _ in session['selected_items']:
        r = int(request.form[f'rating_{idx}'])
        ratings.append((idx, r))
    average_score = sum(r for _,r in ratings) / len(ratings)

    # Append to CSV - use BASE_DIR for cloud compatibility
    path = os.path.join(BASE_DIR, 'data', 'responses.csv')
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            selected_str = ";".join(f"{idx}:{t}" for idx,t in session['selected_items'])
            ratings_str = ";".join(f"{idx}:{r}" for idx,r in ratings)
            writer.writerow([user_id, selected_str, ratings_str, average_score])
    except Exception as e:
        # In production, it's better to log this rather than fail
        print(f"Warning: Could not write to responses.csv: {e}")

    return render_template('thanks.html', score=average_score)

if __name__ == '__main__':
    print("Starting Mexico City Recommender Web App...")
    # For AWS deployment, use 0.0.0.0 to listen on all interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)