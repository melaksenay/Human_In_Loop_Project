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
import os


# Import the NLP recommender - Fixed import path
from utils.nlp_recommender import NLPRecommender

app = Flask(__name__)
app.secret_key = 'mexico_city_recommender_2025'  # For session management

# Initialize recommenders
restaurant_recommender = None
activity_recommender = None

# Load data
def load_data():
    with open('data/activities.json', 'r', encoding='utf-8') as f:
        activities = json.load(f)
    with open('data/restaurants.json', 'r', encoding='utf-8') as f:
        restaurants = json.load(f)
    return activities, restaurants

# Initial questions (simplified from the original ActivityPreprocessor)
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
    """Initialize the NLP recommenders with data"""
    global restaurant_recommender, activity_recommender
    
    # Load data if not already loaded
    activities, restaurants = load_data()
    
    # Initialize restaurant recommender if needed
    if restaurant_recommender is None:
        print("Initializing restaurant recommender...")
        restaurant_recommender = NLPRecommender()
        restaurant_recommender.build_item_embeddings(restaurants)
    
    # Initialize activity recommender if needed
    if activity_recommender is None:
        print("Initializing activity recommender...")
        activity_recommender = NLPRecommender()
        activity_recommender.build_item_embeddings(activities)
    
    return restaurant_recommender, activity_recommender

def get_recommendations(preferences, item_type, count=5):
    """Get recommendations using the NLP recommender"""
    # Initialize recommenders if needed
    restaurant_rec, activity_rec = initialize_recommenders()
    
    # Select the appropriate recommender
    recommender = restaurant_rec if item_type == 'restaurant' else activity_rec
    
    # Make sure the user embedding is initialized with preferences
    if recommender.user_embedding is None:
        preference_text = " ".join(preferences)
        print(f"Initializing {item_type} embedding with: {preference_text[:100]}...")
        recommender.update_user_embedding(preference_text)
        # recommender.exponential_moving_average_embedding_update(preference_text)
    
    # Get recommendations
    print(f"Getting {count} {item_type} recommendations...")
    recommendations = recommender.get_recommendations(k=count)
    
    # Load the actual items
    activities, restaurants = load_data()
    items = restaurants if item_type == 'restaurant' else activities
    
    # Return (idx, item) pairs
    result = [(idx, items[idx]) for idx, _ in recommendations]
    print(f"Returning {len(result)} {item_type} recommendations")
    return result

# Main page
@app.route('/')
def index():
    # Clear any existing session data when visiting the home page
    print("Clearing session on index page visit")
    session.clear()
    session['user_id'] = str(uuid4())
    responses_path = os.path.join(app.root_path, 'data', 'responses.csv')
    if not os.path.exists(responses_path):
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
        
    choice_index = int(request.form.get('choice'))
    current_question = initial_questions[session['question_index']]
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
    restaurant_rec.update_user_embedding(choice_text)
    activity_rec.update_user_embedding(choice_text)
    # restaurant_rec.exponential_moving_average_embedding_update(choice_text)
    # activity_rec.exponential_moving_average_embedding_update(choice_text)
    
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
    
    # Get detailed information about selected items
    selected_items_info = []
    activities, restaurants = load_data()
    
    for idx, item_type in selected_items:
        try:
            if item_type == 'restaurant':
                item = restaurants[idx]
                selected_items_info.append({
                    'name': item['name'],
                    'tags': item.get('tags', []),
                    'type': 'restaurant'
                })
            else:
                item = activities[idx]
                selected_items_info.append({
                    'name': item['name'],
                    'tags': item.get('tags', []),
                    'type': 'activity'
                })
        except (IndexError, KeyError) as e:
            # Skip items that don't exist (shouldn't happen but just in case)
            print(f"Error accessing item {idx} of type {item_type}: {e}")
    
    print(f"Selected items info: {selected_items_info}")
    
    return render_template('recommendations.html', 
                           restaurant_recommendations=restaurant_recs,
                           activity_recommendations=activity_recs,
                           selected_items_info=selected_items_info)

# Handle selection of an item to refine recommendations
@app.route('/select/<item_type>/<int:item_id>', methods=['POST'])
def select_item(item_type, item_id):
    print(f"Selected {item_type} with ID {item_id}")
    # Only add if not already in selected items
    max_sel = session.get('max_selection', 0)
    current = session.get('selected_items', [])
    # If they’ve already reached their limit, go rate instead
    if len(current) >= max_sel:
        return redirect(url_for('rate'))
    
    # Load data
    activities, restaurants = load_data()
    
    # Initialize selected items if needed
    if 'selected_items' not in session:
        session['selected_items'] = []
    
    # Convert to integer and check for valid id
    item_id = int(item_id)
    
    # Only add if not already in selected items
    if not any(idx == item_id and type_ == item_type for idx, type_ in session['selected_items']):
        # Add the selected item
        session['selected_items'].append((item_id, item_type))
        print(f"Updated selected items: {session['selected_items']}")
    else:
        print(f"Item already in selected_items")
    
    # Debugging
    print(f"Session items before: {session.get('selected_items')}")
    
    # Make sure selected_items persists in the session
    session.modified = True
    
    # Debugging
    print(f"Session items after: {session.get('selected_items')}")
    
    # Initialize recommenders
    restaurant_rec, activity_rec = initialize_recommenders()
    
    # Get the selected item and update the embedding
    if item_type == 'restaurant':
        selected_item = restaurants[item_id]
        # Mark as recommended to avoid showing again
        restaurant_rec.mark_as_recommended(item_id)
        # Update user embedding with the selected item
        item_text = " ".join(selected_item.get('tags', []))
        print(f"Updating restaurant recommender with: {item_text}")
        restaurant_rec.update_user_embedding(item_text)
        # restaurant_rec.exponential_moving_average_embedding_update(item_text)
    else:
        selected_item = activities[item_id]
        # Mark as recommended to avoid showing again
        activity_rec.mark_as_recommended(item_id)
        # Update user embedding with the selected item
        item_text = " ".join(selected_item.get('tags', []))
        print(f"Updating activity recommender with: {item_text}")
        activity_rec.update_user_embedding(item_text)
        # activity_rec.exponential_moving_average_embedding_update(item_text)
    
    # Add tags from the selected item to preferences
    if 'preferences' not in session:
        session['preferences'] = []
    
    new_tags = selected_item.get('tags', [])
    session['preferences'].extend(new_tags)
    print(f"Updated preferences: {session['preferences']}")
    
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
    # Gather selected_items_info exactly as you do in /recommendations
    activities, restaurants = load_data()
    info = []
    for idx, t in session.get('selected_items', []):
        item = (restaurants if t=='restaurant' else activities)[idx]
        info.append((idx, {'name': item['name']}))
    return render_template('rate.html', selected_items_info=info)

@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    user_id = session['user_id']
    # Build ratings dict and compute score (e.g. average)
    ratings = []
    for idx, _ in session['selected_items']:
        r = int(request.form[f'rating_{idx}'])
        ratings.append((idx, r))
    average_score = sum(r for _,r in ratings) / len(ratings)

    # Append to CSV
    path = os.path.join(app.root_path, 'data', 'responses.csv')
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        selected_str = ";".join(f"{idx}:{t}" for idx,t in session['selected_items'])
        ratings_str  = ";".join(f"{idx}:{r}" for idx,r in ratings)
        writer.writerow([user_id, selected_str, ratings_str, average_score])

    return render_template('thanks.html', score=average_score)



if __name__ == '__main__':
    print("Starting Mexico City Recommender Web App...")
    app.run(debug=True)