import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class ActivityPreprocessor:
    def __init__(self, transformer_model='distilbert-base-uncased'):
        #get tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.model = AutoModel.from_pretrained(transformer_model)
        self.user_embedding = None  # This is the user tower representation --> WE'RE GETTING THIS BEFORE ANYTHING ELSE.
        self.item_embeddings = {}   # The item tower: mapping from item ID to embedding vector --> EMBEDS TEXT FROM JSON FILE.

        # Define initial questions with fixed choices and their tags
        self.initial_questions = [
            {
                "question": "What's your budget preference?",
                "choices": [
                    {"text": "Budget-friendly", "tags": ["price:low", "street-food", "free-activities"]},
                    {"text": "Mid-range", "tags": ["price:medium", "casual-dining", "paid-activities"]},
                    {"text": "Luxury", "tags": ["price:high", "fine-dining", "premium-experiences"]}
                ]
            },
            {
                "question": "What kind of activities interest you most?",
                "choices": [
                    {"text": "Cultural & Museums", "tags": ["cultural", "museums", "history", "art"]},
                    {"text": "Food & Dining", "tags": ["food", "restaurants", "culinary", "dining"]},
                    {"text": "Outdoor & Adventure", "tags": ["outdoor", "adventure", "nature", "active"]}
                ]
            },
            {
                "question": "What's your preferred experience type?",
                "choices": [
                    {"text": "Tourist Attractions", "tags": ["tourist", "popular", "guided-tours"]},
                    {"text": "Local Experiences", "tags": ["local", "authentic", "hidden-gems"]},
                    {"text": "Mix of Both", "tags": ["tourist", "local", "varied-experiences"]}
                ]
            }
        ]

    def embed_text(self, text):
        """
        Embed text using the model.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling: create a single vector from the token embeddings.
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding

    def update_user_embedding(self, selection_text):
        """
        Update user embedding using moving average (maybe change this)
        """
        new_embedding = self.embed_text(selection_text)
        if self.user_embedding is None:
            self.user_embedding = new_embedding
        else:
            # Simple moving average
            self.user_embedding = (self.user_embedding + new_embedding) / 2

    def create_item_embedding(self, tags):
        """
        Make embeddings for tags coming from the JSON file.
        For now, we don't use the title of the activity.
        The tags (e.g: ['outdoor', 'adventure', 'price:low']) are joined into a single string.
        """
        text = " ".join(tags)
        embedding = self.embed_text(text)
        return embedding

    def build_item_embeddings(self, activities_df):
        """
        Build and store the item tower representations.
        It assumes that `activities_df` includes a 'tags' column where each entry is a list of tags.
        """
        for idx, row in activities_df.iterrows():
            tags = row.get('tags', [])
            if tags:
                self.item_embeddings[idx] = self.create_item_embedding(tags)
    
    def recommend_items(self, k=5):
        """
        Recommend the top k items from the item tower based on cosine similarity
        with the current user tower representation.
        """
        if self.user_embedding is None:
            raise ValueError("User embedding not initialized. Update it with a selection first.")
        if not self.item_embeddings:
            raise ValueError("Item embeddings not available. Build the item tower first.")
        
        # Prepare a list of item IDs and their embeddings.
        item_ids = list(self.item_embeddings.keys())
        embeddings = np.array([self.item_embeddings[i] for i in item_ids])
        
        # Compute cosine similarities between the user embedding and each item embedding.
        user_vec = self.user_embedding.reshape(1, -1)
        similarities = cosine_similarity(user_vec, embeddings)[0]
        
        # Get the indices of the top k most similar items.
        top_indices = np.argsort(similarities)[::-1][:k]
        top_items = [(item_ids[i], similarities[i]) for i in top_indices]
        return top_items

    def present_questionnaire(self):
        """Present questions to user via console and collect their choices"""
        for question_data in self.initial_questions:
            print("\n" + question_data["question"])
            for i, choice in enumerate(question_data["choices"], 1):
                tags_str = ", ".join(choice["tags"])
                print(f"{i}. {choice['text']} (Tags: {tags_str})")
            
            while True:
                try:
                    choice = int(input("Enter your choice (1-3): "))
                    if 1 <= choice <= 3:
                        selected = question_data["choices"][choice-1]
                        # Update user embedding with the tags from their choice
                        self.update_user_embedding(" ".join(selected["tags"]))
                        break
                    else:
                        print("Please enter a number between 1 and 3")
                except ValueError:
                    print("Please enter a valid number")

#testing the preprocessor:
if __name__ == "__main__":
    preprocessor = ActivityPreprocessor()
    preprocessor.present_questionnaire()
