import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class NLPRecommender:
    def __init__(self, transformer_model='distilbert-base-uncased'):
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.model = AutoModel.from_pretrained(transformer_model)
        self.user_embedding = None  # User's preference embedding
        self.item_embeddings = {}   # Mapping from item ID to embedding vector
        self.recommended_items = set()  # Keep track of recommended items

    def embed_text(self, text):
        """
        Create an embedding for the given text using the transformer model.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling to get a single vector from token embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding

    def update_user_embedding(self, selection_text):
        """
        Update the user embedding based on new selection.
        Uses moving average when updating existing embedding.
        """
        new_embedding = self.embed_text(selection_text)
        if self.user_embedding is None:
            self.user_embedding = new_embedding
        else:
            # Simple moving average for updating
            self.user_embedding = (self.user_embedding + new_embedding) / 2
    
    def exponential_moving_average_embedding_update(self, selection_text, alpha = 0.8):
        """
        Update the user embedding based on new selection.
        Uses exponential moving average when updating existing embedding.
        """
        new_embedding = self.embed_text(selection_text)
        if self.user_embedding is None:
            self.user_embedding = new_embedding
        else:
            #Exponential moving average
            self.user_embedding = (1-alpha) * self.user_embedding + (alpha) * new_embedding
            self.user_embedding /= np.linalg.norm(self.user_embedding) #normalize the embedding

    def create_item_embedding(self, item, weight_multiplier= 3):
        """
        Create an embedding for an item based on its tags.
        What changed: I added a weight multiplier to the price tags. This is kinda quick and dirty, but it works.
        The idea is to give more weight to the price tags when creating the embedding.
        """
        # Create text representation from tags
        tags = item.get('tags', [])
        # name = item.get('name', '')
        text = " ".join(tags)
        
        price_tags = [tag for tag in tags if tag.startswith('price:')]
        other_tags = [tag for tag in tags if not tag.startswith("price:")]
        
        weighted_tags = price_tags*weight_multiplier  + other_tags
        text = " ".join(weighted_tags)
        
        # # Add name for additional context
        '''We might not need this.'''
        # if name:
        #     text = name + " " + text
        return self.embed_text(text)

    def build_item_embeddings(self, items_list):
        """
        Build embeddings for all items in the given list.
        """
        for idx, item in enumerate(items_list):
            self.item_embeddings[idx] = self.create_item_embedding(item)
        return self.item_embeddings

    def get_recommendations(self, k=5):
        """
        Get top-k recommendations based on cosine similarity.
        """
        if self.user_embedding is None:
            raise ValueError("User embedding not initialized. Update it with a selection first.")
        
        if not self.item_embeddings:
            raise ValueError("Item embeddings not available. Build the item tower first.")
        
        # Prepare item IDs and embeddings
        item_ids = list(self.item_embeddings.keys())
        
        # Filter out already recommended items
        available_items = [idx for idx in item_ids if idx not in self.recommended_items]
        
        if not available_items:
            return []  # No more items to recommend
        
        # Get embeddings for available items
        embeddings = np.array([self.item_embeddings[i] for i in available_items])
        
        # Compute cosine similarity with user embedding
        user_vec = self.user_embedding.reshape(1, -1)
        similarities = cosine_similarity(user_vec, embeddings)[0]
        
        # Create pairs of (item_id, similarity)
        similarity_pairs = list(zip(available_items, similarities))
        
        # Sort by similarity (descending)
        similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k items
        return similarity_pairs[:k]

    def mark_as_recommended(self, item_id):
        """
        Mark an item as recommended to avoid duplicate recommendations.
        """
        self.recommended_items.add(item_id)