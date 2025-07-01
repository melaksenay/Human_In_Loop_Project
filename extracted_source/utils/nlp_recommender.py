import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
from pinecone import Pinecone, ServerlessSpec

class NLPRecommender:
    def __init__(self, transformer_model='sentence-transformers/all-MiniLM-L6-v2', index_name=None):
        # Initialize the tokenizer and model to create embeddings (smaller model for memory efficiency)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.model = AutoModel.from_pretrained(transformer_model)
        
        # Pinecone configuration
        self.index_name = index_name
        self.index = None
        self.pc = None  # Pinecone client instance
        
        # Application state
        self.user_embedding = None
        self.recommended_items = set()   # Keep track of recommended items
        
        # Initialize Pinecone connection
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection and create index if needed."""
        # Get Pinecone API key from environment variable
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            print("Warning: PINECONE_API_KEY not found in environment variables")
            return
            
        # Initialize Pinecone with new SDK
        self.pc = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        if self.index_name and self.index_name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2 model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Fixed: use valid Pinecone region
                )
            )
        
        # Connect to the index
        if self.index_name:
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
    
    def populate_index(self, items_list):
        """Populate Pinecone index with item embeddings and full metadata."""
        if not self.index:
            print("Pinecone index not initialized")
            return
            
        print(f"Populating Pinecone index with {len(items_list)} items...")
        
        # Process items in batches
        batch_size = 100
        vectors = []
        
        for idx, item in enumerate(items_list):
            # Create embedding from item name and tags
            item_text = f"{item.get('name', '')} " + " ".join(item.get('tags', []))
            embedding = self.embed_text(item_text)
            
            # Prepare vector for Pinecone with ALL item data in metadata
            vector = {
                'id': str(idx),
                'values': embedding.tolist(),
                'metadata': {
                    'original_index': idx,
                    'name': item.get('name', ''),
                    'tags': item.get('tags', []),
                    # Store all possible fields from the item
                    'description': item.get('description', ''),
                    'price': item.get('price', ''),
                    'location': item.get('location', ''),
                    'address': item.get('address', ''),
                    'cuisine': item.get('cuisine', ''),
                    'rating': item.get('rating', ''),
                    'hours': item.get('hours', ''),
                    'phone': item.get('phone', ''),
                    'website': item.get('website', ''),
                    # Add any other fields dynamically
                    **{k: str(v) for k, v in item.items() if k not in ['name', 'tags'] and v is not None}
                }
            }
            vectors.append(vector)
            
            # Upsert batch when full
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors)
                vectors = []
                print(f"Upserted batch ending at index {idx}")
        
        # Upsert remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors)
            print(f"Upserted final batch of {len(vectors)} vectors")
        
        print("Pinecone index population complete")

    def embed_text(self, text):
        """Creates an embedding for the given text using the transformer model."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling to get a single vector
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def create_item_embedding(self, item):
        """Creates an embedding for a single item (restaurant or activity)."""
        item_text = f"{item.get('name', '')} " + " ".join(item.get('tags', []))
        return self.embed_text(item_text)

    def build_item_embeddings(self, items_list):
        """This method is deprecated - use populate_index instead for Pinecone."""
        print("Warning: build_item_embeddings is deprecated when using Pinecone. Use populate_index instead.")
        return {}

    def update_user_embedding(self, selection_text):
        """Update the user embedding based on new selection."""
        new_embedding = self.embed_text(selection_text)
        if self.user_embedding is None:
            self.user_embedding = new_embedding
        else:
            # Simple average of old and new embeddings
            self.user_embedding = (self.user_embedding + new_embedding) / 2
        
        # Normalize the embedding
        self.user_embedding = self.user_embedding / np.linalg.norm(self.user_embedding)

    def exponential_moving_average_embedding_update(self, selection_text, alpha=0.8):
        """Update the user embedding using exponential moving average."""
        new_embedding = self.embed_text(selection_text)
        if self.user_embedding is None:
            self.user_embedding = new_embedding
        else:
            self.user_embedding = (1-alpha) * self.user_embedding + (alpha) * new_embedding
        
        # Normalize the embedding
        self.user_embedding = self.user_embedding / np.linalg.norm(self.user_embedding)

    def mark_as_recommended(self, item_id):
        """Mark an item as already recommended to avoid showing it again."""
        self.recommended_items.add(item_id)

    def get_recommendations(self, k=5):
        """Get recommendations using Pinecone vector similarity search."""
        if not self.index:
            print("Pinecone index not initialized")
            return []
            
        if self.user_embedding is None:
            print("User embedding not initialized")
            return []
        
        # Prepare filter to exclude already recommended items
        filter_dict = None
        if self.recommended_items:
            filter_dict = {
                "original_index": {"$nin": list(self.recommended_items)}
            }
        
        # Query Pinecone for similar vectors
        try:
            results = self.index.query(
                vector=self.user_embedding.tolist(),
                top_k=k,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Extract recommendations with full metadata
            recommendations = []
            for match in results['matches']:
                item_id = int(match['metadata']['original_index'])
                score = match['score']
                metadata = match['metadata']
                recommendations.append((item_id, score, metadata))
            
            print(f"Found {len(recommendations)} recommendations from Pinecone")
            return recommendations
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []
