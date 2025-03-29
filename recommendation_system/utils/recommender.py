import pandas as pd
import os
from recommendation_system.data.preprocessor import ActivityPreprocessor

class ActivityRecommender:
    def __init__(self, activities_path):
        # Initialize preprocessor
        self.preprocessor = ActivityPreprocessor()
        
        # Load activities and create embeddings
        self.activities_df = pd.read_json(activities_path)
        self.preprocessor.build_item_embeddings(self.activities_df)
        
        # Initialize user preferences through questionnaire
        self.preprocessor.present_questionnaire()
    
    def get_recommendations(self, n_recommendations=5):
        """Get top N recommendations based on current user preferences"""
        top_items = self.preprocessor.recommend_items(k=n_recommendations)
        
        # Get activity details for recommended items
        recommendations = []
        for item_id, similarity_score in top_items:
            activity = self.activities_df.iloc[item_id]
            recommendations.append({
                'name': activity['name'],
                'tags': activity['tags'],
                'similarity': round(similarity_score, 3)
            })
        
        return recommendations
    
    def update_preference(self, activity_id):
        """Update user preferences based on selected activity"""
        activity = self.activities_df.iloc[activity_id]
        # Use the activity's tags to update user preferences
        tags = activity.get('tags', [])
        if tags:
            self.preprocessor.update_user_embedding(" ".join(tags))
        return f"Updated preferences based on: {activity['name']}"

# Use it:
if __name__ == "__main__":
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils directory
    root_dir = os.path.dirname(os.path.dirname(current_dir))  # project root
    data_path = os.path.join(root_dir, "data", "restaurants.json")
    
    print(f"Looking for file at: {data_path}")  # Debug print
    
    recommender = ActivityRecommender(data_path)
    
    # recommender = ActivityRecommender("../../Human_In_Loop_Project/data/restaurants.json")
    
    # Get initial recommendations
    recommendations = recommender.get_recommendations()
    print("\nInitial Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} (similarity: {rec['similarity']})")
    
    # simulation:
    while True:
        try:
            choice = int(input("\nSelect an activity (1-5) or 0 to exit: "))
            if choice == 0:
                break
            if 1 <= choice <= len(recommendations):
                selected_id = choice - 1
                print(recommender.update_preference(selected_id))
                
                # Get the new recs with the updated preferences:
                recommendations = recommender.get_recommendations()
                print("\nNew Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec['name']} (similarity: {rec['similarity']})")
            else:
                print("Please enter a valid number")
        except ValueError:
            print("Please enter a valid number") 