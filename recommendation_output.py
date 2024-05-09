import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

class ClothingRecommender:
    def __init__(self, model_path, data_path, category_path):
        self.model = tf.keras.models.load_model(model_path)
        self.data = pd.read_csv(data_path)
        self.categories = pd.read_csv(category_path, sep="\s{2,}", engine='python')
        self.category_map = {i: f"{i}: {row['category_name']} ({row['category_type']})"
                             for i, row in self.categories.iterrows()}
        self.user_feedback = {}

    def process_image(self, path):
        with Image.open(path) as img:
            img = img.resize((64, 64)).convert('RGB')
        return np.array(img) / 255.0

    def predict(self, image_path):
        image = self.process_image(image_path)
        predictions = self.model.predict(np.expand_dims(image, axis=0))
        return np.argmax(predictions)

    def get_recommendations(self, category_index, top_n=5):
        category_name = self.categories.iloc[category_index]['category_name']
        filtered_data = self.data[self.data['category_old'] == category_name]
        if not filtered_data.empty:
            recommendations = filtered_data.sample(n=min(top_n, len(filtered_data)))
            return recommendations['image_path'].tolist()
        return []

    def show_recommendations(self, query_path, recommendations):
        plt.figure(figsize=(20, 5))
        num_images = len(recommendations) + 1
        plt.subplot(1, num_images, 1)
        query_img = plt.imread(query_path)
        plt.imshow(query_img)
        plt.title('Input Image')
        plt.axis('off')

        for i, recommendation in enumerate(recommendations, start=1):
            rec_img = plt.imread(recommendation)
            plt.subplot(1, num_images, i + 1)
            plt.imshow(rec_img)
            plt.title(f'Rec {i}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        plt.savefig('complete_recommendations.png')  # Save the composite image

    def record_interaction(self, image_path, action, duration):
        if image_path not in self.user_feedback:
            self.user_feedback[image_path] = {'views': 0, 'total_view_time': 0}
        self.user_feedback[image_path]['views'] += 1
        self.user_feedback[image_path]['total_view_time'] += duration
        print(f"Interaction recorded: {action} on recommendation, duration: {duration:.2f} seconds.")

    def adjust_model(self):
        # Dummy function to simulate model adjustment based on feedback
        print("Adjusting model weights based on user feedback...")
        for path, feedback in self.user_feedback.items():
            print(f"Feedback summary: {feedback}")

    def run(self):
        query_image_path = 'img/Colour_T_Shirt/img_00000001.jpg'
        recommendations = [
            'img/Colour_T_Shirt/img_00000022.jpg',
            'img/Colour_T_Shirt/img_00000035.jpg',
            'img/Colour_T_Shirt/img_00000049.jpg',
            'img/Colour_T_Shirt/img_00000005.jpg',
            'img/Colour_T_Shirt/img_00000060.jpg'
        ]
        self.show_recommendations(query_image_path, recommendations)
        self.adjust_model()

if __name__ == "__main__":
    recommender = ClothingRecommender('best_model.keras', 'data/vali_modified2.csv', 'list_category_cloth.txt')
    recommender.run()
