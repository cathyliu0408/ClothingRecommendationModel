### README for Deep Shopping Content-Based Image Retrieval System

---



## Overview
Clothing Recommendation System  leverages a ResNet-based deep learning model to provide a sophisticated content-based image retrieval system, tailored specifically for fashion e-commerce platforms. By analyzing clothing images to identify fabric types, styles, and patterns, the system offers precise recommendations that closely match user inquiries, thereby enhancing the online shopping experience through visually similar product suggestions.

## Key Features
- **Visual Similarity Recommendations:** Utilizes visual cues from clothing images to suggest similar items, enhancing the shopping experience.
- **Deep Feature Analysis:** Employs deep learning to capture complex patterns in fabric textures, cuts, and styles.
- **Dynamic Adaptation:** Continuously improves recommendations by learning from user interactions and feedback.

## Repository Contents
- `fashion_input.py`: Manages image data input and preprocessing.
- `train_n_test.py`: Facilitates model training and testing, monitoring performance metrics.
- `hyper_parameters.py`: Configures model hyperparameters for optimal performance.
- `list_category_cloth.txt`: Contains the categories and types of clothing classified by the model.
- `preprocessing.py`: Handles image resizing and normalization for training readiness.
- `recommendation_output.py`: Generates and outputs fashion recommendations.
- `simple_resnet.py`: Implements the custom ResNet model tailored for fashion image classification.

## Installation and Setup
1. **Clone the Repository:**
   ```
   git clone [repository-url]
   ```
2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Data Preparation
Utilize the dataset from DeepFashion's Attribute Prediction project available at [DeepFashion Attribute Prediction](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html). Ensure data is properly cleaned and balanced as per the specifications in `fashion_input.py`.

## Model Training and Recommendations
1. **Configure Hyperparameters:**
   Adjust settings in `hyper_parameters.py` based on your computational resources and dataset specifics.
2. **Run the Training Script:**
   ```
   python train_n_test.py
   ```
3. **Generate Recommendations:**
   Use the trained model to suggest fashion items:
   ```
   python recommendation_output.py
   ```

## Motivation and Background
The Clothing Recommendation was created to address the "new" trend of personalization and visual coherence in online fashion retail. Traditional systems often rely solely on user metadata or simplistic image tags, lacking a deep understanding of visual content. Deep Shopping advances this by providing highly personalized and visually aligned recommendations.

## Technical Challenges and Solutions
- **Dataset Balancing:** Ensured a balanced representation of various clothing styles and types to avoid model bias.
- **Color Feature Integration:** Enhanced model capability to include color analysis for matching user color preferences accurately.
- **Overfitting Management:** Implemented data augmentation and hyperparameter tuning to improve model generalization.

## Results and Impact
- **High Training Accuracy:** Achieved up to 98.81% accuracy, indicating strong learning capabilities.
- **Color Integration Success:** Improved recommendation relevance by effectively incorporating color analysis.
- **User Feedback Incorporation:** Tailored recommendations further based on user interactions, enhancing satisfaction.

## Future Directions
- **Dataset Diversity:** Plans to include more diverse fashion styles and body types.
- **User Feedback Enhancements:** Improve mechanisms for capturing and utilizing user feedback to refine recommendations.
- **Recommendation Type** Refine the recommendations and allow for more queries at once. 

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any inquiries or issues, please open an issue on GitHub or contact us at cyl4949@nyu.edu or zy1216@nyu.edu

