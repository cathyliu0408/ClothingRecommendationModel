import tensorflow as tf
from fashion_input import prepare_df, load_data_numpy
from simple_resnet import ResNet50
import os
import numpy as np
from hyper_parameters import get_arguments
from tensorflow.keras.callbacks import TensorBoard
import datetime
import pandas as pd
import matplotlib.pyplot as plt

args = get_arguments()

TRAIN_DIR = 'logs_' + args.version + '/'

def get_dataset(df, batch_size):
    """Assumes `load_data_numpy` returns suitable numpy arrays for x and y."""
    images, labels, _ = load_data_numpy(df)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset.shuffle(buffer_size=1024).batch(batch_size)

def plot_learning_curve(history, output_file):
    """Plots and saves the learning curve using training history."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='train_accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='val_accuracy', color='green')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def train():
    train_df = prepare_df(args.train_path, usecols=['image_path', 'category', 'x1', 'y1', 'x2', 'y2'])
    vali_df = prepare_df(args.vali_path, usecols=['image_path', 'category', 'x1', 'y1', 'x2', 'y2'])

    print("Training dataset loaded, first 5 rows:")
    print(train_df.head())
    
    train_dataset = get_dataset(train_df, args.batch_size)
    val_dataset = get_dataset(vali_df, args.batch_size)

    model = ResNet50(input_shape=(64, 64, 3), classes=6)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=[tensorboard_callback], verbose=1)

    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    model.save(os.path.join(TRAIN_DIR, 'final_model.h5'))
    print(f"Model saved in {TRAIN_DIR}")

def load_category_labels(filepath):
    """Loads category labels from a text file, skipping the first row."""
    # Skip the initial numeric line and correctly read the column headers
    categories = pd.read_csv(filepath, delim_whitespace=True, skiprows=1)
    # Print to confirm the correct columns were read
    print("Columns in the loaded file:", categories.columns.tolist())
    # Extract the category names into a list
    return categories['category_name'].tolist()

def visualize_classification_tables(test_df, true_labels, predictions, label_names, output_file_correct, output_file_incorrect):
    """Creates and saves classification analysis tables with images for correct and incorrect classifications."""
    # Separate indices for correctly and incorrectly classified samples
    correct_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true == pred]
    incorrect_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true != pred]

    def create_classification_table(df_indices, table_title, output_file):
        """Helper function to create a table of images and labels for given indices."""
        # Creating a DataFrame for easier analysis
        classification_df = pd.DataFrame(columns=['Label', 'Image Path', 'True Label', 'Predicted Label'])
        for i, index in enumerate(df_indices[:10]):  # Limit to 10 for visualization
            image_path = test_df.iloc[index]['image_path']
            true_label = label_names[true_labels[index]]
            predicted_label = label_names[predictions[index]]

            classification_df = classification_df._append({
                'Label': i + 1,
                'Image Path': image_path,
                'True Label': true_label,
                'Predicted Label': predicted_label
            }, ignore_index=True)

        fig, axs = plt.subplots(1, 11, figsize=(22, 5))
        axs[0].axis('off')

        # Create the text table in the first subplot
        cell_text = [[row['Label'], row['True Label'], row['Predicted Label']] for _, row in classification_df.iterrows()]
        axs[0].table(cellText=cell_text, colLabels=['Label', 'True Label', 'Predicted Label'], cellLoc='center', loc='center')

        # Add images in individual subplots
        for i, ax in enumerate(axs[1:]):
            if i < len(classification_df):
                img_path = classification_df.iloc[i]['Image Path']
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.axis('off')

        fig.suptitle(table_title, fontsize=14)
        fig.savefig(output_file)

    # Create separate tables for correct and incorrect classifications
    create_classification_table(correct_indices, 'Correctly Classified Images', output_file_correct)
    create_classification_table(incorrect_indices, 'Misclassified Images', output_file_incorrect)

# Example usage in the `test` function
def test():
    test_df = prepare_df(args.test_path, usecols=['image_path', 'category', 'x1', 'y1', 'x2', 'y2'], shuffle=False)
    test_df = test_df.iloc[-25:, :]  # Assuming the last 25 entries are the test set

    test_dataset = get_dataset(test_df, 25)

    model = tf.keras.models.load_model(os.path.join(TRAIN_DIR, 'final_model.h5'))

    predictions_prob = model.predict(test_dataset)
    predictions = np.argmax(predictions_prob, axis=-1)

    images, labels, _ = load_data_numpy(test_df)

    # Load the category names
    category_filepath = 'list_category_cloth.txt'
    label_names = pd.read_csv(category_filepath, delim_whitespace=True, skiprows=1)['category_name'].tolist()

    # Output files for the analysis tables
    output_correct_path = os.path.join(TRAIN_DIR, 'correct_classification_analysis.png')
    output_incorrect_path = os.path.join(TRAIN_DIR, 'misclassification_analysis.png')

    # Visualize both correctly and incorrectly classified tables
    visualize_classification_tables(test_df, labels, predictions, label_names, output_correct_path, output_incorrect_path)

    # Save predictions
    np.save(os.path.join(TRAIN_DIR, 'predictions.npy'), predictions_prob)
    print('Prediction array has shape', predictions_prob.shape)

if __name__ == "__main__":
    train()
    test()
