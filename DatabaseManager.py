# Import required libraries for image processing, database operations, and machine learning
import tensorflow as tf
import os
import csv
import numpy as np
import pandas as pd
import mysql.connector
from io import BytesIO
from PIL import Image
import faiss
from sklearn.decomposition import PCA

# Import TensorFlow components for easier access
keras = tf.keras
models = tf.keras.models
applications = tf.keras.applications
layers = tf.keras.layers

# Import specific components from EfficientNet for image preprocessing and model
preprocess_input = applications.efficientnet.preprocess_input
EfficientNetB0 = applications.EfficientNetB0


class LogoSimilarityClassifier:
    def __init__(self, db_config, logo_directory):
        """
        Initialize the classifier with database and logo directory

        Args:
            db_config (dict): Database connection parameters containing host, user, password, and database name
            logo_directory (str): Path to the directory containing logo images
        """
        # Store database configuration and logo directory path
        self.db_config = db_config
        self.logo_directory = logo_directory
        self.connection = None
        self.cursor = None

        # Initialize EfficientNetB0 model for feature extraction
        # Using pre-trained weights from ImageNet, excluding top classification layer
        # Using average pooling to reduce spatial dimensions
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        self.feature_extractor = models.Model(
            inputs=base_model.input,
            outputs=base_model.output
        )

    def connect_database(self):
        """
        Establish connection to MySQL database and create necessary table structure
        Table includes columns for image ID, name, image data, and extracted features
        """
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()

            # Create table with features column as LONGBLOB to store large feature vectors
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                image_data LONGBLOB NOT NULL,
                features LONGBLOB NULL
            )
            """)
        except Exception as e:
            print(f"Database connection error: {e}")

    def reset_table(self):
        """
        Reset the table structure completely
        Drops existing table and recreates it with the correct schema
        Useful for starting fresh with a clean database
        """
        try:
            # Drop existing table if it exists
            self.cursor.execute("DROP TABLE IF EXISTS images")

            # Recreate table with improved schema
            self.cursor.execute("""
            CREATE TABLE images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                image_data LONGBLOB NOT NULL,
                features LONGBLOB NULL
            )
            """)

            self.connection.commit()
            print("Table reset successfully")
        except Exception as e:
            print(f"Error resetting table: {e}")
            self.connection.rollback()

    def insert_images(self):
        """
        Insert images from the logo directory into the database
        - Resizes images to 224x224 (EfficientNet input size)
        - Converts images to RGB format
        - Stores images as binary data in the database
        """
        for filename in os.listdir(self.logo_directory):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                # Process image
                file_path = os.path.join(self.logo_directory, filename)
                try:
                    # Open and preprocess image
                    img = Image.open(file_path).convert('RGB')
                    img = img.resize((224, 224))

                    # Convert image to bytes for database storage
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    # Insert image data into database
                    query = "INSERT INTO images (name, image_data) VALUES (%s, %s)"
                    self.cursor.execute(query, (filename, img_bytes))
                    self.connection.commit()
                    print(f"Inserted '{filename}'")

                except Exception as e:
                    print(f"Error processing '{filename}': {e}")

    def extract_features(self):
        """
        Extract features from all logos using EfficientNet model
        - Loads images from database
        - Preprocesses images for model input
        - Extracts features using EfficientNet
        - Reduces feature dimensionality using PCA
        - Stores features back in database
        Returns:
            tuple: (reduced_features, names) - PCA-reduced feature vectors and corresponding image names
        """
        # Fetch images from database
        self.cursor.execute("SELECT id, name, image_data FROM images")
        results = self.cursor.fetchall()

        # Prepare images for feature extraction
        logos = []
        names = []
        for row in results:
            names.append(row[1])
            img_data = row[2]
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            logos.append(img_array)

        # Preprocess and extract features using EfficientNet
        logos_np = np.array(logos)
        logos_np = preprocess_input(logos_np)
        features = self.feature_extractor.predict(logos_np, batch_size=32, verbose=1)

        # Reduce feature dimensionality using PCA for better clustering
        reducer = PCA(n_components=50)
        reduced_features = reducer.fit_transform(features)

        # Store reduced features in database
        update_query = "UPDATE images SET features = %s WHERE name = %s"
        for name, feat in zip(names, reduced_features):
            feat_bytes = feat.tobytes()
            self.cursor.execute(update_query, (feat_bytes, name))
        self.connection.commit()

        return reduced_features, names

    def cluster_logos(self, features, names):
        """
        Cluster logos using FAISS and K-means clustering
        Args:
            features: PCA-reduced feature vectors
            names: List of image names
        Returns:
            dict: Clusters with cluster IDs as keys and lists of image names as values
        """
        # Initialize FAISS index for similarity search
        dimension = features.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(features.astype('float32'))

        # Perform K-means clustering with 10 clusters
        kmeans = faiss.Kmeans(dimension, 10, niter=20, verbose=True)
        kmeans.train(features.astype('float32'))
        labels = kmeans.assign(features.astype('float32'))

        # Organize images into clusters
        clusters = {}
        for i, label in enumerate(labels[0]):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(names[i])

        return clusters

    def export_clusters(self, clusters):
        """
        Export clustering results to CSV and HTML formats
        Args:
            clusters: Dictionary of clusters with image names
        Creates:
            - clusters_output.csv: CSV file with cluster assignments
            - clusters_output.html: Visual HTML representation of clusters
        """
        # Prepare data for CSV export
        data_csv = []
        for cluster_id, image_list in clusters.items():
            if not image_list:
                continue  # Skip empty clusters

            # Extract company name from first image in cluster
            first_image_name = image_list[0]
            representative_company = first_image_name.split("_")[0]

            for image_name in image_list:
                data_csv.append([representative_company, image_name])

        # Write CSV file
        with open('clusters_output.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Cluster_Name', 'Image_Name'])
            writer.writerows(data_csv)

        # Prepare data for HTML visualization
        data_html = []
        for cluster_id, image_list in clusters.items():
            if not image_list:
                continue

            # Extract representative company name for cluster label
            first_image_name = image_list[0]
            representative_company = first_image_name.split("_")[0]
            cluster_label = f"Group {representative_company}"

            # Create HTML image tags for each logo in cluster
            for image_name in image_list:
                image_path = os.path.join(self.logo_directory, image_name)
                if os.path.exists(image_path):
                    img_tag = f'<img src="{image_path}" alt="{image_name}" width="100" height="100">'
                    data_html.append([cluster_label, img_tag])

        # Create HTML visualization using pandas DataFrame
        df = pd.DataFrame(data_html, columns=['Cluster_Label', 'Logo'])
        html_content = df.to_html(
            classes='table table-striped table-bordered',
            index=False,
            border=0,
            escape=False
        )

        # Create complete HTML document with Bootstrap styling
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Clustered Logos</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container">
                <h1 class="my-4">Clustered Logos</h1>
                {html_content}
            </div>
        </body>
        </html>
        """

        # Write HTML file
        with open('clusters_output.html', 'w') as file:
            file.write(html_template)

        print("Clusters exported to CSV and HTML")

    def close_connection(self):
        """
        Safely close database connection and cursor
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()


def main():
    """
    Main execution function that demonstrates the complete workflow:
    1. Database connection
    2. Table reset
    3. Image insertion
    4. Feature extraction
    5. Clustering
    6. Result export
    """
    # Database configuration parameters
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'cleopatra1',
        'database': 'Logos'
    }

    # Directory containing logo images
    logo_directory = 'logos'

    # Initialize classifier instance
    classifier = LogoSimilarityClassifier(db_config, logo_directory)

    try:
        # Execute complete workflow
        classifier.connect_database()
        classifier.reset_table()
        classifier.insert_images()
        features, names = classifier.extract_features()
        clusters = classifier.cluster_logos(features, names)
        classifier.export_clusters(clusters)

    finally:
        # Ensure database connection is properly closed
        classifier.close_connection()


if __name__ == "__main__":
    main()