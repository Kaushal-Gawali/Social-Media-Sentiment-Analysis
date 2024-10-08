import numpy as np
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import seaborn as sns
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


class MyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analysis")
        self.geometry("800x600")
        # Frame for the first row of buttons
        frame_row1 = tk.Frame(self)
        frame_row1.pack(side=tk.TOP, pady=10)  # Pack the frame at the top with some padding

        # Button 1: Label Encoding
        self.btn_open = tk.Button(frame_row1, text="1. Label Encoding", command=self.label_encoding)
        self.btn_open.pack(side=tk.LEFT, padx=10)  # Pack the button to the left with some padding

        # Button 2: Heatmap
        self.btn_open = tk.Button(frame_row1, text="2. Heatmap and Relevant Attributes CSV", command=self.heatmap)
        self.btn_open.pack(side=tk.LEFT, padx=10)  # Pack the button to the left with some padding

        # Button 3: Normalization
        self.btn_open = tk.Button(frame_row1, text="3. Normalization", command=self.normalization)
        self.btn_open.pack(side=tk.LEFT, padx=10)  # Pack the button to the left with some padding

        # Button 4: Sigmoid
        self.btn_open = tk.Button(frame_row1, text="4. Sigmoid", command=self.Sigmoide)
        self.btn_open.pack(side=tk.LEFT, padx=10)  # Pack the button to the left with some padding

        # Frame for the second row of buttons
        frame_row2 = tk.Frame(self)
        frame_row2.pack(side=tk.TOP, pady=10)  # Pack the frame at the top with some padding

        # Button 5: SVM
        self.btn_open = tk.Button(frame_row2, text="5. SVM", command=self.apply_svm)
        self.btn_open.pack(side=tk.LEFT, padx=10)  # Pack the button to the left with some padding

        # Button 6: Decision Tree
        self.btn_open = tk.Button(frame_row2, text="6. Decision Tree", command=self.apply_decision_tree)
        self.btn_open.pack(side=tk.LEFT, padx=10)  # Pack the button to the left with some padding

        # Button 7: Random Forest
        self.btn_open = tk.Button(frame_row2, text="7. Random Forest", command=self.apply_random_forest)
        self.btn_open.pack(side=tk.LEFT, padx=10)  # Pack the button to the left with some padding

        # Button 8: Naive Bayes
        self.btn_open = tk.Button(frame_row2, text="8. Naive Bayes", command=self.NaiveBayes)
        self.btn_open.pack(side=tk.LEFT, padx=10)  # Pack the button to the left with some padding
        
        frame_row3 = tk.Frame(self)
        frame_row3.pack(side=tk.TOP, pady=10)  # Pack the frame at the top with some padding
        
        insert_label = tk.Label(frame_row3, text="Insert Text:")
        insert_label.pack(side=tk.LEFT, padx=0)  # Pack the label to the left with some padding
        self.input_text = tk.Entry(frame_row3)
        self.input_text.pack(side=tk.LEFT, padx=0)  # Pack the input field to the left with some padding
        
        insert_label = tk.Label(frame_row3, text="Sentiment:")
        insert_label.pack(side=tk.LEFT, padx=0)  # Pack the label to the left with some padding
        self.input_sentiment = tk.Entry(frame_row3)
        self.input_sentiment.pack(side=tk.LEFT, padx=0)  # Pack the input field to the left with some padding
        
        
        insert_label = tk.Label(frame_row3, text="Hashtags:")
        insert_label.pack(side=tk.LEFT, padx=0)  # Pack the label to the left with some padding
        self.input_hashtags = tk.Entry(frame_row3)
        self.input_hashtags.pack(side=tk.LEFT, padx=10)  # Pack the input field to the left with some padding
        
        
        insert_label = tk.Label(frame_row3, text="Retweets:")
        insert_label.pack(side=tk.LEFT, padx=0)  # Pack the label to the left with some padding
        self.input_retweets = tk.Entry(frame_row3)
        self.input_retweets.pack(side=tk.LEFT, padx=10)  # Pack the input field to the left with some padding
        
        
        insert_label = tk.Label(frame_row3, text="Likes:")
        insert_label.pack(side=tk.LEFT, padx=0)  # Pack the label to the left with some padding
        self.input_likes = tk.Entry(frame_row3)
        self.input_likes.pack(side=tk.LEFT, padx=10)  # Pack the input field to the left with some padding
        
        
        insert_label = tk.Label(frame_row3, text="Month:")
        insert_label.pack(side=tk.LEFT, padx=0)  # Pack the label to the left with some padding
        self.input_month = tk.Entry(frame_row3)
        self.input_month.pack(side=tk.LEFT, padx=10)  # Pack the input field to the left with some padding
        
        
        insert_label = tk.Label(frame_row3, text="Hour:")
        insert_label.pack(side=tk.LEFT, padx=0)  # Pack the label to the left with some padding
        self.input_hour = tk.Entry(frame_row3)
        self.input_hour.pack(side=tk.LEFT, padx=10)  # Pack the input field to the left with some padding
        
        

            # Insert button
        
        self.insert_button = tk.Button(frame_row3, text="Insert", command=self.insert_data)
        self.insert_button.pack(side=tk.LEFT, padx=10)  # Pack the button to the left with some padding
        self.insert_button = tk.Button(frame_row3, text="Check", command=self.check)
        self.insert_button.pack(side=tk.LEFT, padx=10)  # Pack the button to the left with some padding
        frame_row4 = tk.Frame(self)
        frame_row4.pack(side=tk.TOP, pady=10)  # Pack the frame at the top with some padding
        
        self.fig, self.graph_ax = plt.subplots(figsize=(4, 2), dpi=100)
        self.canvas_widget = FigureCanvasTkAgg( self.fig, master=self)
        self.canvas_widget.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.canvas_widget.draw()
        
    def label_encoding(self):
        # Load your dataset
        df = pd.read_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/sentimentdataset.csv')
        # Get only the columns with text attributes
        text_columns = df.select_dtypes(include=['object']).columns
        # Initialize LabelEncoder
        encoder = LabelEncoder()
        # Apply label encoding to each text attribute column
        for column in text_columns:
            df[column] = encoder.fit_transform(df[column])
        output_file_path = 'C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Encoded Dataset/encoded_dataset.csv'
        # Save the final DataFrame to a CSV file
        df.to_csv(output_file_path, index=False)

        messagebox.showinfo("Success", "label encoding applied and CSV file generated.")
    def heatmap(self):
        df = pd.read_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Encoded Dataset/encoded_dataset.csv')
        # Compute the correlation matrix
        corr = df.corr()
        corr_matrix = df.corr(numeric_only=True)
        target_corr = corr_matrix['Text']
        threshold = 0.1
        relevant_features = target_corr[abs(target_corr) > threshold].index.tolist()
        X = df[relevant_features]
        # Output path for the relevant dataset
        output_path_file = "C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Heatmap/relevant_dataset.csv"
        X.to_csv(output_path_file, index=False)
        # Set up the matplotlib figure
        plt.figure(figsize=(10, 8))
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, annot=True, fmt="0.02f", cmap='coolwarm', linewidths=0.5)
        # Add title
        plt.title('Correlation Heatmap of 13 Attributes')
        # Save the plot as a PDF
        plt.savefig('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Heatmap/correlation_heatmap.pdf')
        messagebox.showinfo("Success", "Correlation Heatmap and CSV file generated.")
        # Show the plot
        plt.show()
        self.update()
    def normalization(self):
        # Replace 'your_file.csv' with the path to your CSV file
        input_filename = 'C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Heatmap/relevant_dataset.csv'
        output_filename = 'C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Normalization/normalized_dataset.csv'
        data = self.read_csv(input_filename)

        # Extracting columns from the data
        sentiment = [float(row[1]) for row in data[1:]]  # Assuming sentiment is in the second column (index 1)
        retweets = [float(row[3]) for row in data[1:]]    # Assuming retweets is in the fourth column (index 3)
        likes = [float(row[4]) for row in data[1:]]       # Assuming likes is in the fifth column (index 4)
        month = [float(row[5]) for row in data[1:]]       # Assuming month is in the sixth column (index 5)
        hour = [float(row[6]) for row in data[1:]]        # Assuming hour is in the seventh column (index 6)
        texts = [row[0] for row in data[1:]]              # Assuming text is in the first column (index 0)
        hashtags = [row[2] for row in data[1:]]           # Assuming hashtags is in the third column (index 2)

        # Normalizing numerical attributes
        normalized_sentiment = self.normalize(sentiment)
        normalized_retweets = self.normalize(retweets)
        normalized_likes = self.normalize(likes)
        normalized_month = self.normalize(month)
        normalized_hour = self.normalize(hour)

        # Normalizing text and hashtags attributes
        text_dict = {text: i for i, text in enumerate(set(texts))}
        hashtag_dict = {hashtag: i for i, hashtag in enumerate(set(hashtags))}
        normalized_texts = self.normalize([text_dict[text] for text in texts])
        normalized_hashtags = self.normalize([hashtag_dict[hashtag] for hashtag in hashtags])

        # Updating the data with normalized values
        updated_data = []
        updated_data.append(data[0])  # Append header
        for i in range(len(data) - 1):
            row = [
                normalized_texts[i],
                normalized_sentiment[i],
                normalized_hashtags[i],
                normalized_retweets[i],
                normalized_likes[i],
                normalized_month[i],
                normalized_hour[i]
            ]
            updated_data.append(row)

        # Write the updated data to a new CSV file
        self.write_csv(output_filename, updated_data)
        messagebox.showinfo("Success", "normalization applied and CSV file generated.")
    def normalize(self,attribute):
        min_val = min(attribute)
        max_val = max(attribute)
        normalized_attribute = [(x - min_val) / (max_val - min_val) for x in attribute]
        return normalized_attribute

    # Function to read data from CSV file
    def read_csv(self,filename):
        data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read header row
            data.append(header)   # Append header to data
            for row in reader:
                data.append(row)
        return data

    # Function to write data to CSV file
    def write_csv(self,filename, data):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    def Sigmoide(self):
        df = pd.read_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Normalization/Normalized_dataset.csv')
        
        # Apply sigmoid function to all columns except non-numeric ones
       # Apply sigmoid function to 'Sentiment' column
        df['Sentiment_Sigmoid'] = self.sigmoid(df['Sentiment'])

# Define the threshold for binary conversion
        threshold = 0.664116585874051   # Adjust the threshold as needed

# Apply sigmoid function to all columns except non-numeric ones
        df_numeric = df.select_dtypes(include=[np.number])  # Select only numeric columns
        df[df_numeric.columns] = df_numeric.apply(lambda x: x.map(self.sigmoid))  # Apply sigmoid function to numeric columns

# Create a new column with binary labels based on the threshold
        df['Sentiment_Binary'] = (df['Sentiment_Sigmoid'] >= threshold).astype(int)
       
        
           
           # Optionally, display a message box to indicate success
        messagebox.showinfo("Success", "Sigmoid function applied and csv file generated.")
      #  messagebox.showinfo("Success", "Sigmoid function applied and CSV file generated.")

        # Define the file path for saving the CSV file
        output_file_path = 'C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Sigmoid/binary_sentiment_data.csv'

        # Save the DataFrame to a CSV file
        df.to_csv(output_file_path, index=False)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def apply_svm(self):
        self.update()
        self.update_idletasks()
       
        df = pd.read_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Sigmoid/binary_sentiment_data.csv')
        

        X_numeric = df[['Text', 'Hashtags', 'Likes', 'Retweets', 'Month', 'Hour']]
        y = df['Sentiment_Binary']

        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

        C_values = [0.1, 1, 10, 100, 1000]
        accuracy_values = []

        for C in C_values:
            svm_classifier = SVC(kernel='rbf', C=C, gamma='scale')
            svm_classifier.fit(X_train, y_train)
            y_pred = svm_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_values.append(accuracy)
        sum = 0
        flag = 0
        for i in accuracy_values:
            sum = sum + i
            flag = flag + 1
        svm_avg = sum/flag
        print("SVM accuracy:",svm_avg*100)

        self.graph_ax.clear()
        # self.graph_ax.plot(C_values, accuracy_values, marker='o', linestyle='-')
        # self.graph_ax.set_title('SVM Performance vs. Regularization Parameter (C)')
        # self.graph_ax.set_xlabel('Regularization Parameter (C)')
        # self.graph_ax.set_ylabel('Accuracy')
        # self.canvas_widget.draw()
        plt.clf()
        plt.plot(C_values, accuracy_values, marker='o', linestyle='-')
        plt.title('SVM Performance vs. Regularization Parameter (C)')
        plt.xlabel('Regularization Parameter (C)')
        plt.ylabel('Accuracy')
        plt.savefig('C:/Users/tusha/OneDrive/Desktop/DMWUI/svm_performance.pdf')
        messagebox.showinfo("Accuracy:", svm_avg*100)
    def apply_decision_tree(self):
        df = pd.read_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Sigmoid/binary_sentiment_data.csv')

        X_numeric = df[['Text', 'Hashtags', 'Likes', 'Retweets', 'Month', 'Hour']]
        y = df['Sentiment_Binary']

        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

        max_depth_values = range(1, 21)
        accuracy_values = []

        for depth in max_depth_values:
            decision_tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
            decision_tree.fit(X_train, y_train)
            y_pred = decision_tree.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_values.append(accuracy)
        sum = 0
        flag = 0
        for i in accuracy_values:
            sum = sum + i
            flag = flag + 1
        dt_avg = sum/flag
        print("Decision Tree accuracy:",dt_avg*100)
        self.graph_ax.clear()
        plt.clf()
        plt.plot(max_depth_values, accuracy_values, marker='o', linestyle='-')
        plt.title('Decision Tree Performance vs. max_depth ')
        plt.xlabel('max_depth')
        plt.ylabel('Accuracy')
        plt.savefig('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/decisiontree_performance.pdf')
        messagebox.showinfo("Accuracy:", dt_avg*100)
    def apply_random_forest(self):
        df = pd.read_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Sigmoid/binary_sentiment_data.csv')
        X_numeric = df[['Text', 'Hashtags', 'Likes', 'Retweets', 'Month', 'Hour']]
        y = df['Sentiment_Binary']

        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

        estimators_values = range(1, 101)
        accuracy_values = []

        for estimators in estimators_values:
            random_forest = RandomForestClassifier(n_estimators=estimators, random_state=42)
            random_forest.fit(X_train, y_train)
            y_pred = random_forest.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_values.append(accuracy)
        sum = 0
        flag = 0
        for i in accuracy_values:
            sum = sum + i
            flag = flag + 1
        rf_avg = sum/flag
        print("Random Forest Accuracy:",rf_avg*100)
        self.graph_ax.clear()
        plt.clf()
        plt.plot(estimators_values , accuracy_values, marker='o', linestyle='-')
        plt.title('Random Forest Performance vs. Number of Estimators ')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.savefig('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/randomforest_performance.pdf')
        messagebox.showinfo("Accuracy:", rf_avg*100)
    def NaiveBayes(self):
        # Step 1: Read data
        df = pd.read_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Sigmoid/binary_sentiment_data.csv')

        # Step 2: Data Preparation
        X_numeric = df[['Text', 'Hashtags', 'Likes', 'Retweets','Month','Hour']]  # Numerical features
        y = df['Sentiment_Binary']  # Replace 'Target_Label' with the name of your target variable

        # Step 3: Split data into training and testing sets
        X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

        # Step 4: Initialize Naive Bayes classifier
        naive_bayes = GaussianNB()

        # Step 5: Train the Naive Bayes model
        naive_bayes.fit(X_numeric_train, y_train)

        # Step 6: Model Evaluation
        y_pred = naive_bayes.predict(X_numeric_test)
        print(classification_report(y_test, y_pred))

        # Optionally, you can print the accuracy of the model
        accuracy = naive_bayes.score(X_numeric_test, y_test)
        print("Naive Bayes Accuracy:", accuracy*100)
        messagebox.showinfo("Accuracy:", accuracy*100)
    def insert_data(self):
        # Get user input text from the entry widget
        user_text = self.input_text.get()
        user_sentiment = self.input_sentiment.get()
        user_hashtags = self.input_hashtags.get()
        user_retweets = self.input_retweets.get()
        user_month = self.input_month.get()
        user_hour = self.input_hour.get()
        user_likes = self.input_likes.get()
        

        df = pd.read_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/sentimentdataset.csv').dropna(how='all')
    
        # Define default values for other columns
        default_values = {
            'Text': user_text,
            'Sentiment': user_sentiment,  # Assuming default sentiment value is 0
            'Timestamp': "15-01-2023  12:30:00",
            'User': "User001",
            'Platform':"Facebook",
            'Hashtags': user_hashtags,
            'Retweets': user_retweets,   # Assuming default number of retweets is 0
            'Likes': user_likes,      # Assuming default number of likes is 0
            'Country':"USA",
            'Year':"2023",
            'Month': user_month,      # Assuming default month value is 1 (January)
            'Day':"15",
            'Hour': user_hour        # Assuming default hour value is 0 (midnight)
            }
    
        # Create a new DataFrame with the new row
        new_row = {}
        new_row.update(default_values)
        new_df = pd.DataFrame([new_row])
    
        # Concatenate the original DataFrame with the new DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
        s = "Data inserted successfully"
        messagebox.showinfo("Operation: ",s)
        
        # Save the updated DataFrame back to the CSV file
        df.to_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/sentimentdataset.csv', index=False)

    def check(self):
        # Read the dataset
        df = pd.read_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Sigmoid/binary_sentiment_data.csv')
    
        # Get the last row of the DataFrame
        last_row = df.iloc[-1]
    
        # Extract features from the last row
        features = last_row[['Text', 'Hashtags', 'Likes', 'Retweets','Month','Hour']].values.reshape(1, -1)
    
        # Call the function to apply SVM model
        predicted_sentiment = self.apply_svmLast(features)
    
        # Display the predicted sentiment
        if predicted_sentiment == 1:
            messagebox.showinfo("Predicted Sentiment", "Positive")
        else:
            messagebox.showinfo("Predicted Sentiment", "Negative")

    def apply_svmLast(self, features):
        df = pd.read_csv('C:/Users/admin/Desktop/Programs cllg/MyProjects/DMW/Sigmoid/binary_sentiment_data.csv')
        X_numeric = df[['Text', 'Hashtags', 'Likes', 'Retweets','Month','Hour']]
        y = df['Sentiment_Binary']
    
        # Train SVM on the whole dataset
        svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')
        svm_classifier.fit(X_numeric, y)
    
        if features is not None:
            # If features are provided, predict the sentiment for the last row
            predicted_sentiment = svm_classifier.predict(features)
            return predicted_sentiment




        



if __name__ == "__main__":
    app = MyApp()
    app.mainloop()