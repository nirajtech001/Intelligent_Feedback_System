
---

# **Showcase Arena Platform**  

A virtual platform that provides students with an immersive project showcase experience, designed to enhance engagement and visibility beyond the limitations of a physical event.

## **Objective**  
To give students a better experience than a physical showcase, enabling them to:  
- Present their projects to a wider audience.  
- Gain valuable feedback from peers, teachers, and other stakeholders.  
- Overcome challenges like limited resources, poor infrastructure, and FOMO (fear of missing out).  

---

## **Key Features**  
1. **Arena & Announcements**  
   - A virtual gallery where students pitch their projects.  
   - Visitors can choose attractive pitches to explore in detail.  

2. **Project Presentation**  
   - Students can showcase their projects using Google Slides or YouTube videos embedded via iframes.  

3. **Live Interaction (Cubicle)**  
   - Zoom calls and virtual rooms for real-time discussions.  
   - Visitors can ask questions and leave feedback.  

4. **Feedback Mechanism**  
   - DT Social Scorecard for collecting feedback.  

5. **Gamification**  
   - PowerBI Dashboard to track progress and scores.  
   - Live Dashboard to keep track of visitor engagement and feedback.  

6. **Memories and Recognition**  
   - AI Powered digital **Slam Book** capturing interactions and key takeaways.   

---

## **Team Composition**  

| Role               | Responsibility             | Team Member |  
|--------------------|----------------------------|-------------|  
| Lead Developer     | Backend, DFDs of modules   | Aditya      |  
| Project Manager    | UI of scaffolding          | Anjika      | 
| Data Scientist     | ML module for Slam Book    | Niraj       | 
| Dev1               | Slam Book                  | Mujahid     |  
| Dev2               | Arena, Gallery             | Omkar       |  
| Dev3               | Cubicle                    | Srilakshmi  |  
 

---

## **Technical Implementation**  

Here's the updated README section with **Data Loading and Preprocessing** included:

---

# **Project Showcase Virtual Platform**

## **1. Installing Required Libraries**

```bash
pip install pandas scikit-learn tensorflow
```

## **2. Importing Libraries**

```python
import numpy as np
import pandas as pd
import requests
import zipfile
import os
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import Constant
```

## **3. Loading the Dataset**

```python
df = pd.read_csv("/content/Dataset_SB.csv")
```

### Dataset Sample:

| Participant      | Compliments                                  | Questions                                | Feedback                            | Generated Summary/Takeaway                      |
|------------------|----------------------------------------------|------------------------------------------|-------------------------------------|----------------------------------------|
| Alice Williams   | Your project's creativity is impressive.     | How did you come up with the idea?       | Excellent execution. Consider...    | Impressive project with creative...    |
| Bob Davis        | Clear and concise documentation.             | What challenges did you face?            | Great attention to detail...        | Clear documentation and attention...   |

---

## **4. Data Preprocessing**

### **a. Extracting Features and Target:**
```python
X = df[['Compliments', 'Questions', 'Feedback']]
y = df['Summary/Takeaway']
```

### **b. Combining Text Columns:**
```python
# Concatenate the Compliments, Questions, and Feedback columns
X_combined = X['Compliments'] + ' ' + X['Questions'] + ' ' + X['Feedback']
```

### **c. Creating a New DataFrame for Training:**
```python
df1 = pd.DataFrame({'Response': X_combined, 'Summary/Takeaway': y})
```

---

## **5. Text Tokenization and Padding**

```python
# Initialize and fit tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df1['Response'].tolist())

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df1['Response'].tolist())

# Pad sequences to ensure uniform input length
padded_sequences = pad_sequences(sequences)
```

---

This setup prepares the dataset for the next steps in building and training the LSTM-based model.

### 2. **Embedding Layer**  
- Downloaded and loaded **GloVe** embeddings (100d).  
- Created an embedding matrix using pre-trained embeddings.  

### 3. **Model Architecture**  
- Used an LSTM-based sequential model:  
  - Embedding layer initialized with GloVe vectors.  
  - Two LSTM layers (300 units each) with dropout for regularization.  
  - Dense layer with softmax activation for classification.  

### 4. **Model Compilation**  
- Compiled using `adam` optimizer and categorical cross-entropy loss.  

### **Code Snippet for Model Definition**  

```python  
model = Sequential()  
model.add(Embedding(input_dim=vocab_size,  
                    output_dim=embedding_dim,  
                    input_length=X.shape[1],  
                    embeddings_initializer=Constant(embedding_matrix),  
                    trainable=False))  
model.add(LSTM(300, return_sequences=True))  
model.add(Dropout(0.3))  
model.add(LSTM(300))  
model.add(Dense(vocab_size, activation='softmax'))  

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
print(model.summary())  
```  

---

## **Installation and Setup**  
1. **Clone the Repository**  
   ```bash  
   git clone <repo-url>  
   cd showcase-arena  
   ```  
2. **Download GloVe Embeddings**  
   ```bash  
   wget https://nlp.stanford.edu/data/glove.6B.zip  
   unzip glove.6B.zip  
   ```  
3. **Install Dependencies**  
   ```bash  
   pip install -r requirements.txt  
   ```  
4. **Run the Model Training**  
   ```bash  
   python train_model.py  
   ```  

---

## **Future Enhancements**  
- Expand the ML module to include sentiment analysis from feedback.  
- Introduce advanced gamification elements for user engagement.  
- Enable multi-language support for a global audience.  


---  
