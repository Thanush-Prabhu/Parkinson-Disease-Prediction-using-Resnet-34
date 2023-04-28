# Parkinson-Disease-Prediction-using-Resnet-34(by using Intel oneAPI)
Our project aims to develop a deep learning-based system for the prediction of Parkinson's disease. Parkinson's disease is a neurodegenerative disorder that affects millions of people worldwide that causes unintended or uncontrollable movements, such as shaking, stiffness, and difficulty with balance and coordination, and early diagnosis is crucial for better management of the disease.Symptoms usually begin gradually and worsen over time.

![8525-parkinsons-disease (1)](https://user-images.githubusercontent.com/105986236/235113789-26fb65fb-ee98-4bb5-bc50-4e9aa9da0843.jpg)

# Parkinson vs Healthy Drawings
![1_Ve21pukGvC4M5P_2wC3RPQ](https://user-images.githubusercontent.com/105986236/235114649-4c99eb23-8bf9-4fd9-b4c1-128fb49a5a46.jpg)

# Dataset
The dataset used in this project contains handwritten images of spirals and waves. The data was taken directly from Kevin Mader's Parkinsons Drawing. Attached is the medical article documenting how the data was recorded, how the subjects were chosen, and other features they recorded such as pen pressure, speed, etc. The dataset consists of 600 images in total, with 300 images of each class.

The images were preprocessed and resized to 224x224 pixels before being fed into the ResNet-34 pretrained model for training and validation.

# Model
To build our Parkinson's disease prediction model, we utilized the ResNet-34 pre-trained convolutional neural network architecture. With this approach, we were able to achieve an accuracy of 95% in distinguishing between healthy and Parkinson's disease handwriting patterns. Our model was trained on a dataset of handwritten spiral and wave drawings, as described earlier. which shows that this deep learning-based system has the potential to be used as a screening tool for early detection of Parkinson's disease.

# Data Agumentation

We applied custom image transformations to augment the dataset in our project using the fastai library and PIL. The "CustomTransform" class applies various enhancements to the input image, creating diverse data for training a robust model. These transformations were used in our "DataBlock" for preprocessing.
![image](https://user-images.githubusercontent.com/105986236/235123001-2f3a7718-d632-4d9e-913a-c7c209d14c30.png)

# Results after Training
![image](https://user-images.githubusercontent.com/105986236/235117723-d2408d75-3ab5-4e0e-b469-ecadf0fb45e3.png)
# Optimizing our Model
# Using unfreeze
By using the unfreeze technique, we were able to fine-tune our pre-trained model and observed an improvement in the model's predictions. The prediction accuracy increased slightly after unfreezing and fine-tuning the model.

![image](https://user-images.githubusercontent.com/105986236/235117866-3697cac3-f5f4-4257-89e4-60f5aef121ea.png)
# Learning rate
![image](https://user-images.githubusercontent.com/105986236/235120313-0a80f0c8-2e13-4554-8a06-f1fdbf48d1c2.png)

After plotting and understanding the learning rate, we were able to achieve 95% accuracy on our model, which is a significant improvement from our initial results. The learning rate played a critical role in achieving this level of accuracy.

![image](https://user-images.githubusercontent.com/105986236/235118051-72d75573-ce5e-4112-9c69-62af2935d814.png)

# Usage

HOW TO RUN THIS PROJECT:

✅STEP 1: Download the dataset from the resource mentioned in the project.

✅STEP 2: Clone the GitHub repository.

✅STEP 3: Extract the notebook from the repository and open it in Jupyter Notebook.

✅STEP 4: Perform data augmentation and preprocessing on the dataset.

✅STEP 5: Run the code to train and test the model using the ResNet-34 pre-trained model.

✅STEP 6: If desired, unfreeze the pre-trained layers of the model to check for further improvement.

✅STEP 7: Plot and understand the learning rate to optimize the model.

✅STEP 8: Execute the code in Intel oneAPI cloud platform to significantly reduce execution time and GPU usage.

✅STEP 9: Evaluate the model accuracy and use it for Parkinson's disease prediction.

# Conclusion

The development of a deep learning-based system for the prediction of Parkinson's disease is an important step towards early diagnosis and better management of the disease. In this project, we utilized the ResNet-34 model and a dataset of handwritten spiral and wave drawings to achieve a prediction accuracy of 95%. We also applied techniques such as data augmentation and fine-tuning to further optimize our model. The unfreeze and learning rate experiments were conducted to fine-tune the model, which led to a significant improvement in its performance. The results of this study suggest that deep learning-based systems have the potential to serve as a valuable tool in the early diagnosis and prediction of Parkinson's disease. Future work in this area could involve exploring the use of other deep learning architectures and expanding the dataset to include a wider range of drawing patterns.
