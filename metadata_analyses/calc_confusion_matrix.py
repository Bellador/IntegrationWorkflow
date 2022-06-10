#!/usr/bin/env python
# coding: utf-8

# ------------------------------DEPENDENCIES----------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import ConfusionMatrixDisplay
# ----------------------------------------------------------------------------------------------------------------------

''' 
Code is used to test new rk models from google colab by inferring on various test sets
to check performance to select the final (production) model version for running our workflow
'''

class RedKiteModel:
    '''
    Loads a transfer-learning model which is trained to differentiate Red Kites (Milvus milvus) from other bird species)
    Based on ResNet50
    '''
    def __init__(self, MODEL_PATH=None, threshold=0.5):
        self.MODEL_PATH = MODEL_PATH
        self.THRESHOLD = threshold
        self.model = self.load_custom_model()
        print(f'[*] Red Kite model with threshold {self.THRESHOLD} loaded')

    def load_custom_model(self):
        # load model via tensorflow load_model function
        return load_model(self.MODEL_PATH)

    def detection(self, image_path, img_name, IMAGE_SIZE=(224, 224)):
        path_to_load = os.path.join(image_path, img_name)
        img = image.load_img(path_to_load, target_size=IMAGE_SIZE)
        # resized_img = cv2.resize(norm_loaded_img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        # reshaped_img = resized_img.reshape(1, 224, 224, 3)
        x = image.img_to_array(img)
        # rescale image in the same way as during training
        x = x / 255
        x = np.expand_dims(x, 0) # axis=
        a = self.model.predict(x)
        score = a[0][1]
        # check threshold to be considered red kite
        # index 0 means no_red_kite, 1 means red_kite
        if score >= self.THRESHOLD:
            prediction = 1  # red_kite
            class_name = 'Red Kite'
        else:
            prediction = 0  # not_red_kite
            class_name = 'Not Red Kite'
        print(f'RED KITE MODEL - image: {img_name}, prediction: {prediction}, class name: {class_name}, score: {score}')
        return class_name, score, prediction

if __name__ == '__main__':
    # chose if images to read are TP or FP red kite images
    TP_RK_IMAGES = "./filtered_flickr_red_kite_final_model_test"
    TN_RK_IMAGES = "./ebird_not_red_kite_final_model_test"
    RED_KITE_MODEL_PATH = r"<ENTER HERE - PATH TO MODEL STORAGE>"

    redkite_model_inst = RedKiteModel(MODEL_PATH=RED_KITE_MODEL_PATH, threshold=0.5)

    # initiate variables to store prediction result for performance calculation
    TP_PREDs = 0
    FP_PREDS = 0
    TN_PREDS = 0
    FN_PREDS = 0
    # generate list that holds the true labels of the test data
    y_test = []
    # generate list that holds the predicted labels of the test data
    y_pred = []
    # iterate over files
    for dir_index, tuple_ in enumerate([('TP', TP_RK_IMAGES), ('TN', TN_RK_IMAGES)], 1):
        TO_QUERY, image_dir = tuple_
        for index, image_ in enumerate(os.listdir(image_dir), 1):
            try:
                rk_class_name, rk_score, prediction = redkite_model_inst.detection(image_dir, image_)  # norm_image
                # append to y_test the true label
                if TO_QUERY == 'TP':
                    y_test.append(1)
                elif TO_QUERY == 'TN':
                    y_test.append(0)
                # append to y_pred the predicted label
                y_pred.append(prediction)
                # append to single variables the prediction outcome
                if rk_class_name == 'Red Kite':
                    if TO_QUERY == 'TP':
                        TP_PREDs += 1
                    elif TO_QUERY == 'TN':
                        FP_PREDS += 1
                elif rk_class_name == 'Not Red Kite':
                    if TO_QUERY == 'TP':
                        FN_PREDS += 1
                    elif TO_QUERY == 'TN':
                        TN_PREDS += 1
            except Exception as e:
                print(f'ERROR: {e}\n skipping...')
            print(f'\rIndex: {index}  ', end='')

    print(f'TP: {TP_PREDs}, TN: {TN_PREDS}, FP: {FP_PREDS}, FN: {FN_PREDS}')

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['not_red_kite', 'red_kite'])

    disp.plot(cmap=plt.cm.Blues)
    save_path = os.path.join(r"./models/confusion_matrix_plots", f'{redkite_model_inst.model_name[:-3]}.jpg')
    plt.show()
