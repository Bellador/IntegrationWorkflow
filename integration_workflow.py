#!/usr/bin/env python
# coding: utf-8

# ------------------------------DEPENDENCIES----------------------------------------------------------------------------
import os
import re
import sys
import csv
import cv2
import time
import shutil
import tarfile
import argparse
import numpy as np
import urllib.request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
# ----------------------------------------------------------------------------------------------------------------------
class DataLoader:
    '''
    Loads the Flickr dataset from CSV and functions as iterator by providing always the next post with metadata
    '''
    def __init__(self, data_path):
        self.data_path = data_path
        self.csv_filepath, \
        self.image_folder_path, \
        self.included_folderpath, \
        self.excluded_folderpath, \
        self.included_logpath, \
        self.excluded_logpath, \
        self.missing_logpath, \
        self.model_decisions_logpath, \
        self.model_settings_logpath= self.setup()
        self.csv_iterator = self.generator_iterator()
        print('[*] CSV iterator loaded')

    def setup(self):
        '''
        1. identify the filepaths for the csv file and the images from the general data_path which stands for a workspace
        2. creates necessary folders that hold the output of the workflow namely the included and excluded posts
        3. create also included and excluded posts in .txt files for easy usage and storage later on together with the initial .csv file
        :return:
        '''
        image_folder_pattern = r'images'
        for file in os.listdir(self.data_path):
            if file.endswith('.csv'):
                csv_filepath = os.path.join(self.data_path, file)
            # find image folder
            try:
                re.search(image_folder_pattern, file).group(0)
                image_folder_path = os.path.join(self.data_path, file)
            except:
                continue

        included_folderpath = os.path.join(self.data_path, 'included')
        excluded_folderpath = os.path.join(self.data_path, 'excluded')
        logs_folderpath = os.path.join(self.data_path, 'logs')

        # holds image IDs that were included
        included_logpath = os.path.join(logs_folderpath, 'included_postlog.txt')
        # holds image IDs that were excluded
        excluded_logpath = os.path.join(logs_folderpath, 'excluded_postlog.txt')
        # holds missing image IDs
        missing_logpath = os.path.join(logs_folderpath, 'missing_postlog.txt')
        # holds the decision every node in the model made for each post ID
        model_decisions_logpath = os.path.join(logs_folderpath, 'model_decisions_logpath.csv')
        # holds model settings for given inference
        model_settings_logpath = os.path.join(logs_folderpath, 'model_settings_logpath.txt')

        return csv_filepath, image_folder_path, included_folderpath, excluded_folderpath, included_logpath, excluded_logpath, \
               missing_logpath, model_decisions_logpath, model_settings_logpath


    def generator_iterator(self):
        f_ = open(self.csv_filepath, newline='', encoding='utf-8')
        csv_iterator = csv.reader(f_, delimiter=',', quotechar='|')
        return csv_iterator

    def next_post(self):
        return next(self.csv_iterator)


class TextAnalyser:
    '''
    class capable of finding taxonomic references in Flickr metadata e.g. descriptions, tags etc.
    '''
    taxa_dict = {
        'Milvus milvus': {
            'en_name': ['red kite', 'redkite'],
            'de_name': ['roter Milan', 'Rotmilan', 'Gabelweih', 'Königsweihe'],
            'fr_name': ['milan royal', 'milanroyal'],
            'it_name': ['nibbio reale', 'nibbioreale'],
            'sp_name': ['milano real', 'milanoreal'],
            'nl_name': ['rode wouw', 'rodewouw']
        }
    }

    def __init__(self, text_data, taxon='Milvus milvus'):
        self.text_data = text_data
        self.target_taxon = taxon
        self.latin_name_present = False
        self.common_name_present = False
        self.processed_text = self.processing_text()
        self.latin_taxon_name_extractor()
        self.common_taxon_name_extractor()

    def processing_text(self):
        '''
        preprocess textual components of a Flickr post and its metadata
        :return:
        '''
        title = self.text_data[3]
        description = self.text_data[21]
        tags = self.text_data[11]
        tag_string = ' '.join(tags.split(';'))
        processed_text = title + ' ' + description + ' ' + tag_string
        return processed_text

    def latin_taxon_name_extractor(self):
        try:
            re.search(self.target_taxon, self.processed_text, re.IGNORECASE).group(0) #match =
            self.latin_name_present = True
        except AttributeError as e:
            pass

    def common_taxon_name_extractor(self):
        # english taxon name
        en_pattern = '|'.join(TextAnalyser.taxa_dict[self.target_taxon]['en_name'])
        try:
            re.search(en_pattern, self.processed_text, re.IGNORECASE).group(0)
            self.common_name_present = True
            return True
        except AttributeError as e:
            pass
        # german taxon names
        de_pattern = '|'.join(TextAnalyser.taxa_dict[self.target_taxon]['de_name'])
        try:
            re.search(de_pattern, self.processed_text, re.IGNORECASE).group(0)
            self.common_name_present = True
            return True
        except AttributeError as e:
            pass
        # french taxon name
        fr_pattern = '|'.join(TextAnalyser.taxa_dict[self.target_taxon]['fr_name'])
        try:
            re.search(fr_pattern, self.processed_text, re.IGNORECASE).group(0)
            self.common_name_present = True
            return True
        except AttributeError as e:
            pass
        # italian taxon name
        it_pattern = '|'.join(TextAnalyser.taxa_dict[self.target_taxon]['it_name'])
        try:
            re.search(it_pattern, self.processed_text, re.IGNORECASE).group(0)
            self.common_name_present = True
            return True
        except AttributeError as e:
            pass
        # spanish taxon name
        sp_pattern = '|'.join(TextAnalyser.taxa_dict[self.target_taxon]['sp_name'])
        try:
            re.search(sp_pattern, self.processed_text, re.IGNORECASE).group(0)
            self.common_name_present = True
            return True
        except AttributeError as e:
            pass
        # netherlands taxon name
        nl_pattern = '|'.join(TextAnalyser.taxa_dict[self.target_taxon]['nl_name'])
        try:
            re.search(nl_pattern, self.processed_text, re.IGNORECASE).group(0)
            self.common_name_present = True
            return True
        except AttributeError as e:
            pass


class PretrainedBirdModel:
    '''
    Loads a pre-trained model to specifically differentiate between birds and multiple other classes.
    ResNet101 model initialised with COCO weights
    '''
    def __init__(self, threshold=0.65):
        '''
        adjustable parameters below
        '''
        # check TensorFlow 2 Detection Model Zoo for most recent models
        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
        self.MODEL_DATE = '20200711'
        self.MODEL_NAME = 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'
        self.IMAGE_SIZE = (640, 640) # needs to be adapted to the loaded model's input layer!
        self.threshold = threshold  # 0.6 prediction accuracy to be added to the image and text 2 speech
        self.prediction_model, self.PATH_TO_LABELS = self.load_model()
        # load label index that matches output integer with literal class names
        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)
        # self.detection(self.prediction_model, self.PATH_TO_LABELS, default_img=self.TEST_IMAGES_PATH)
        print(f'[*] bird model with threshold {self.threshold} loaded')

    def load_model(self):
        MODELS_DIR = './models/pre-trained-models'
        # Download and extract model
        MODEL_TAR_FILENAME = self.MODEL_NAME + '.tar.gz'
        MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
        MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + self.MODEL_DATE + '/' + MODEL_TAR_FILENAME
        PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
        PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(self.MODEL_NAME, 'checkpoint/'))
        PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(self.MODEL_NAME, 'pipeline.config'))
        if not os.path.exists(PATH_TO_CKPT):
            print('[*] downloading model. This may take a while... ', end='')
            urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
            tar_file = tarfile.open(PATH_TO_MODEL_TAR)
            tar_file.extractall(MODELS_DIR)
            tar_file.close()
            os.remove(PATH_TO_MODEL_TAR)
            print('[+] model download complete.')

        # Download labels file
        LABEL_FILENAME = 'mscoco_label_map.pbtxt'
        LABELS_DOWNLOAD_BASE = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
        PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(self.MODEL_NAME, LABEL_FILENAME))
        if not os.path.exists(PATH_TO_LABELS):
            print('[*] downloading label file... ', end='')
            urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
            print('[+] label file download complete.')
        # Load the model
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)
        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()
        return detection_model, PATH_TO_LABELS

    def extract_prediction_metric(self, detections, label_id_offset=1):
        boxes = detections['detection_boxes'][0].numpy()
        classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
        scores = detections['detection_scores'][0].numpy()

        # find highest bird specific class score
        highest_bird_class_score = 0
        for i_ in range(boxes.shape[0]):
            class_name_ = self.category_index[classes[i_]]['name']
            score_ = scores[i_]
            if class_name_ == 'bird' and score_ > highest_bird_class_score:
                highest_bird_class_score = score_
        # find class with score above threshold
        highest_general_class_score = 0
        highest_general_class_name = None
        for i in range(boxes.shape[0]):
            class_name = self.category_index[classes[i]]['name']
            score = scores[i]
            if score > self.threshold and score > highest_general_class_score:
                # store it in the object_dict for now
                highest_general_class_name = class_name
                highest_general_class_score = score
        return highest_general_class_name, highest_general_class_score, highest_bird_class_score


    @staticmethod
    def get_model_detection_function(model):
        '''
        detection on single image
        :return:
        '''

        @tf.function
        def detect_fn(input_tensor):
            """Detect objects in image."""
            image, shapes = model.preprocess(input_tensor)
            prediction_dict = model.predict(input_tensor, shapes)
            detections = model.postprocess(prediction_dict, shapes)
            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn
        #return highest_bird_model_score_class_name, highest_bird_model_score, bird_class_score


class RedKiteModel:
    '''
    Loads a transfer-learning model which is trained to differentiate Red Kites (Milvus milvus) from other bird species)
    Based on ResNet50
    '''
    def __init__(self, MODEL_PATH=None, threshold=0.5):
        self.THRESHOLD = threshold
        self.MODEL_PATH = MODEL_PATH
        self.model = self.load_custom_model()
        print(f'[*] Red Kite model with threshold {self.THRESHOLD} loaded')

    def load_custom_model(self):
        # load model via tensorflow load_model function
        return load_model(self.MODEL_PATH)

    def detection(self, image_path, img_name, IMAGE_SIZE=(224, 224)):
        img = image.load_img(os.path.join(os.path.join(image_path)), target_size=IMAGE_SIZE)
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
        return class_name, score

def write_model_decision_log(decisions, bird_class_score, rk_score, included, path):
    # add model scores plus finals decision (included 0 or 1) to decisions
    decisions = decisions + [bird_class_score, rk_score, included]
    assert len(decisions) == 8, 'Error: decision list not len == 8'
    decisions = [str(decision) for decision in decisions]
    decision_str = ';'.join(decisions) + '\n'
    with open(path, 'at', encoding='utf-8') as f:
        f.write(decision_str)


if __name__ == '__main__':
    # ---- RED KITE DETECTION WORKFLOW ---------------------------------------------------------------------------------
    '''
    This script automatically separates a Flickr corpus into Red Kite relevant and non-relevant datasets
    This is done by analysing a posts textual and visual components to make a prediction
    Images are analysed by two CNNs, one for detecting birds and the following specifically for detecting Red Kites
    A post's text (title, description and tags) are analysed for six language variations of the name Red Kite 
    '''
    # defines the main working directory. Ihe script searches in that location for the Flickr data (folder *images* and file *.csv)
    # the output of the workflow will also be stored and logged in that location
    WORKLOAD_PATH = '<ENTER HERE - ABSOLUTE PATH>'
    RED_KITE_MODEL_PATH = r"<ENTER HERE - PATH TO MODEL STORAGE>"
    # check for CUDA (GPU) enabled device for inference.
    print(f'[*] CUDA devices found: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    print("[*] Num GPUs available: ", len(tf.config.list_physical_devices('GPU')))
    # uncomment below to turn OFF CUDA
    ## os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # ----  INITIATE NECESSARY MODULES ---------------------------------------------------------------------------------
    print('[*] loading essential workflow components')
    # load necessary files and create necessary folder structure for output and logging
    data_loader_inst = DataLoader(WORKLOAD_PATH)
    # initialise bird object detection model
    bird_model_inst = PretrainedBirdModel(threshold=0.5)
    # get detection function with build model TensorFlow graph
    bird_detection_fn = bird_model_inst.get_model_detection_function(bird_model_inst.prediction_model)
    # initialise Red Kite image classification model
    redkite_model_inst = RedKiteModel(threshold=0.5, MODEL_PATH=RED_KITE_MODEL_PATH)
    # saving model settings to log file
    bird_model_confidence_threshold = bird_model_inst.threshold
    rk_model_confidence_threshold = redkite_model_inst.THRESHOLD
    with open(data_loader_inst.model_settings_logpath, 'w', encoding='utf-8') as f:
        f.write(f'bird_model_name: {bird_model_inst.MODEL_NAME}\n')
        f.write(f'bird_model_confidence_threshold: {bird_model_confidence_threshold}\n')
        f.write(f'rk_model_name: {redkite_model_inst.model_name}\n')
        f.write(f'rk_model_confidence_threshold: {rk_model_confidence_threshold}\n')
    # iterating over flickr post corpus
    stopwatch = time.time()
    for index, post in enumerate(data_loader_inst.csv_iterator):
        # timer on inference times
        end_iter = time.time()
        comp_time = end_iter - stopwatch
        stopwatch = end_iter
        if index != 0:
            print(f'-------- PROCESSING TIME INDEX {index - 1}: {comp_time} sec --------------------------------------')
        # set some path variables
        post_id = post[5]
        post_image_path = os.path.join(data_loader_inst.image_folder_path, f'{post_id}.jpg')
        # check if image exists (Flickr image might not have been downloaded due to deletion or error)
        if not os.path.isfile(post_image_path):
            # add id to not_handled list
            with open(data_loader_inst.missing_logpath, 'a', encoding='utf-8') as f:
                f.write(f'{post_id}\n')
            print(f'[-] ID {post_id} missing - reason: image not found')
            continue
        # initialise TextAnalyser which checks for latin and common taxon names within the post's text
        text_analyser_inst = TextAnalyser(post)
        # initialise model scores for workflow decision logging
        bird_class_score = 0
        rk_score = 0
        # start logging information + decisions for the given post
        model_decision_tracker = [post_id]
        # check latin name
        if text_analyser_inst.latin_name_present:
            # add to model decision tracker - latin name present
            model_decision_tracker = model_decision_tracker + [1, '-', '-', '-']
            # latin name found -> include post and copy image to 'included folder'
            print(f'[+] ID {post_id} included - reason: latin taxon name')
            shutil.copyfile(post_image_path, os.path.join(data_loader_inst.included_folderpath, f'{post_id}.jpg'))
            with open(data_loader_inst.included_logpath, 'a', encoding='utf-8') as f:
                f.write(f'{post_id}\n')
            # write model decision string
            write_model_decision_log(model_decision_tracker, bird_class_score, rk_score, 1,
                                     data_loader_inst.model_decisions_logpath)
            # clean up - delete text analyser inst
            del text_analyser_inst
            # go to next post
            continue
        # no latin name present
        else:
            # add to model decision tracker - latin name not present
            model_decision_tracker.append(0)
            # now definitely an image analysis will be performed
            loaded_image = cv2.imread(post_image_path)
            # normalise image (necessary for both models)
            # resizing of the image will be done separately, since the two models have different input layer resolutions!
            norm_loaded_img = loaded_image
            # check for common taxon names present in text
            if text_analyser_inst.common_name_present:
                # add to model decision tracker - common name present
                model_decision_tracker.append(1)
                # common taxon name found -> pass to bird model image classification model for inference
                # modify image for inference including resizing to appropriate input layer format
                resized_image = cv2.resize(norm_loaded_img, bird_model_inst.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                input_tensor = tf.convert_to_tensor(np.expand_dims(resized_image, 0), dtype=tf.float32)
                detections, predictions_dict, shapes = bird_detection_fn(input_tensor)
                highest_bird_model_score_class_name, highest_bird_model_score, bird_class_score = bird_model_inst.extract_prediction_metric(
                    detections, label_id_offset=1)
                print(f'BIRD MODEL - image: {post_id}, class name: {highest_bird_model_score_class_name}, score: {highest_bird_model_score}')

                if bird_class_score > bird_model_inst.threshold:
                    # add to model decision tracker - bird skipped and red kite present
                    model_decision_tracker.append(1)
                    model_decision_tracker.append('-')
                    # likely to include a red kite -> include post and copy image to 'included folder'
                    print(f'[+] ID {post_id} included - reason: common taxon name + bird detected, score: {bird_class_score}')
                    shutil.copyfile(post_image_path, os.path.join(data_loader_inst.included_folderpath, f'{post_id}.jpg'))
                    with open(data_loader_inst.included_logpath, 'a', encoding='utf-8') as f:
                        f.write(f'{post_id}\n')
                    # write model decision string
                    write_model_decision_log(model_decision_tracker, bird_class_score, rk_score, 1,
                                             data_loader_inst.model_decisions_logpath)
                    # clean up - delete text analyser inst
                    del text_analyser_inst
                    # go to next post
                    continue
                # likely no red kite present -> exclude post and copy image to 'excluded folder'
                else:
                    # add to model decision tracker - bird skipped and red kite not present
                    model_decision_tracker.append(0)
                    model_decision_tracker.append('-')
                    print( f'[-] ID {post_id} excluded - reason: common taxon name + no bird, score {bird_class_score}')
                    # copies excluded images to corresponding folder - when working with large corpora, this is disabled
                    ## shutil.copyfile(post_image_path, os.path.join(data_loader_inst.excluded_folderpath, f'{post_id}.jpg'))
                    with open(data_loader_inst.excluded_logpath, 'a', encoding='utf-8') as f:
                        f.write(f'{post_id}\n')
                    # write model decision string
                    write_model_decision_log(model_decision_tracker, bird_class_score, rk_score, 0,
                                             data_loader_inst.model_decisions_logpath)
                    # clean up - delete text analyser inst
                    del text_analyser_inst
                    # go to next post
                    continue
            # no en, ge, fr, nl, sp, it taxon name found
            else:
                # add to model decision tracker - common name not present
                model_decision_tracker.append(0)
                # pass image to bird model
                # modify image for inference including resizing to appropriate input layer format
                resized_image = cv2.resize(norm_loaded_img, bird_model_inst.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                input_tensor = tf.convert_to_tensor(np.expand_dims(resized_image, 0), dtype=tf.float32)
                detections, predictions_dict, shapes = bird_detection_fn(input_tensor)
                highest_bird_model_score_class_name, highest_bird_model_score, bird_class_score = bird_model_inst.extract_prediction_metric(
                    detections, label_id_offset=1)
                print(f'BIRD MODEL - image: {post_id}, class name: {highest_bird_model_score_class_name}, score: {highest_bird_model_score}')

                if bird_class_score > bird_model_inst.threshold:
                    # add to model decision tracker - bird present
                    model_decision_tracker.append(1)
                    # image likely to contain a bird -> pass to red kite image classification model for inference
                    rk_class_name, rk_score = redkite_model_inst.detection(post_image_path, post_id)  # norm_image

                    if rk_class_name == 'Red Kite':
                        # add to model decision tracker - red kite present
                        model_decision_tracker.append(1)
                        # likely to include a Red Kite -> include post and copy image to 'included folder'
                        print(f'[+] ID {post_id} included - reason: bird + red kite detected, rk_score: {rk_score}')
                        shutil.copyfile(post_image_path, os.path.join(data_loader_inst.included_folderpath, f'{post_id}.jpg'))
                        with open(data_loader_inst.included_logpath, 'a', encoding='utf-8') as f:
                            f.write(f'{post_id}\n')
                        # write model decision string
                        write_model_decision_log(model_decision_tracker, bird_class_score, rk_score, 1,
                                                 data_loader_inst.model_decisions_logpath)
                        # clean up - delete text analyser inst
                        del text_analyser_inst
                        # go to next post
                        continue
                    else:
                        # add to model decision tracker - no Red Kite found!
                        model_decision_tracker.append(0)
                        # likely no red kite present -> exclude post and copy image to 'excluded folder'
                        print(f'[-] ID {post_id} excluded - reason: bird detected + but no red kite, rk_score {rk_score}')
                        # copies excluded images to corresponding folder - when working with large corpora, this is disabled
                        ## shutil.copyfile(post_image_path, os.path.join(data_loader_inst.excluded_folderpath, f'{post_id}.jpg'))
                        with open(data_loader_inst.excluded_logpath, 'a', encoding='utf-8') as f:
                            f.write(f'{post_id}\n')
                        # write model decision string
                        write_model_decision_log(model_decision_tracker, bird_class_score, rk_score, 0,
                                                 data_loader_inst.model_decisions_logpath)
                        # clean up - delete text analyser inst
                        del text_analyser_inst
                        # go to next post
                        continue
                else:
                    # add to model decision tracker - bird not present and red kite model skipped
                    model_decision_tracker.append(0)
                    model_decision_tracker.append('-')
                    # likely no bird present -> exclude post and copy image to 'excluded folder'
                    print(f'[-] ID {post_id} excluded - reason: no taxon name + no bird, class: {highest_bird_model_score_class_name}, score: {highest_bird_model_score}')
                    # copies excluded images to corresponding folder - when working with large corpora, this is disabled
                    ## shutil.copyfile(post_image_path, os.path.join(data_loader_inst.excluded_folderpath, f'{post_id}.jpg'))
                    with open(data_loader_inst.excluded_logpath, 'a', encoding='utf-8') as f:
                        f.write(f'{post_id}\n')
                    # write model decision string
                    write_model_decision_log(model_decision_tracker, bird_class_score, rk_score, 0, data_loader_inst.model_decisions_logpath)
                    # clean up - delete text analyser inst
                    del text_analyser_inst
                    # go to next post
                    continue