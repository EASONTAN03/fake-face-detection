import yaml
import os 
import cv2
import numpy as np
import pandas as pd
import json
import random
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import operation_utils as utils
import preprocess_utils as preprocess

start_time = time.time()

current_dir = os.getcwd()
print(current_dir)
base_path = os.path.abspath(os.path.join(current_dir, '..'))
print("Base Path:", base_path)

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('../params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Set dataset and benchmark parameters
interim_data_path = os.path.join(base_path,config['dataset']['interim'])
processed_data_path = os.path.join(base_path,config['dataset']['processed'])
dataset = config['configs']['dataset']
benchmark = config['configs']['benchmark']

# Set random seed for reproducibility
seed = params['make_dataset']['seed']
np.random.seed(seed)
random.seed(seed)

# Prepare dataset parameters
param_prepare = params['prepare']
prepare_benchmark = param_prepare['benchmark']
output_type = param_prepare['output_type']
output_format=param_prepare['output_format']
color=param_prepare['color']
resize = tuple(param_prepare['resize'])
normalize = param_prepare['normalize']
preprocess_method = param_prepare['method']
extract_stats = param_prepare['extract_stats']
extract_mtcnn=param_prepare['extract_mtcnn']

# Define input and output directories
data_dir = f'{dataset}_{benchmark}'
input_dir = os.path.join(interim_data_path, data_dir)
output_dir = os.path.join(processed_data_path, data_dir)
datatype=["real","fake"]

mtcnn_features_path=os.path.join(output_dir, f'features_mtcnn.npy')
mtcnn_labels_path=os.path.join(output_dir, f'labels_mtcnn.npy')

input_type_dir=[]
output_type_dir=[]

if "train"in output_type:
    input_type_dir.append(os.path.join(input_dir, "train"))
    output_type_dir.append(os.path.join(output_dir, str(prepare_benchmark), "train"))
if "val"in output_type:
    input_type_dir.append(os.path.join(input_dir, "val"))
    output_type_dir.append(os.path.join(output_dir, str(prepare_benchmark), "val"))
if "test"in output_type:
    input_type_dir.append(os.path.join(input_dir, "test"))
    output_type_dir.append(os.path.join(output_dir, str(prepare_benchmark), "test"))


for index, dir in enumerate(output_type_dir):
    utils.create_dir(dir)
    if extract_mtcnn==True and os.path.exists(mtcnn_features_path): 
        resize_images=np.load(mtcnn_features_path)
        labels=np.load(mtcnn_labels_path)
    else :
        real_images, real_filenames=utils.read_images(os.path.join(input_type_dir[index],datatype[0]))
        fake_images, fake_filenames=utils.read_images(os.path.join(input_type_dir[index],datatype[1]))
        real_images=np.array(real_images)
        fake_images=np.array(fake_images)
        filenames=real_filenames+fake_filenames
        images = np.concatenate((real_images, fake_images))
        labels = np.array([0] * len(real_images) + [1] * len(fake_images))
        # Resize and normalize images
        ref_images, resize_images = [], []
        for img in images:
            ori, norm = preprocess.resize_and_normalize(img, resize, normalize)
            ref_images.append(ori)
            resize_images.append(norm)
        if extract_mtcnn==True:
            mtcnn_images=[preprocess.extract_with_mtcnn(img) for img in resize_images]
            resize_images=mtcnn_images
            np.save(mtcnn_features_path,np.array(resize_images))
            np.save(mtcnn_labels_path,np.array(labels))

    if color == 'YCbCr':
        resize_images = [cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) for img in resize_images]
    elif color == 'Gray':
        resize_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in resize_images]
    elif color == 'RGB':
        resize_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in resize_images]

    # ref_images = np.array(ref_images)
    # resize_images = np.array(resize_images)
    processed_images = np.array(resize_images)

    # Shuffle images and labels
    # indices = np.arange(resize_images.shape[0])
    # np.random.shuffle(indices)
    # ref_images = ref_images[indices]

    # processed_images = resize_images[indices]
    # labels = labels[indices]
    # filenames = np.array(filenames)[indices] 
    # filenames = filenames.tolist()

    compute_hist=param_prepare['compute_hist']

    # Apply texture extraction/edge detection techniques


    preprocess_method_str=[]
    if "clahe" in preprocess_method:
        clip_limit = param_prepare['clahe']['clip_limit']
        tile_grid_size = tuple(param_prepare['clahe']['tile_grid_size'])
        bins = param_prepare['clahe']['bins']
        processed_images = preprocess.apply_clahe(processed_images, clip_limit, tile_grid_size, bins, compute_hist)
        preprocess_method_str.append(f'{preprocess_method}, clip_limit:{clip_limit} ,tile_grid_size:{tile_grid_size}')
    if "lbp" in preprocess_method:
        radius = param_prepare['lbp']['radius']
        n_points = param_prepare['lbp']['n_points']
        method = param_prepare['lbp']['method']
        processed_images = preprocess.apply_lbp(processed_images, radius, n_points, method, compute_hist)
        preprocess_method_str.append(f'{preprocess_method}, radius:{radius}, n_points:{n_points}, method:{method}')
    if "sobel" in preprocess_method:
        kernel = param_prepare['sobel']['kernel']
        bins = param_prepare['sobel']['bins']
        processed_images = preprocess.apply_sobel(processed_images, kernel, bins, compute_hist)
        preprocess_method_str.append(f'{preprocess_method}, kernel:{kernel}')
    if "fft" in preprocess_method:
        bins = param_prepare['fft']['bins']
        processed_images = preprocess.apply_fft(resize_images, bins, compute_hist)
        preprocess_method_str.append(f'{preprocess_method}')
    if "ela" in preprocess_method:
        processed_images = preprocess.apply_ela(resize_images)
        preprocess_method_str.append(f'{preprocess_method}')       
    if "none" in preprocess_method:
        preprocess_method_str.append(f'{preprocess_method}')

    processed_images = np.array(processed_images, np.float32)

    if extract_stats==True:
        features = np.array([preprocess.extract_statistics(img) for img in processed_images])
    else:
        features=processed_images

    # Save preprocessed images and labels
    utils.create_dir(output_dir)
    if output_format == 'npy':
        np.save(os.path.join(output_dir, f'features_{prepare_benchmark}.npy'), features)
        np.save(os.path.join(output_dir, f'labels_{prepare_benchmark}.npy'), labels)

    elif output_format == 'jpg':
        output_image_dir = output_type_dir[index]
        utils.create_dir(output_image_dir)
        real_2save_dir=os.path.join(output_image_dir, "real")
        fake_2save_dir=os.path.join(output_image_dir, "fake")
        utils.create_dir(real_2save_dir)
        utils.create_dir(fake_2save_dir)

        for img, filename, label in zip(processed_images, filenames, labels):
            if img.ndim == 2:  # If the image is grayscale, convert it to 3-channel grayscale RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:  # If it's already in 3 channels (BGR), convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if label == 0:
                image_path = os.path.join(real_2save_dir, filename)  # Keep original filename
                cv2.imwrite(image_path, img)  # Save image as JPG
            elif label == 1:
                image_path = os.path.join(fake_2save_dir, filename)  # Keep original filename
                cv2.imwrite(image_path, img)  # Save image as JPG
            else: 
                print(f"Invalid label for image {filename}")

# Calculate the total runtime
end_time = time.time()
runtime = end_time - start_time
print(f"Total runtime: {runtime:.2f} seconds")

if output_format == 'npy':
    log_data = {
        "dataset": data_dir,
        "prepare_benchmark": prepare_benchmark,
        "seed": seed,
        "interpolation_resize": f"color:{color}, resize:{resize}, normalize:{normalize}, extract_mtcnn={extract_mtcnn}, extract_stats:{extract_stats}",
        "preprocess_method": preprocess_method_str,
        "features_shape": features.shape,
        "labels_shape": labels.shape,
        "output_dir": output_dir,
        "runtime_seconds": runtime  # Log the runtime
    }

else:
    log_data = {
        "dataset": data_dir,
        "prepare_benchmark": prepare_benchmark,
        "seed": seed,
        "interpolation_resize": f"color:{color}, resize:{resize}, normalize:{normalize}, extract_mtcnn={extract_mtcnn}, extract_stats:{extract_stats}",
        "preprocess_method": preprocess_method_str,
        "output_dir": output_dir,
        "runtime_seconds": runtime  # Log the runtime
    }


# Path to log file
log_file_path = os.path.join(output_dir, "log.json")
utils.write_json(log_data, log_file_path)
        
print(f"Data has been successfully preprocessed and saved to {output_dir}")


        