# -*- coding: utf-8 -*-

# Suspend FutureWarnings from np and tf
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import numpy as np
    # from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('Agg')


    import tensorflow.compat.v1 as tf
    tfk = tf.keras
    
    import time, os, argparse
    from tqdm import tqdm
    
    # Import local functions
    import evaluation, utils, models_le, models_res#, optimizers
    import cal_methods

# %% Command line arguements

parser = argparse.ArgumentParser(description='Framework for training and evaluation.')
parser.add_argument(
        '--dataset', '-d', 
        help="choose a dataset", 
        choices=['MNIST', 'OC'], 
        default='MNIST')
parser.add_argument(
        '--architecture', '-a', 
        help="choose a network architecture", 
        choices=['LeNet', 'ResNet'], 
        default='LeNet')
parser.add_argument(
        '--labelsmooth', '-s', 
        help="pre-processing: label smoothing factor between 0-1", 
        type=float, 
        default=0.0)
parser.add_argument(
        '--method', '-m', 
        help="choose a method", 
        choices=['Base', 'Drop', 'CDrop', 'LLCDrop', 'VI', 'LLVI'], 
        default='Base')
parser.add_argument(
        '--calibration', '-c', 
        help="post-processing: calibration method", 
        choices=['NC', 'TS'], 
        default='NC')
parser.add_argument(
        '--verbose', '-v', 
        help="0 -- quiet, 1 -- print out results, 2 -- print out results and plots", 
        type=int, 
        choices=[0, 1, 2], 
        default=1)
parser.add_argument(
        '--mode', 
        help="train or test", 
        choices=['train', 'test'], 
        default='train')
args = parser.parse_args()

# %% Parameters and Dataset
DATA_NAME = args.dataset
# in ['MNIST', 'OC']
ARCHI_NAME = args.architecture
# in ['LeNet', 'ResNet']
LABEL_SMOOTH = args.labelsmooth
# between 0-1, 0.1 shoube be enough
METHOD_NAME = args.method
# in ['Base', 'Drop', 'CDrop', 'LLCDrop', 'VI', 'LLVI']
CAL_NAME = args.calibration
# in ['NC', 'TS']

#print(f"DATA_NAME: {DATA_NAME}, {type(DATA_NAME)}")
#print(f"ARCHI_NAME: {ARCHI_NAME}, {type(ARCHI_NAME)}")
#print(f"LABEL_SMOOTH: {LABEL_SMOOTH}, {type(LABEL_SMOOTH)}")
#print(f"METHOD_NAME: {METHOD_NAME}, {type(METHOD_NAME)}")
#print(f"CAL_NAME: {CAL_NAME}, {type(CAL_NAME)}")
#print(f"verbose: {args.verbose}, {type(args.verbose)}")

BATCH_SIZE = 128
MC_SAMPLES = 5 # number of samples for prediciton

dict_config = {
        'DATA_NAME':DATA_NAME, 
        'ARCHI_NAME':ARCHI_NAME, 
        'LABEL_SMOOTH':LABEL_SMOOTH, 
        'METHOD_NAME':METHOD_NAME, 
        'CAL_NAME':CAL_NAME}
_postprocess = {'NC':cal_methods.NoCalibration, 'TS':cal_methods.TemperatureScaling}
try:
    CAL_METHOD = _postprocess[CAL_NAME]
except KeyError:
    raise ValueError("CAL_NAME should be in ['NC', 'TS'].")

if DATA_NAME == 'MNIST':
    SAMPLE_SHAPE = (28, 28, 1)
    N_CLASSES = 10
    EPOCHS = 40
    VAL_SPLIT = 0.2
    #  Load dataset
    train_generator, class_weights, X_val, Y_val, test_generator = utils.load_mnist(
            train_percentage=0.025,
            val_percentage=VAL_SPLIT,
            batch_size=BATCH_SIZE,
            label_smoothing=LABEL_SMOOTH)
elif DATA_NAME == 'OC':
    SAMPLE_SHAPE = (80, 80, 1)
    N_CLASSES = 2
    EPOCHS = 50
    DATA_DIR='./OralCancer_DataSet3/'
    VAL_SPLIT = 0.1
    #  Load dataset
    train_generator, class_weights, X_val, Y_val, test_generator = utils.load_OC(
            data_root=DATA_DIR, 
            sample_shape=SAMPLE_SHAPE, 
            val_percentage=VAL_SPLIT, 
            batch_size=BATCH_SIZE, 
            label_smoothing=LABEL_SMOOTH)
else:
    raise ValueError("DATA_NAME should be in ['MNIST', 'OC'].")

DIR_LOG = "./logs/weights/"
if not os.path.exists(DIR_LOG):
    os.makedirs(DIR_LOG)
WEIGHT_PATH = DIR_LOG + f'{DATA_NAME}_{ARCHI_NAME}_LS{LABEL_SMOOTH}_{METHOD_NAME}_{CAL_NAME}.ckpt'

# %% Create the model

if ARCHI_NAME == 'LeNet':
    if METHOD_NAME == 'Base':
        MC_SAMPLES = 0
        model = models_le.LeNet_base(input_shape=SAMPLE_SHAPE, classes=N_CLASSES)
    elif METHOD_NAME == 'Drop':
        model = models_le.LeNet_dropout(input_shape=SAMPLE_SHAPE, classes=N_CLASSES)
    elif METHOD_NAME == 'CDrop':
        model = models_le.LeNet_concreteDropout(input_shape=SAMPLE_SHAPE, classes=N_CLASSES, N=len(train_generator.x))
    elif METHOD_NAME == 'LLCDrop':
        model = models_le.LeNet_llconcreteDropout(input_shape=SAMPLE_SHAPE, classes=N_CLASSES, N=len(train_generator.x))
    elif METHOD_NAME == 'VI':
        model = models_le.LeNet_vi_flipout(input_shape=SAMPLE_SHAPE, classes=N_CLASSES, num_updates=len(train_generator.x))
    elif METHOD_NAME == 'LLVI':
        model = models_le.LeNet_llvi(input_shape=SAMPLE_SHAPE, classes=N_CLASSES)
    else:
        raise ValueError("METHOD_NAME should be in ['Base', 'Drop', 'CDrop', 'LLCDrop', 'VI', 'LLVI'].")
elif ARCHI_NAME == 'ResNet':
    if METHOD_NAME == 'Base':
        MC_SAMPLES = 0
        model = models_res.ResNet50_base(input_shape=SAMPLE_SHAPE, classes=N_CLASSES)
    elif METHOD_NAME == 'Drop':
        model = models_res.ResNet50_dropout(input_shape=SAMPLE_SHAPE, classes=N_CLASSES)
    elif METHOD_NAME == 'CDrop':
        model = models_res.ResNet50_concreteDropout(input_shape=SAMPLE_SHAPE, classes=N_CLASSES, N=len(train_generator.x))
    elif METHOD_NAME == 'LLCDrop':
        model = models_res.ResNet50_llconcreteDropout(input_shape=SAMPLE_SHAPE, classes=N_CLASSES, N=len(train_generator.x))
    elif METHOD_NAME == 'VI':
        model = models_res.ResNet50_vi_flipout(input_shape=SAMPLE_SHAPE, classes=N_CLASSES, num_updates=len(train_generator.x))
    elif METHOD_NAME == 'LLVI':
        model = models_res.ResNet50_llvi(input_shape=SAMPLE_SHAPE, classes=N_CLASSES, num_updates=len(train_generator.x))
    else:
        raise ValueError("METHOD_NAME should be in ['Base', 'Drop', 'CDrop', 'LLCDrop', 'VI', 'LLVI'].")
else:
    raise ValueError("ARCHI_NAME should be in ['LeNet', 'ResNet'].")
    
# %% Compile the model
if METHOD_NAME in ['VI', 'LLVI']:
    model.compile(optimizer='adam', loss=utils.viloss, metrics=['accuracy'])
else:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %% Callbacks
mc = utils.ModelCheckpoint(WEIGHT_PATH, monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1)
es = utils.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
rp = utils.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
ts = utils.TimeHistory()

# %% Train
if args.mode == 'train':
    history = model.fit_generator(
            generator = train_generator,
            epochs = EPOCHS,
            steps_per_epoch=train_generator.x.shape[0] // BATCH_SIZE + 1,
            verbose=1,
            class_weight=class_weights,
            validation_data=(X_val, Y_val),
            callbacks=[mc, es, rp, ts])

# %% Inference
try:
    model.load_weights(WEIGHT_PATH)
except:
    raise RuntimeError(f"Cannot find trained weights {WEIGHT_PATH}.")


if METHOD_NAME == 'Base':
    # Get logits and inference time
    Y_logits_val, Y_logits, t_test = utils.predict_logits(model, X_val, test_generator)
    # Calibration
    Y_probs, t_cal = utils.calibrate(CAL_METHOD, Y_logits_val, Y_val, Y_logits)
else:
    _t_test = time.time()
    Y_logits_val_arr = []
    Y_logits_arr = []
    ts_test = []
    for i in tqdm(range(MC_SAMPLES)):
        Y_logits_val, Y_logits, _ = utils.predict_logits(model, X_val, test_generator)
        Y_logits_val_arr.append(Y_logits_val)
        Y_logits_arr.append(Y_logits)
#        ts_test.append(t_test)
    Y_logits_val_arr = np.asarray(Y_logits_val_arr)
    Y_logits_val = np.mean(Y_logits_val_arr, axis=0)
    Y_logits_arr = np.asarray(Y_logits_arr)
    Y_logits = np.mean(Y_logits_arr, axis=0)
    Y_probs, t_cal = utils.calibrate(CAL_METHOD, Y_logits_val, Y_val, Y_logits)
    t_test = (time.time() - _t_test - t_cal) / MC_SAMPLES
    
# =============================================================================
#     for i in tqdm(range(MC_SAMPLES)):
#         Y_logits_val, Y_logits, t_test = utils.predict_logits(model, X_val, test_generator)
#         Y_probs, t_cal = utils.calibrate(CAL_METHOD, Y_logits_val, Y_val, Y_logits)
#         Y_probs_arr.append(Y_probs)
#         ts_cal.append(t_cal)
#         ts_test.append(t_test)
#     Y_probs_arr = np.asarray(Y_probs_arr)
#     Y_probs = np.mean(Y_probs_arr, axis=0)
#     t_test = sum(ts_test) / len(ts_test)
#     t_cal = sum(ts_cal) / len(ts_cal)
# 
# =============================================================================

if args.mode == 'train':
    t_train = utils.compute_train_time(ts)
else:
    t_train = 0.0

# training time per epoch
if args.verbose > 0:
    print('\n\n')
    print(f'Training time: \t{t_train:.4f} s/epoch')
    print(f'Calibration time: {t_cal:.4f} s for {len(Y_val)} validation samples')
    print(f'Inference time: {t_test:.4f} s for {len(test_generator.y)} samples')
    print(f'Number of parameters: {model.count_params()/1e6:.3f} e6')

dict_metrics = {
        'train':t_train, 
        'cal':t_cal, 
        'test':t_test, 
        'size':model.count_params()/1e6}
# %% Evaluation
#import evaluation
evaluation.evaluate_model(Y_probs, test_generator.y, 
                          config=dict_config, general_metrics=dict_metrics,
                          verbose=args.verbose, savefile=True)

#%% Plot learning curve
if args.mode == 'train':
    utils.accuracy_curve(history, dict_config, saveimg=True, verbose=args.verbose)
# %%
