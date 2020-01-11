# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from tqdm import tqdm
import csv, os

def evaluate_model(Y_probs, Y_test, config, general_metrics, bins=10, verbose=1, savefile=False):
    """
    Evaluates the model and calculates the calibration errors.
    
    Parameters:
        Y_probs: (numpy.ndarray) with predicted probabilities of test data
        Y_test: (numpy.ndarray) with test data labels
        config = {
                'DATA_NAME':DATA_NAME, 
                'ARCHI_NAME':ARCHI_NAME, 
                'LABEL_SMOOTH':LABEL_SMOOTH, 
                'METHOD_NAME':METHOD_NAME, 
                'CAL_NAME':CAL_NAME}
        general_metrics = {
                'train': training time s/epoch, 
                'cal': calibration time, 
                'test': inference time, 
                'size': number of parameters (1e6)}
        verbose: 0 -- quiet, 1 -- print out results, 2 -- print out results and plots
        savefile: if to save results into files
        
    Returns:
        
    """

    title = (f"{config['DATA_NAME']}_{config['ARCHI_NAME']}"
                f"_LS{config['LABEL_SMOOTH']}_{config['METHOD_NAME']}_{config['CAL_NAME']}")

    Y_preds = np.argmax(Y_probs, axis=1)

    # Confidence of prediction
    Y_confs = np.max(Y_probs, axis=1)  # Take only maximum confidence
    Y_true = np.argmax(Y_test, axis=1)
        
    acc = accuracy_score(Y_true, Y_preds)
    nll = log_loss(Y_true, Y_probs)
    
    # Probability of positive class
    Y_prob_true = np.array([Y_probs[i, idx] for i, idx in enumerate(Y_true)])
    brier = brier_score_loss(np.ones((len(Y_test),), dtype=int), Y_prob_true)
    
    # Calculate ECE & MCE
    ece, mce, u_accus, u_gaps, u_neg_gaps = uniform_binning(Y_confs, Y_preds, Y_true, bins, plot=True)
    
    # Calculate AECE & AMCE
    infer_results = [[Y_probs[i][Y_preds[i]], Y_true[i] == Y_preds[i]] for i in range(len(Y_test))]
    aece, amce, a_locations, a_accus, a_gaps, a_neg_gaps, a_widths = adaptive_binning(infer_results, plot=True)
    
    if savefile:
        # Write results into .csv file
        header = ['Title', 'Accuracy', 'ECE', 'MCE', 'AECE', 'AMCE', 'NLL', 'Brier score', 
                  't_train(s/epoch)', 't_cal', 't_test', 'n_parameters(1e6)']
        dir_res = './logs/results/'
        csv_name = 'preliminary_results.csv'
        if not os.path.exists(dir_res):
            os.makedirs(dir_res)
        if not os.path.exists(dir_res + csv_name):
            with open(dir_res + csv_name, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        with open(dir_res + csv_name, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([title, acc, ece, mce, aece, amce, nll, brier, 
                             general_metrics['train'], 
                             general_metrics['cal'],
                             general_metrics['test'],
                             general_metrics['size']])
            
        # write results for plotting into .csv file
        csv_name_plot = f'{title}.csv'
        with open(dir_res + csv_name_plot, 'a+', newline='') as f:
            writer = csv.writer(f)
            for line in [u_accus, u_gaps, u_neg_gaps, a_locations, a_accus, a_gaps, a_neg_gaps, a_widths]:
                writer.writerow(line)

        
    generate_plots(bins, u_accus, u_gaps, u_neg_gaps, a_locations, a_accus, a_gaps, a_neg_gaps, a_widths, 
                   savefile, title)
    
    if verbose > 0:
        print(f"Accuracy:\t{acc:.5f}")
        print(f"ECE:\t\t{ece:.5f}")
        print(f"MCE:\t\t{mce:.5f}")
        print(f"AECE:\t\t{aece:.5f}")
        print(f"AMCE:\t\t{amce:.5f}")
        print(f"NLL:\t\t{nll:.5f}")
        print(f"Brier score:\t{brier:.5f}")
    if verbose > 1:
        plt.show()

    return 


def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin

def uniform_binning(conf, pred, true, n_bins=10, plot=True):
    """
    Uniform binning to calculate ECE and McE.
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        n_bin: (int): number of bins
        
    Returns:
        ece: expected calibration error
        mce: maximum calibration error
    """
    
    bin_size = 1 / n_bins
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(true)
    confs = [0 for i in range(n_bins)]
    accus = [0 for i in range(n_bins)]
    gaps = [0 for i in range(n_bins)]
    neg_gaps = [0 for i in range(n_bins)]
    
    ece = 0
    mce = 0
    cal_errors = []
    for i in range(len(upper_bounds)):
        accus[i], confs[i], len_bin = compute_acc_bin(upper_bounds[i]-bin_size, upper_bounds[i], conf, pred, true)
        ece += np.abs(accus[i]-confs[i])*len_bin/n
        cal_errors.append(np.abs(accus[i]-confs[i]))
        if confs[i] - accus[i] > 0:
            gaps[i] = confs[i] - accus[i]
        else:
            neg_gaps[i] = confs[i] - accus[i]
        mce = max(cal_errors)
        
#    # Plot the Reliability Diagram if needed.
#    if plot:
#        x_location = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)
#        
#        plt.style.use('ggplot')
#        f1,ax = plt.subplots()
#        ax.set_aspect('equal')
#        plt.bar(x_location, accus, width=bin_size, edgecolor="#006392", color="#006392")
#        plt.bar(x_location, gaps, width=bin_size, bottom=accus, edgecolor="#c72e28", color="#c72e28", alpha=0.8)
#        plt.bar(x_location, neg_gaps, width=bin_size, bottom=accus, edgecolor="#C4961A", color="#be9c2d")
#        plt.title('Uniform Binning', fontsize=18)
#        plt.legend(['Accuracy','Positive gap','Negative gap'], fontsize=12, loc=2)
#        plt.xlim(0, 1)
#        plt.ylim(0, 1)
#        plt.xlabel('Confidence', fontsize=15, color = "black")
#        plt.ylabel('Accuracy', fontsize=15, color = "black")
#        plt.plot([0,1], [0,1], linestyle="--", color="#52854C")
#        plt.show()
    
    return ece, mce, accus, gaps, neg_gaps
      

def adaptive_binning(infer_results, plot = True):
    '''
    This function implement adaptive binning. It returns AECE, AMCE and some other useful values.

    Arguements:
    infer_results (list of list): a list where each element "res" is a two-element list denoting the infer result of a single sample. res[0] is the confidence score r and res[1] is the correctness score c. Since c is either 1 or 0, here res[1] is True if the prediction is correctd and False otherwise.
    plot (boolean): a boolean value to denote wheather to plot a Reliability Diagram.

    Return Values:
    AECE (float): expected calibration error based on adaptive binning.
    AMCE (float): maximum calibration error based on adaptive binning.
    cofidence (list): average confidence in each bin.
    accuracy (list): average accuracy in each bin.
    cof_min (list): minimum of confidence in each bin.
    cof_max (list): maximum of confidence in each bin.

    '''

    # Intialize.
    infer_results.sort(key = lambda x : x[0], reverse = True)
    n_total_sample = len(infer_results)

    assert infer_results[0][0] <= 1 and infer_results[1][0] >= 0, 'Confidence score should be in [0,1]'

    z=1.645
    num = [0 for i in range(n_total_sample)]
    final_num = [0 for i in range(n_total_sample)]
    correct = [0 for i in range(n_total_sample)]
    confidence = [0 for i in range(n_total_sample)]
    cof_min = [1 for i in range(n_total_sample)]
    cof_max = [0 for i in range(n_total_sample)]
    accuracy = [0 for i in range(n_total_sample)]
    
    ind = 0
    target_number_samples = float('inf')

    # Traverse all samples for a initial binning.
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        # Merge the last bin if too small.
        if num[ind] > target_number_samples:
            if (n_total_sample - i) > 40 and cof_min[ind] - infer_results[-1][0] > 0.05:
                ind += 1
                target_number_samples = float('inf')
        num[ind] += 1
        confidence[ind] += confidence_score

        assert correctness in [True,False], 'Expect boolean value for correctness!'
        if correctness == True:
            correct[ind] += 1

        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)
        # Get target number of samples in the bin.
        if cof_max[ind] == cof_min[ind]: 
            target_number_samples = float('inf')
        else:
            target_number_samples = (z / (cof_max[ind] - cof_min[ind])) ** 2 * 0.25

    n_bins = ind + 1

    # Get final binning.
    if target_number_samples - num[ind] > 0:
        needed = target_number_samples - num[ind]
        extract = [0 for i in range(n_bins - 1)]
        final_num[n_bins - 1] = num[n_bins - 1]
        for i in range(n_bins - 1):
            extract[i] = int(needed * num[ind] / n_total_sample)
            final_num[i] = num[i] - extract[i]
            final_num[n_bins - 1] += extract[i]
    else:
        final_num = num
    final_num = final_num[:n_bins]

    # Re-intialize.
    num = [0 for i in range(n_bins)]    
    correct = [0 for i in range(n_bins)]
    confidence = [0 for i in range(n_bins)]
    cof_min = [1 for i in range(n_bins)]
    cof_max = [0 for i in range(n_bins)]
    accuracy = [0 for i in range(n_bins)]
    gap = [0 for i in range(n_bins)]
    neg_gap = [0 for i in range(n_bins)]
    # Bar location and width.
    x_location = [0 for i in range(n_bins)]
    width = [0 for i in range(n_bins)]


    # Calculate confidence and accuracy in each bin.
    ind = 0
    for i, confindence_correctness in enumerate(infer_results):

        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        num[ind] += 1
        confidence[ind] += confidence_score

        if correctness == True:
            correct[ind] += 1
        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)

        if num[ind] == final_num[ind]:
            confidence[ind] = confidence[ind] / num[ind] if num[ind] > 0 else 0
            accuracy[ind] = correct[ind] / num[ind] if num[ind] > 0 else 0
            left = cof_min[ind]
            right = cof_max[ind]
            x_location[ind] = (left + right) / 2
            width[ind] = (right - left) * 0.9
            if confidence[ind] - accuracy[ind] > 0:
                gap[ind] = confidence[ind] - accuracy[ind]
            else:
                neg_gap[ind] = confidence[ind] - accuracy[ind]
            ind += 1

    # Get AECE and AMCE based on the binning.
    AMCE = 0
    AECE = 0
    for i in range(n_bins):
        AECE += abs((accuracy[i] - confidence[i])) * final_num[i] / n_total_sample
        AMCE = max(AMCE, abs((accuracy[i] - confidence[i])))

#    # Plot the Reliability Diagram if needed.
#    if plot:
#        plt.style.use('ggplot')
#        f1,ax = plt.subplots()
#        ax.set_aspect('equal')
#        plt.bar(x_location, accuracy, width, color="#006392")
#        plt.bar(x_location, gap, width, bottom=accuracy, color="#c72e28", alpha=0.8)
#        plt.bar(x_location, neg_gap, width, bottom=accuracy, color="#be9c2d")
#        plt.title('Adaptive Binning', fontsize=18)
#        plt.legend(['Accuracy','Positive gap','Negative gap'], fontsize=12, loc=2)
#        plt.xlim(0, 1)
#        plt.ylim(0, 1)
#        plt.xlabel('Confidence', fontsize=15, color = "black")
#        plt.ylabel('Accuracy', fontsize=15, color = "black")
#        plt.plot([0,1], [0,1], linestyle="--", color="#52854C")
#        plt.show()

    return AECE, AMCE, x_location, accuracy, gap, neg_gap, width

def generate_plots(u_bins, u_accus, u_gaps, u_neg_gaps, a_locations, a_accus, a_gaps, a_neg_gaps, a_widths, 
                   saveimg, title):
    
    u_bin_size = 1 / u_bins
    u_locations = np.arange(0+u_bin_size/2, 1+u_bin_size/2, u_bin_size)
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10.5, 5), sharex='col', sharey='row')
    fig.subplots_adjust(wspace=0.15)
    fig.suptitle(title, fontsize=19)
    names = [" (Uniform Binning)", " (Adaptive Binning)"]
    
    # Uniform plot
    u_accu_plot = ax[0].bar(u_locations, u_accus, width=u_bin_size, 
                    label='Accuracy', edgecolor="#006392", color="#006392")
    u_gap_plot = ax[0].bar(u_locations, u_gaps, width=u_bin_size, 
                    label='Positive gap', bottom=u_accus, edgecolor="#c72e28", color="#c72e28", alpha=0.8)
    u_ngap_plot = ax[0].bar(u_locations, u_neg_gaps, width=u_bin_size, 
                    label='Negative gap', bottom=u_accus, edgecolor="#C4961A", color="#be9c2d")
    ax[0].set_aspect('equal')
    ax[0].plot([0,1], [0,1], linestyle="--", color="#52854C")
    ax[0].legend(handles=[u_accu_plot, u_gap_plot, u_ngap_plot], fontsize=12, loc=2)
    ax[0].set_title('Uniform Binning', fontsize=15)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel('Confidence', fontsize=15, color = "black")
    ax[0].set_ylabel('Accuracy', fontsize=15, color = "black")
    
    # Adaptive plot
    a_accu_plot = ax[1].bar(a_locations, a_accus, width=a_widths, 
                    label='Accuracy', color="#006392")
    a_gap_plot = ax[1].bar(a_locations, a_gaps, width=a_widths, 
                    label='Positive gap', bottom=a_accus, color="#c72e28", alpha=0.8)
    a_ngap_plot = ax[1].bar(a_locations, a_neg_gaps, width=a_widths, 
                    label='Negative gap', bottom=a_accus, color="#be9c2d")
    ax[1].set_aspect('equal')
    ax[1].plot([0,1], [0,1], linestyle="--", color="#52854C")
    ax[1].legend(handles=[a_accu_plot, a_gap_plot, a_ngap_plot], fontsize=12, loc=2)
    ax[1].set_title('Adaptive Binning', fontsize=15)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('Confidence', fontsize=15, color = "black")
#    ax[1].set_ylabel('Accuracy', fontsize=15, color = "black")
    
    if saveimg:
        dir_imgs = './logs/imgs/'
        if not os.path.exists(dir_imgs):
            os.makedirs(dir_imgs)
        plt.savefig(dir_imgs + f'{title}.png', format='png', dpi=100, transparent=True, bbox_inches='tight')
        plt.savefig(dir_imgs + f'{title}.pdf', format='pdf', dpi=100, transparent=True, bbox_inches='tight')
    
    return
