import sys
import numpy
import torch
import math
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report


def train_one_epoch(args, model, data_loader_train, optimizer=None, epoch=None, logger=None):

    model.train()
    logger.info('Current param_groups[0].lr = {}'.format(optimizer.param_groups[0]['lr']))
    for step, batch_input in enumerate(data_loader_train):
        loss = model(batch_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not math.isfinite(loss.item()):
            logger.info("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        if step % args.print_fre == 0:
            if logger is None:
                print(f'epoch {epoch} \tstep {step}/{len(data_loader_train)} \tloss {loss.item():.6f}')
            else:
                logger.info(f'epoch {epoch} \tstep {step}/{len(data_loader_train)} \tloss {loss.item():.6f}')

    return model


def avg_auc(gts, preds, logger=None):

    auc_macro = roc_auc_score(gts, preds, multi_class='ovr', average='macro')
    auc_weighted = roc_auc_score(gts, preds, multi_class='ovr', average='weighted')

    if logger is None:
        print(f"Macro-averaged AUC (One-vs-Rest): {auc_macro:.4f}")
        print(f"Weighted-averaged AUC (One-vs-Rest): {auc_weighted:.4f}")
    else:
        logger.info(f"Macro-averaged AUC (One-vs-Rest): {auc_macro:.4f}")
        logger.info(f"Weighted-averaged AUC (One-vs-Rest): {auc_weighted:.4f}")

    auc_macro = roc_auc_score(gts, preds, multi_class='ovo', average='macro')
    auc_weighted = roc_auc_score(gts, preds, multi_class='ovo', average='weighted')

    if logger is None:
        print(f"Macro-averaged AUC (One-vs-One): {auc_macro:.4f}")
        print(f"Weighted-averaged AUC (One-vs-One): {auc_weighted:.4f}")
    else:
        logger.info(f"Macro-averaged AUC (One-vs-One): {auc_macro:.4f}")
        logger.info(f"Weighted-averaged AUC (One-vs-One): {auc_weighted:.4f}")


def mean_avg_precision(gts, preds, num_classes, logger=None):
    individual_ap_scores = []

    for i, class_label in enumerate(range(num_classes)):
        # y_true_one_hot[:, i] selects the column for the current class (binary true labels: 0 or 1)
        # y_score_probabilities[:, i] selects the predicted probabilities for the current class
        # Check if the class exists in the true labels to avoid division by zero or errors
        # if np.sum(y_true_one_hot[:, i]) > 0: # This check is usually not strictly necessary for average_precision_score
        ap_score = average_precision_score(gts[:, i], preds[:, i])
        individual_ap_scores.append(ap_score)
        if logger is None:
            print(f"  Class {class_label:2d} AP: {ap_score:.4f}")
        else:
            logger.info(f"  Class {class_label:2d} AP: {ap_score:.4f}")

    # --- 4. Calculate Mean Average Precision (mAP) ---
    if individual_ap_scores:
        mean_average_precision = numpy.mean(individual_ap_scores)
        if logger is None:
            print(f"Mean Average Precision (mAP): {mean_average_precision:.4f}")
        else:
            logger.info(f"Mean Average Precision (mAP): {mean_average_precision:.4f}")
    else:
        print("Could not calculate mAP as no individual AP scores were computed.")


def evaluation(model, data_loader, args, epoch=0, is_multi_drug=False, logger=None):

    model.eval()

    target_matrix_list = []
    batch_pred_list = []
    evaluation_result = []
    for batch_input in tqdm(data_loader, desc='inferencing ...'):

        batch_result = model(batch_input)

        if args.only_single_reaction:
            # pred_idx = torch.argmax(batch_result, dim=1)
            # batch_pred = torch.zeros_like(batch_result).to('cpu')
            # batch_pred[torch.arange(batch_result.shape[0]), pred_idx] = 1
            batch_pred = batch_result.to('cpu').detach()
        else:
            batch_pred = (batch_result > args.thr).float().to('cpu')
        # Construct the label matrix
        target_matrix = torch.zeros_like(batch_pred)
        for i, sample in enumerate(batch_input):
            target_matrix[i, sample['patient_reaction_pos_ids']] = 1

        target_matrix_list.append(target_matrix)
        batch_pred_list.append(batch_pred)

        # Generate model outputs
        for input, pred in zip(batch_input, batch_pred):
            input['pred'] = pred
            evaluation_result.append(input)

    import pickle as pkl
    import os
    file_name = f'evaluation_result_mtd{epoch}_.pkl' if is_multi_drug else f'evaluation_result_all{epoch}_.pkl'
    with open(os.path.join(args.output_dir, file_name), 'wb') as f:
        pkl.dump(evaluation_result, f)

    # Compute subset accuracy
    targets = torch.cat(target_matrix_list, dim=0)
    predictions = torch.cat(batch_pred_list, dim=0)
    # subset_accuracy = accuracy_score(targets, batch_pred)
    # macro_recall = recall_score(targets, predictions, average='macro', zero_division=0)
    # micro_recall = recall_score(targets, predictions, average='micro', zero_division=0)
    # samples_recall = recall_score(targets, predictions, average='samples', zero_division=0)

    avg_auc(gts=targets, preds=predictions, logger=logger)
    mean_avg_precision(gts=targets, preds=predictions, num_classes=targets.shape[1], logger=logger)

    pred_idx = torch.argmax(predictions, dim=1)
    predictions_ = torch.zeros_like(predictions).to('cpu')
    predictions_[torch.arange(predictions_.shape[0]), pred_idx] = 1
    report = classification_report(targets, predictions_, zero_division=0)
    logger.info(f'report = \n{report}')

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    targets_idx = torch.argmax(targets, dim=1).cpu().numpy()
    pred_idx = torch.argmax(predictions_, dim=1).cpu().numpy()
    cm = confusion_matrix(targets_idx, pred_idx)

    import pickle as pkl
    with open('./output/cm.pkl', 'wb') as f:
        pkl.dump(cm, f)
    #
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.figure(figsize=(8, 9))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # # plt.imshow(cm, cmap='hot', interpolation='nearest')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.tight_layout()
    # plt.show()
    #
    # print(cm)