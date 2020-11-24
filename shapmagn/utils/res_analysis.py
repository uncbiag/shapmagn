import os
import numpy as np
from video_cue.utils.file_util import read_csv_file
from video_cue.criterions.metrics import get_multi_metric
from sklearn.metrics import log_loss
from video_cue.utils.file_util import read_txt_into_list
import subprocess
import matplotlib.pyplot as plt


def compute_vid_prob(vid_prob,prob_fn=np.mean):
    vid_prob_mean = prob_fn(vid_prob)
    return vid_prob_mean



def get_log_loss(prob, gt):
    if isinstance(prob,list):
        prob = np.array(prob)
    if isinstance(gt,list):
        gt = np.array(gt)
    gt = gt.astype(prob.dtype)
    loss = log_loss(gt, prob, labels=[0, 1],eps =1e-7)
    return loss

def plot_roc(probs_list,name_list, gt, title="ROC"):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score,roc_curve
    result_table = pd.DataFrame(columns=['method', 'fpr', 'tpr', 'auc'])

    for probs, name in zip(probs_list, name_list):

        fpr, tpr, _ = roc_curve(gt, probs)
        auc = roc_auc_score(gt, probs)

        result_table = result_table.append({'method': name,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('method', inplace=True)

    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title(title, fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()

def get_metric(prob,gt):
    from sklearn.metrics import roc_auc_score,roc_curve
    metric = {}
    metric["acc"] =np.sum((prob>0.5) == gt) / len(gt)
    metric["precision"] =  np.sum(np.logical_and(prob>0.5, gt)) / np.sum((prob>0.5)==True)
    metric["auc"] =roc_auc_score(gt,prob)
    return metric




def get_gt_info(gt_path):
    csv_file_path = gt_path
    video_name_list, label_list = read_csv_file(csv_file_path)
    get_video_id_fn = lambda x: x[:-4]  # remove .mp4
    video_id_list = [get_video_id_fn(name) for name in video_name_list]
    gt_info = {name:1-label for name, label in zip(video_id_list,label_list)} # inverse the label, during training we take 0 as fake
    return gt_info




def get_vid_pred_info(pred_label_path):
    csv_file_path = pred_label_path
    video_name_list, label_list = read_csv_file(csv_file_path)
    get_video_id_fn = lambda x: x  if '.mp4' not in x else x[:-4]
    video_id_list = [get_video_id_fn(name) for name in video_name_list]
    pred_info = {}
    for name, pred_label in zip(video_id_list, label_list):
        if name in pred_info:
            pred_info[name] = pred_info[name] and pred_label # 0 refers to fake, here we use and operation for multi-subject video
        else:
            pred_info[name] = pred_label
    return pred_info, video_id_list


def get_frame_pred_info(pred_probs_path):
    video_probs_list = np.load(pred_probs_path,allow_pickle=True).tolist()
    return video_probs_list





def get_pred_prob_gt_label_list(folder_path, gt_info, phase):
    pred_path = os.path.join(folder_path,"predictions_{}_-1.csv".format(phase))
    prob_path = os.path.join(folder_path,"pro_{}_-1.npy".format(phase))
    pred_info, video_id_list = get_vid_pred_info(pred_path)
    video_probs_list = get_frame_pred_info(prob_path)
    #prob_info = {}
    probs_info = {}
    probs_cat_info= {}
    for vid_id, probs in zip(video_id_list, video_probs_list):
        if vid_id in probs_info:
            # 0 refers to fake, here we take the min prob for fake prediction and max prob for real prediction
            #prob_info[vid_id] = min(prob, prob_info[vid_id]) if pred_info[vid_id]==0 else max(prob, prob_info[vid_id])
            probs_info[vid_id] = np.concatenate((probs_info[vid_id], probs),0)
            probs_cat_info[vid_id] = probs_cat_info[vid_id]+[len(probs)]
        else:
            #prob_info[vid_id] = prob
            probs_info[vid_id] = probs
            probs_cat_info[vid_id] = [len(probs)]
    video_id_list = list(set(video_id_list))
    vid_prob_list = [probs_info[vid_id].mean() for vid_id in video_id_list]
    vid_probs_list = [probs_info[vid_id].squeeze() for vid_id in video_id_list]
    vid_label_list = [gt_info[vid_id] for vid_id in video_id_list]
    vid_cat_info_list = [probs_cat_info[vid_id] for vid_id in video_id_list]
    return np.array(vid_prob_list),vid_probs_list, np.array(vid_label_list),video_id_list, vid_cat_info_list


def match_two_sequence(vid_probs1, vid_probs2, cat_info1, cat_info2):
    assert len(cat_info1) == len(cat_info2)
    split_id1 = [0] + list(np.cumsum(cat_info1))
    split_id2 = [0] + list(np.cumsum(cat_info2))
    shared_len_list = [min(len1,len2) for len1, len2 in zip(cat_info1,cat_info2)]
    vid_probs_shared1 = [vid_probs1[split_id1[sid]:split_id1[sid]+shared_len_list[sid]] for sid in range(len(shared_len_list))]
    vid_probs_shared2 = [vid_probs2[split_id2[sid]:split_id2[sid]+shared_len_list[sid]] for sid in range(len(shared_len_list))]
    vid_probs_shared1 = np.concatenate(vid_probs_shared1)
    vid_probs_shared2 = np.concatenate(vid_probs_shared2)
    return vid_probs_shared1, vid_probs_shared2


def get_vid_probs_attr(vid_probs, refer_probs=None, attr="acc"):
    if attr == "acc":
        attr = np.sum(vid_probs>0.5)/len(vid_probs)
    if attr == "mean":
        attr = np.mean(vid_probs)
    if attr == "refer_mean":
        attr = np.mean(refer_probs)
    if attr == "cross_mean":
        alpha = 0.85
        attr = np.mean(vid_probs)*alpha + np.mean(refer_probs)*(1-alpha)
    if attr == "cross_union":
        attr = (np.sum(vid_probs[vid_probs>0.5]) + np.sum(refer_probs[refer_probs>0.5]))/len(vid_probs)/2
    if attr == "cross_inter":
        ind = np.where(np.logical_and(vid_probs>0.5, refer_probs>0.5))
        attr = (np.sum(vid_probs[ind]) + np.sum(refer_probs[ind]))/len(vid_probs)/2
    if attr == "one_domain":
        probs = vid_probs.copy()
        probs[np.where(refer_probs>0.95)] = refer_probs[np.where(refer_probs>0.95)]
        probs = np.maximum(probs,vid_probs)
        attr = np.mean(probs)
    if attr == "cross_multi":
        probs = vid_probs * refer_probs
        attr = np.mean(probs)
    if attr == "cross_inter_multi":
        ind = np.where(np.logical_and(vid_probs>0.5, refer_probs>0.5))
        attr = np.sum(vid_probs[ind] *refer_probs[ind])/len(vid_probs)
    if attr == "cross_max_union":
        raise ValueError



    return attr

def get_fea_for_plot(vid_probs_list1, vid_probs_list2, weight_mode="cross_mean"):
    x = [get_vid_probs_attr(vid_probs,attr="acc") for vid_probs in vid_probs_list1]
    y = [get_vid_probs_attr(vid_probs,attr="acc") for vid_probs in vid_probs_list2]
    z = [get_vid_probs_attr(vid_probs1,vid_probs2,attr=weight_mode)
     for vid_probs1, vid_probs2 in zip(vid_probs_list1,vid_probs_list2)]

    return np.array(x), np.array(y), np.array(z)


def plot_fea(vertice, fake_index, real_index, z_label="value"):
    from mpl_toolkits import mplot3d

    ax = plt.axes(projection='3d')
    xdata, ydata, zdata = vertice
    ax.scatter3D(xdata[fake_index], ydata[fake_index], zdata[fake_index], cmap='Reds')
    ax.scatter3D(xdata[real_index], ydata[real_index], zdata[real_index], cmap='Blues')
    ax.set_xlabel('percentage of texture fake per video', fontsize=10)
    ax.set_ylabel('percentage of temporal fake per video', fontsize=10)
    ax.set_zlabel("score", fontsize=10)
    plt.show()
    plt.clf()
    print("")


def plot_distribution_for_frame_and_temporal(frame_vid_probs_list, temporal_vid_probs_list,gt_info,vid_id_list,weight_mode="cross_mean"):
    vertice = get_fea_for_plot(frame_vid_probs_list,temporal_vid_probs_list,weight_mode)
    gt_list = [gt_info[vid_id] for vid_id in vid_id_list]
    gt_np = np.array(gt_list)
    real_index = np.where(gt_np==1)
    fake_index = np.where(gt_np==0)
    plot_fea(vertice,fake_index,real_index, weight_mode)








def get_common_part(gt_info, pred_info, print_metric=True):
    pred_keys = pred_info.keys()
    gt_keys= gt_info.keys()
    assert set(pred_keys).issubset(set(gt_keys))
    pred_label_list= []
    gt_shared_label_list = []
    for name in pred_keys:
        pred_label_list.append(pred_info[name])
        gt_shared_label_list.append(gt_info[name])

    pred_label_np = np.array(pred_label_list)
    gt_shared_label_np = np.array(gt_shared_label_list)
    if print_metric:
        metric = get_multi_metric(pred_label_np[None],gt_shared_label_np[None])
        #metric = np.sum(pred_label_np==gt_shared_label_np)/len(pred_label_np)
        print(metric)
    return pred_label_np, gt_shared_label_np


def explore_methods_acc(method_dict, gt):
    binary_overlap = np.zeros_like(gt)
    for method in method_dict:
        binary_metric=method_dict[method]==gt
        score = np.sum(binary_metric)/len(gt)
        binary_overlap += binary_metric
        print("the method {} score is {}".format(method, score))
    score = np.sum(binary_overlap>0)/len(gt)
    print("the best combined score is {}".format(score))


def save_video_for_analyze(output_folder,selected_condition,vid_label_info):
    os.makedirs(output_folder, exist_ok=True)
    vid_id_dict = {}
    for i in range(len(vid_label_info)):
        name = vid_label_info[i][0]
        if name in vid_id_dict:
            vid_id_dict[name].append(i)
        else:
            vid_id_dict[name] = [i]

    label_list = []
    path_list = []
    for vid_name in np.array(video_id_list)[selected_condition]:
        for i in vid_id_dict[vid_name]:
            label_list.append(vid_label_info[i][1])
            path_list.append(vid_label_info[i][2])

    cmd = ""
    for label, vid_path in zip(label_list,path_list):
        output_path = os.path.join(output_folder, label)
        os.makedirs(output_path,exist_ok=True)
        cmd= "cp {} {} \n".format(vid_path, os.path.join(output_path,os.path.split(vid_path)[-1]))
        p = subprocess.Popen(cmd, shell=True)
        p.wait()


phase = "val"
use_local_machine =True
server_name = "/playpen-raid1" +"/" if not use_local_machine else "/home/zyshen/remote/llr11_mount"
analyze_path = server_name +"/zyshen/debug/dfdc_res_analyze/frame_temporal_368_new"
txt_file = server_name +"/zyshen/data/dfdc/resol_368_new/{}/file_path_list.txt".format(phase)
gt_path = server_name +"/Data/DFDC/dfdc/{}/labels.csv".format(phase)
frame_pred_folder = server_name +"/zyshen/data/dfdc/resol_368/efficienet_net_fea_mean_eval_230/records"
frame_pred_folder = server_name +"/zyshen/data/dfdc/resol_368_new/efficienet_net_fea_eval_250/records"
#frame_pred_folder = server_name +"/zyshen/data/dfdc/resol_368_train_part_1/efficienet_net_fea_eval60/records"
#frame_pred_folder = server_name +"/zyshen/data/dfdc/resol_368_train_part_2/train_weight_net_texture_part1_temporal_all_01modelweight_eval_110/records"
temporal_pred_folder= server_name +"/zyshen/data/dfdc/resol_368/temporal_net_fea_max_tuneall_eval_140/records"
temporal_pred_folder= server_name +"/zyshen/data/dfdc/resol_368_new/eval_temporal_net_pure_20/records"
gt_info = get_gt_info(gt_path)
frame_vid_prob,frame_vid_probs_list_raw, vid_label,video_id_list, frame_cat_list = get_pred_prob_gt_label_list(frame_pred_folder, gt_info, phase)
temporal_vid_prob,temporal_vid_probs_list_raw, _, _, temporal_cat_list = get_pred_prob_gt_label_list(temporal_pred_folder, gt_info, phase)

frame_vid_probs_list, temporal_vid_probs_list =[], []
for vid_probs1, vid_probs2,cat_info1, cat_info2 in zip(frame_vid_probs_list_raw,temporal_vid_probs_list_raw, frame_cat_list,temporal_cat_list):
    probs1, probs2 = match_two_sequence(vid_probs1, vid_probs2, cat_info1, cat_info2)
    frame_vid_probs_list.append(probs1)
    temporal_vid_probs_list.append(probs2)
weighted_prob =np.array([get_vid_probs_attr(frame_probs, temporal_probs,attr="cross_mean") for frame_probs, temporal_probs in zip(frame_vid_probs_list,temporal_vid_probs_list)])
frame_loss = get_log_loss(frame_vid_prob, vid_label)
temporal_loss = get_log_loss(temporal_vid_prob, vid_label)
weighted_loss = get_log_loss(weighted_prob,vid_label)
frame_acc = get_metric(frame_vid_prob,vid_label)
temporal_metric = get_metric(temporal_vid_prob,vid_label)
weighted_metric = get_metric(weighted_prob,vid_label)

print("the frame loss is {}, the temporal loss is {}, the weighted loss is {}".format(frame_loss, temporal_loss, weighted_loss))
print("the frame metric is {}, the temporal metric is {}, the weighted metric is {}".format(frame_acc, temporal_metric, weighted_metric))
plot_distribution_for_frame_and_temporal(frame_vid_probs_list,temporal_vid_probs_list,gt_info,video_id_list,weight_mode="cross_mean")
overlap = ((frame_vid_prob>0.5)==vid_label)+((temporal_vid_prob>0.5)==vid_label)
print("potential accuracy can get is {}".format(np.sum(overlap>0)/len(vid_label)))
vid_label_info = read_txt_into_list(txt_file)
output_folder = os.path.join(analyze_path,phase,"tp_frame_and_fp_temporal")
#selected_condition = np.where((frame_vid_prob<0.5)==vid_label)
selected_condition = np.where(np.logical_and((frame_vid_prob>0.5)==vid_label,(temporal_vid_prob<0.5)==vid_label))
#save_video_for_analyze(output_folder,selected_condition,vid_label_info)


dfdc_pred_path= server_name +"/zyshen/debug/{}_nostrategy_res.cvs".format(phase)
#dfdc_pred_path= server_name +"/zyshen/debug/test_nostrategy_val_res_111.cvs"
dfdc_info, dfdc_name_list = get_vid_pred_info(dfdc_pred_path)
dfdc_info = {vid_id:dfdc_info[vid_id] for vid_id in video_id_list}
dfdc_pred_prob_np, _ = get_common_part(gt_info,dfdc_info, print_metric=False)
dfdc_pred_prob_np = 1-dfdc_pred_prob_np # inverse the prediction, we take 0 as fake
dfdc_weighted_prob = ((dfdc_pred_prob_np)*0.9+temporal_vid_prob*0.1)
dfdc_metric = get_metric(dfdc_pred_prob_np,vid_label)
dfdc_loss = get_log_loss(dfdc_pred_prob_np,vid_label)
dfdc_metric_weighted = get_metric(dfdc_weighted_prob,vid_label)
dfdc_loss_weighted = get_log_loss(dfdc_weighted_prob,vid_label)
print("dfdc metric is {}, loss is {}".format(dfdc_metric,dfdc_loss))
print("dfdc metric weighted is {}, loss is {}".format(dfdc_metric_weighted,dfdc_loss_weighted))
output_folder = os.path.join(analyze_path,"fp_dfdc_tp_temporal")
#selected_condition = np.where(((dfdc_pred_prob_np)<0.5)==vid_label)
selected_condition = np.where(np.logical_and(((dfdc_pred_prob_np)<0.5)==vid_label,(temporal_vid_prob>0.5)==vid_label))
#save_video_for_analyze(output_folder,selected_condition,vid_label_info)


plot_roc([frame_vid_prob,temporal_vid_prob,weighted_prob,dfdc_pred_prob_np],['frame',"temporal","linear_weighted","winning"], vid_label, title="ROC {}".format(phase))


overlap = ((dfdc_pred_prob_np>0.5)==vid_label)+((frame_vid_prob>0.5)==vid_label) +((temporal_vid_prob>0.5)==vid_label)
print("potential accuracy can get is {}".format(np.sum(overlap>0)/len(vid_label)))