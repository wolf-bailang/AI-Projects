import os
import time
import torch
import random
import pprint
import numpy as np
from scipy.interpolate import interp1d
from terminaltables import AsciiTable
import deepsnap


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    deepsnap.set_seed(1)


def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def set_path(config):
    if config.mode == 'train':
        # config.EXP_NAME = 'experiments/{cfg.MODE}/easy_{cfg.R_EASY}_hard_{cfg.R_HARD}_m_{cfg.m}_M_{cfg.M}_freq_{cfg.TEST_FREQ}_seed_{cfg.SEED}'.format(cfg=config)
        ####################################################
        config.EXP_NAME = 'experiments/{cfg.MODE}/best_model'.format(cfg=config)
        ####################################################
        config.MODEL_PATH = os.path.join(config.EXP_NAME, 'model')
        config.LOG_PATH = os.path.join(config.EXP_NAME, 'log')
        makedirs(config.MODEL_PATH)
        makedirs(config.LOG_PATH)
    elif config.MODE == 'test':
        config.EXP_NAME = 'experiments/{cfg.MODE}'.format(cfg=config)
    config.OUTPUT_PATH = os.path.join(config.EXP_NAME, 'output')
    makedirs(config.OUTPUT_PATH)
    print('=> exprtiments folder: {}'.format(config.EXP_NAME))

    model_dir = "./{}/".format(config.model_dir) + config.dataset + "/split_" + config.split
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    results_dir = "./{}/".format(config.result_dir) + config.dataset + "/split_" + config.split
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


def save_config(config):
    file_path = os.path.join(config.OUTPUT_PATH, "config.txt")
    fo = open(file_path, "w")
    fo.write("Configurtaions:\n")
    fo.write(pprint.pformat(config))
    fo.close()


def EmebeddingNode2Vec(graph):
    graph_temp = Graph(graph)
    emb = np.load(cfg.DATA_PATH + cfg.dataset + "/emb/Node2Vec2_edges/" + video_name + '.npy')
    node_f = torch.zeros(len(graph_temp.node_feature), 2048 + 128)
    for i in range(len(graph_temp.node_feature)):
        # print(emb[i])
        # print(emb[i].shape)
        node_f[i] = torch.cat([graph_temp.node_feature[i], torch.Tensor(emb[i])], dim=0)
        # print(node_f[i])
        # print(node_f[i].shape)
    graph_temp.node_feature = node_f
    return graph_temp



def get_pred_activations(src, pred, config):
    src = minmax_norm(src)
    if len(src.size()) == 2:
        src = src.repeat((config.NUM_CLASSES, 1, 1)).permute(1, 2, 0)
    src_pred = src[0].cpu().numpy()[:, pred]
    src_pred = np.reshape(src_pred, (src.size(1), -1, 1))
    src_pred = upgrade_resolution(src_pred, config.UP_SCALE)
    return src_pred


def get_proposal_dict(cas_pred, aness_pred, pred, score_np, vid_num_seg, config):
    prop_dict = {}
    for th in config.CAS_THRESH:
        cas_tmp = cas_pred.copy()
        num_segments = cas_pred.shape[0] // config.UP_SCALE
        cas_tmp[cas_tmp[:, :, 0] < th] = 0
        seg_list = [np.where(cas_tmp[:, c, 0] > 0) for c in range(len(pred))]
        proposals = get_proposal_oic(seg_list, cas_tmp, score_np, pred, config.UP_SCALE, \
                                     vid_num_seg, config.FEATS_FPS, num_segments)
        for i in range(len(proposals)):
            class_id = proposals[i][0][0]
            prop_dict[class_id] = prop_dict.get(class_id, []) + proposals[i]

    for th in config.ANESS_THRESH:
        aness_tmp = aness_pred.copy()
        num_segments = aness_pred.shape[0] // config.UP_SCALE
        aness_tmp[aness_tmp[:, :, 0] < th] = 0
        seg_list = [np.where(aness_tmp[:, c, 0] > 0) for c in range(len(pred))]
        proposals = get_proposal_oic(seg_list, cas_pred, score_np, pred, config.UP_SCALE, \
                                     vid_num_seg, config.FEATS_FPS, num_segments)
        for i in range(len(proposals)):
            class_id = proposals[i][0][0]
            prop_dict[class_id] = prop_dict.get(class_id, []) + proposals[i]
    return prop_dict


def table_format(res_info, tIoU_thresh, title):
    table = [
        ['mAP@{:.1f}'.format(i) for i in tIoU_thresh],
        ['{:.4f}'.format(res_info['mAP@{:.1f}'.format(i)][-1]) for i in tIoU_thresh]
    ]
    table[0].append('mAP@AVG')
    table[1].append('{:.4f}'.format(res_info["average_mAP"][-1]))

    col_num = len(table[0])
    table = AsciiTable(table, title)
    for i in range(col_num):
        table.justify_columns[i] = 'center'

    return '\n' + table.table + '\n'


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def get_proposal_oic(tList, wtcam, final_score, c_pred, scale, v_len, sampling_frames, num_segments, _lambda=0.25,
                     gamma=0.2):
    t_factor = (16 * v_len) / (scale * num_segments * sampling_frames)
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                if len(grouped_temp_list[j]) < 2:
                    continue
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])
                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal))
                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(
                    range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])
                c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])
            temp.append(c_temp)
    return temp


def result2json(result, class_dict):
    result_file = []
    class_idx2name = dict((v, k) for k, v in class_dict.items())
    # print(class_idx2name)
    # {0: 'garbage', 1: 'reach_bag', 2: 'reach_bread', 3: 'carry_bread', 4: 'carry_bag', 5: 'reach_knife',
    # 6: 'carry_knife', 7: 'cut_bread', 8: 'reach_butter', 9: 'carry_butter', 10: 'liftopen_lidButter',
    # 11: 'scoop_butter', 12: 'smear_butter', 13: 'pressclose_lidButter', 14: 'move', 15: 'turn', 16: 'reach_topping',
    # 17: 'carry_topping', 18: 'liftopen_lidTopping', 19: 'reach_toppingpiece', 20: 'carry_toppingpiece',
    # 21: 'pressclose_lidTopping', 22: 'reach_fridge', 23: 'open_fridge', 24: 'close_fridge', 25: 'reach_cuttingboard',
    # 26: 'carry_cuttingboard', 27: 'reach_lidButter', 28: 'carry_lidButter', 29: 'reach_drawer', 30: 'pull_drawer',
    # 31: 'push_drawer', 32: 'walk', 33: 'reach_cabinet', 34: 'carry_plate', 35: 'shift', 36: 'reach_cloth',
    # 37: 'wipe_cloth', 38: 'reach_cereal', 39: 'carry_cereal', 40: 'reach_plate', 41: 'open_cabinet',
    # 42: 'close_cabinet', 43: 'reach_fork', 44: 'carry_fork', 45: 'open_cereal', 46: 'pour_cereal', 47: 'close_cereal',
    # 48: 'screwopen_capMilk', 49: 'pour_milk', 50: 'screwclose_capMilk', 51: 'carry_milk', 52: 'wait',
    # 53: 'reach_milk', 54: 'reach_bowl', 55: 'carry_bowl', 56: 'hold_bowl', 57: 'carry_spoon', 58: 'stir_cereal',
    # 59: 'reach_spoon', 60: 'carry_sugar', 61: 'screwopen_capSugar', 62: 'carry_capSugar', 63: 'scoop_sugar',
    # 64: 'reach_sugar', 65: 'screwclose_capSugar', 66: 'carry_capMilk', 67: 'reach_capMilk', 68: 'pour_sugar',
    # 69: 'reach_handle', 70: 'carry_handle', 71: 'reach_eggcarton', 72: 'carry_eggcarton', 73: 'liftopen_eggcarton',
    # 74: 'reach_egg', 75: 'carry_egg', 76: 'pressclose_eggcarton', 77: 'crack_egg', 78: 'fry_egg', 79: 'reach_salt',
    # 80: 'carry_salt', 81: 'pour_salt', 82: 'transfer_egg2plate', 83: 'reach_spatula', 84: 'carry_spatula',
    # 85: 'reach_oil', 86: 'screwopen_capOil', 87: 'pour_oil', 88: 'screwclose_capOil', 89: 'carry_pan', 90: 'hold_egg',
    # 91: 'transfer_egg2pan', 92: 'reach_pepper', 93: 'grind_pepper', 94: 'carry_pepper', 95: 'reach_cup',
    # 96: 'carry_cup', 97: 'pickup_egg', 98: 'carry_oil', 99: 'screwopen_capPepper', 100: 'twistonOff_faucet',
    # 101: 'wash', 102: 'turnonOff_stove', 103: 'reach_object', 104: 'reach_stove', 105: 'carry_cloth',
    # 106: 'screwopen_capSalt', 107: 'melt_butter', 108: 'transfer_butter2pan', 109: 'screwclose_capPepper',
    # 110: 'pour_pepper', 111: 'reach_pan', 112: 'screwclose_capSalt', 113: 'hold_handle', 114: 'hold_pan',
    # 115: 'carry_capSalt', 116: 'wipe', 117: 'reach_orange', 118: 'carry_orange', 119: 'cut_orange',
    # 120: 'reach_juicer', 121: 'carry_juicer', 122: 'squeeze_orange', 123: 'pour_juice', 124: 'hold_orange',
    # 125: 'hold_cup', 126: 'hold_spatula', 127: 'scramble_egg', 128: 'reach_faucet', 129: 'carry_whisk',
    # 130: 'reach_whisk', 131: 'carry_coffee', 132: 'pour_coffee', 133: 'reach_coffee', 134: 'hold_coffee',
    # 135: 'hold_milk', 136: 'stir_coffee', 137: 'peel_fruit', 138: 'cut_fruit', 139: 'reach_fruit', 140: 'carry_fruit',
    # 141: 'hold_fruit', 142: 'reach_peeler', 143: 'carry_peeler', 144: 'transfer_fruit2bowl', 145: 'stir_salad',
    # 146: 'open_teabag', 147: 'carry_teabag', 148: 'carry_kettle', 149: 'pour_water', 150: 'reach_teabox',
    # 151: 'openClose_teabox', 152: 'reach_teabag', 153: 'reach_kettle', 154: 'carry_teabox',
    # 155: 'carry_chocolatepowder', 156: 'reach_chocolatepowder', 157: 'liftopen_lidChoco', 158: 'carry_lidChoco',
    # 159: 'scoop_chocolatepowder', 160: 'stir_chocolate', 161: 'reach_lidChoco', 162: 'pressclose_lidChoco',
    # 163: 'carry_flour', 164: 'scoop_flour', 165: 'transfer_flour2bowl', 166: 'pressclose_lidFlour',
    # 167: 'stir_pancake', 168: 'cook_pancake', 169: 'reach_flour', 170: 'liftopen_lidFlour', 171: 'pour_flour',
    # 172: 'pour_pancake', 173: 'transfer_pancake2plate', 174: 'carry_lidFlour', 175: 'reach_lidFlour',
    # 176: 'flip_pancake', 177: 'reach_sink'}
    # print(len(result))
    # if len(result) > 1:
    #     print(result)
        # [[[55.0, -0.1256476491689682, 0.0, 523.0]], [[112.0, -8.811951637268066, 524.0, 530.0]],
        # [[55.0, -0.4463222026824951, 531.0, 753.0]], [[112.0, -8.701468467712402, 754.0, 764.0]],
        # [[128.0, -0.8072946071624756, 765.0, 921.0]], [[2.0, -45.903358459472656, 922.0, 923.0]],
        # [[156.0, -71.39659118652344, 924.0, 924.0]], [[21.0, -73.94384002685547, 925.0, 925.0]],
        # [[133.0, -76.49630737304688, 926.0, 926.0]], [[0.0, -79.05142211914062, 927.0, 927.0]],
        # [[156.0, -35.690513610839844, 928.0, 931.0]], [[0.0, -62.87029266357422, 932.0, 933.0]],
        # [[8.0, -96.84774780273438, 934.0, 934.0]], [[128.0, -99.38589477539062, 935.0, 935.0]],
        # [[8.0, -101.92340850830078, 936.0, 936.0]], [[92.0, -54.77861022949219, 937.0, 939.0]],
        # [[32.0, -58.59423065185547, 940.0, 942.0]], [[56.0, -119.73431396484375, 943.0, 943.0]],
        # [[79.0, -122.28414916992188, 944.0, 944.0]], [[64.0, -124.83666229248047, 945.0, 945.0]],
        # [[56.0, -127.38213348388672, 946.0, 946.0]], [[92.0, -129.92442321777344, 947.0, 947.0]],
        # [[6.0, -132.4676513671875, 948.0, 948.0]], [[8.0, -91.69850158691406, 949.0, 950.0]],
        # [[79.0, -140.08590698242188, 951.0, 951.0]], [[107.0, -73.85259246826172, 952.0, 954.0]],
        # [[56.0, -101.84831237792969, 955.0, 956.0]], [[163.0, -155.30062866210938, 957.0, 957.0]],
        # [[70.0, -106.90721130371094, 958.0, 959.0]], [[107.0, -110.28307342529297, 960.0, 961.0]],
        # [[56.0, -59.35236358642578, 962.0, 966.0]], [[0.0, -48.955963134765625, 967.0, 973.0]],
        # [[142.0, -101.70238494873047, 974.0, 976.0]], [[107.0, -205.93179321289062, 977.0, 977.0]],
        # [[14.0, -208.46116638183594, 978.0, 978.0]], [[134.0, -87.427734375, 979.0, 982.0]],
        # [[0.0, -149.09719848632812, 983.0, 984.0]], [[142.0, -49.28809356689453, 985.0, 993.0]],
        # [[159.0, -248.98548889160156, 994.0, 994.0]], [[107.0, -169.38278198242188, 995.0, 996.0]],
        # [[134.0, -256.62152099609375, 997.0, 997.0]], [[97.0, -259.1752624511719, 998.0, 998.0]],
        # [[49.0, -261.7256774902344, 999.0, 999.0]], [[157.0, -264.27532958984375, 1000.0, 1000.0]],
        # [[0.0, -109.7885971069336, 1001.0, 1004.0]], [[56.0, -277.01751708984375, 1005.0, 1005.0]],
        # [[49.0, -142.32981872558594, 1006.0, 1008.0]], [[169.0, -75.62026977539062, 1009.0, 1015.0]],
        # [[64.0, -205.03697204589844, 1016.0, 1017.0]], [[81.0, -208.4276123046875, 1018.0, 1019.0]],
        # [[107.0, -211.81053161621094, 1020.0, 1021.0]], [[56.0, -320.2512512207031, 1022.0, 1022.0]]]
    for i in range(len(result)):
        for j in range(len(result[i])):
            # print(result[i])
            # [[55.0, -0.1256476491689682, 0.0, 523.0]]
            # print(class_idx2name[result[i][j][0]])
            # carry_bowl
            # print(result[i][j][1])
            # -0.1256476491689682
            # print(result[i][j][2])
            # 0.0
            # print(result[i][j][3])
            # 523.0
            line = {'label': class_idx2name[result[i][j][0]], 'score': result[i][j][1],
                    'segment': [result[i][j][2], result[i][j][3]]}
            result_file.append(line)
            # print(line)
            # {'label': 'carry_bowl', 'score': -0.1256476491689682, 'segment': [0.0, 523.0]}
            # print(result_file)
            # [{'label': 'carry_bowl', 'score': -0.1256476491689682, 'segment': [0.0, 523.0]}]
    return result_file


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def save_best_record_thumos(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("Test_acc: {:.4f}\n".format(test_info["test_acc"][-1]))
    fo.write("average_mAP: {:.4f}\n".format(test_info["average_mAP"][-1]))

    tIoU_thresh = np.linspace(0.1, 0.7, 7)
    for i in range(len(tIoU_thresh)):
        fo.write("mAP@{:.1f}: {:.4f}\n".format(tIoU_thresh[i], test_info["mAP@{:.1f}".format(tIoU_thresh[i])][-1]))
    fo.close()


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=1)[0])
        min_val = relu(torch.min(act_map, dim=1)[0])
    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta
    ret[ret > 1] = 1
    ret[ret < 0] = 0
    return ret


def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]

    return keep


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count