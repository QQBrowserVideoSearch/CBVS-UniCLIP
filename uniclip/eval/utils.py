from collections import defaultdict
import numpy as np

def dcg(relevance, topk):
    """
    Calculate discounted cumulative gain.
    @param relevance: Graded and ordered relevances of the results.
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    rel = np.asarray(relevance)[:topk]
    return ((np.power(2, rel) - 1) / np.log2(np.arange(2, len(rel) + 2))).sum()


def ndcg(relevance, topk, eps=1e-6):
    """
    Calculate normalized discounted cumulative gain.
    @param relevance: Graded and ordered relevances of the results.
    """
    ideal_dcg = dcg(sorted(relevance, reverse=True), topk)
    return (dcg(relevance, topk) + eps) / (ideal_dcg + eps)


def ndcg_score(y_true, y_pred, topk=5):
    assert len(y_true) == len(y_pred)
    relevance = y_true[y_pred.argsort()[::-1]]
    return ndcg(relevance, topk)


def get_index(lst=None, item=''):
    tmp = []
    tag = 0
    for i in lst:
        if i == item:
            tmp.append(tag)
        tag += 1
    return tmp


def get_ndcg_res(file):
    scores, labels, qids = [], [], []
    for line in open(file):
        li = line.strip().split('\t')
        if len(li)<4:
            continue

        qid = li[1]
        score = float(li[-1])
        label  = int(li[0])

        scores.append(score)
        labels.append(label)
        qids.append(qid)

    topk=(1, 3, 5, 10)

    query_cnt = len(set(qids))
    ndcgs = []
    for qid in set(qids):
        temp_labels = np.array([labels[i] for i in get_index(qids, qid)]) # 将该qid对应的labels取出
        temp_scores = np.array([scores[i] for i in get_index(qids, qid)]) # 将该qid对应的scores取出

        if len(set(temp_labels)) > 1: # 只有1个label由于没有序关系，所以不存在NDCG的计算
            ndcgs.append([ndcg_score(temp_labels, temp_scores, k) for k in topk])
    ndcgs = np.array(ndcgs, dtype=np.float32).mean(0)
    for k, ndcg in zip(topk, ndcgs):
        print("NDCG@"+str(k)+'\t'+str(ndcg))

   
def pos_neg_order(fname, skip=0):

    maps = defaultdict(list)
    for line in open(fname):
        li = line.strip().split('\t')

        query = li[1]
        docid = li[2]
        label = float(li[0])
        score = float(li[3])
        
        maps[query].append((label, docid, score))

    total, positive, negative, special = 0, 0, 0, 0
    for key, values in maps.items():
        if len(values) == 1:
            continue

        total += 1
        values = sorted(values, key=lambda x: x[0], reverse=True)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[i][0] - values[j][0] <= skip:
                    continue
                if values[i][2] > values[j][2]:
                    positive += 1
                elif values[i][2] < values[j][2]:
                    negative += 1
                else:
                    special += 1
                
    print("    #  total examples: {}, positive: {}, negative: {}, special: {}, pos_neg_rate: {:.03f}: 1".format(
          total, positive, negative, special, 1.0 * (positive + special / 2.0) / max(1.0, negative + special / 2.0)))


def calculate_map(label, score):
    sorted_indices = sorted(range(len(score)), key=lambda i: score[i], reverse=True)
    sorted_labels = [label[i] for i in sorted_indices]
    num_relevant = 0
    sum_precision = 0
    for i, is_relevant in enumerate(sorted_labels):
        if is_relevant:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            sum_precision += precision

    return sum_precision / num_relevant if num_relevant > 0 else 0

def print_map(file):
    scores, labels, qids = [], [], []
    for line in open(file):
        li = line.strip().split('\t')
        if len(li)<4:
            continue

        query = li[1]
        label = int(li[0])
        score = float(li[-1])

        if label <= 1:
            label = 0
        else:
            label = 1

        scores.append(score)
        labels.append(label)
        qids.append(query)

    map_value_list = []
    for qid in set(qids):
        temp_labels = [labels[i] for i in get_index(qids, qid)] # 将该qid对应的labels取出
        temp_scores = [scores[i] for i in get_index(qids, qid)] # 将该qid对应的scores取出
        if sum(temp_labels)<1:
            continue
        map_value_list.append(calculate_map(temp_labels, temp_scores))
    print(np.mean(map_value_list))

def get_metrics(output_file):
    pos_neg_order(output_file, skip=0)
    get_ndcg_res(output_file)
    print_map(output_file)