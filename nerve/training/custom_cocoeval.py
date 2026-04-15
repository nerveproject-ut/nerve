"""
Custom COCOeval with distance evaluation metrics.
Ported from the original deep_old implementation.
"""
from pycocotools.cocoeval import *
import torch
import numpy as np
import time
import datetime


class Custom_COCOeval(COCOeval):
    """
    A customized version of std cocoeval, which computes also an evaluation over distance regression results.
    """

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        dist_error = -np.ones((T, K, A))  # the distance error for data where we have a valid distance estimation (distance >= 0)
        dist_invalid = -np.ones((T, K, A))  # the ratio between invalid distances (distance < 0) and all distances
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue

                    if m == 0:
                        # new code related to distance regression evaluation.
                        # note that this code is not influenced my maxDet.
                        gtMatches = [e['gtMatches'] for e in E]
                        gtIds = [e['gtIds'] for e in E]

                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        if m == 0:
                            total_distance_counter = 0
                            cumulative_diff_valid = 0.0
                            diff_count_valid = 0

                            for gt_id, el in zip(gtIds, gtMatches):  # one for each image
                                if len(gt_id) == 0:
                                    continue
                                gt_id = gt_id[0]
                                for dt_id in el[m]:
                                    dt_id = int(dt_id)
                                    if dt_id == 0:
                                        continue
                                    gt = self.cocoGt.anns[gt_id]
                                    dt = self.cocoDt.anns[dt_id]
                                    if gt['image_id'] != dt['image_id']:
                                        continue  # Skip mismatched pairs instead of asserting
                                    detected_distance = dt['distance']
                                    if detected_distance >= 0:
                                        gt_distance = gt.get('avg_distance', gt.get('distance', -1))
                                        if gt_distance >= 0:
                                            diff = abs(gt_distance - detected_distance)
                                            cumulative_diff_valid += diff
                                            diff_count_valid += 1
                                    total_distance_counter += 1

                            dist_error[t, k, a] = cumulative_diff_valid / diff_count_valid if diff_count_valid > 0 else -1
                            dist_invalid[t, k, a] = (total_distance_counter - diff_count_valid) / total_distance_counter if total_distance_counter > 0 else -1

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'dist_error': dist_error,
            'dist_invalid_ratio': dist_invalid,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(metric='ap', iouThr=None, areaRng='all', maxDets=100):
            # metric can be 'ap', 'ar' or 'de', where de stands for distance error
            assert metric in ['ap', 'ar', 'de', 'dir']
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if metric == 'ap' \
                else 'Average Recall' if metric == 'ar' \
                else 'Average Distance Error' if metric == 'de' \
                else 'Distance Invalidity Ratio'
            typeStr = '(AP)' if metric == 'ap' else '(AR)' if metric == 'ar' else '(ADE)' if metric == 'de' else '(DIR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if metric == 'ap':
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            if metric == 'ar':
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]

            if metric == 'de':
                s = self.eval['dist_error']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind]

            if metric == 'dir':
                s = self.eval['dist_invalid_ratio']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((22,))
            stats[0] = _summarize('ap')
            stats[1] = _summarize('ap', iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize('ap', iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize('ap', areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize('ap', areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize('ap', areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize('ar', maxDets=self.params.maxDets[0])
            stats[7] = _summarize('ar', maxDets=self.params.maxDets[1])
            stats[8] = _summarize('ar', maxDets=self.params.maxDets[2])
            stats[9] = _summarize('ar', areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize('ar', areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize('ar', areaRng='large', maxDets=self.params.maxDets[2])

            stats[12] = _summarize('de', areaRng='all', maxDets=self.params.maxDets[2])
            stats[13] = _summarize('de', areaRng='small', maxDets=self.params.maxDets[2])
            stats[14] = _summarize('de', areaRng='medium', maxDets=self.params.maxDets[2])
            stats[15] = _summarize('de', areaRng='large', maxDets=self.params.maxDets[2])
            stats[16] = _summarize('de', iouThr=.5, maxDets=self.params.maxDets[2])

            stats[17] = _summarize('dir', areaRng='all', maxDets=self.params.maxDets[2])
            stats[18] = _summarize('dir', areaRng='small', maxDets=self.params.maxDets[2])
            stats[19] = _summarize('dir', areaRng='medium', maxDets=self.params.maxDets[2])
            stats[20] = _summarize('dir', areaRng='large', maxDets=self.params.maxDets[2])
            stats[21] = _summarize('dir', iouThr=.5, maxDets=self.params.maxDets[2])

            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()
