"""
RVT Detection Module with Distance Estimation Support.

Extends the base detection module to support distance estimation
and evaluation using proper IoU-based matching.
"""

from typing import Any, Optional, Tuple, Union, Dict
from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode
from models.detection.yolox.utils.boxes import postprocess
from models.detection.yolox_extension.models.detector_distance import YoloXDetectorWithDistance
from nerve.training.rvt.utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from nerve.training.rvt.utils.evaluation.prophesee.io.box_loading import to_prophesee
from nerve.training.rvt.utils.evaluation.distance_evaluator import DistancePropheseeEvaluator
from nerve.training.rvt.utils.padding import InputPadderFromShape
from .utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches


class ModuleWithDistance(pl.LightningModule):
    """
    RVT Detection Module extended with distance estimation.
    
    Uses DistancePropheseeEvaluator for proper distance metrics
    with IoU-based matching, consistent with YOLOX evaluation.
    """
    
    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config

        self.mdl_config = full_config.model
        in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)

        # Use distance-enabled detector
        self.mdl = YoloXDetectorWithDistance(self.mdl_config)
        
        # Check if distance is enabled
        self.distance_enabled = hasattr(self.mdl_config, 'distance') and \
                               self.mdl_config.distance.get('enable', False)
        
        if self.distance_enabled:
            self.dist_min = self.mdl_config.distance.get('min_dist', 0.0)
            self.dist_max = self.mdl_config.distance.get('max_dist', 10.0)

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_name = self.full_config.dataset.name
        self.mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
        self.mode_2_batch_size: Dict[Mode, Optional[int]] = {}
        self.mode_2_psee_evaluator: Dict[Mode, Optional[Union[PropheseeEvaluator, DistancePropheseeEvaluator]]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        
        # Use distance evaluator if distance is enabled
        evaluator_cls = DistancePropheseeEvaluator if self.distance_enabled else PropheseeEvaluator
        evaluator_kwargs = {
            'dataset': dataset_name, 
            'downsample_by_2': self.full_config.dataset.downsample_by_factor_2
        }
        if self.distance_enabled:
            evaluator_kwargs['min_dist'] = self.dist_min
            evaluator_kwargs['max_dist'] = self.dist_max
        
        if stage == 'fit':  # train + val
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            if self.train_metrics_config.compute:
                self.mode_2_psee_evaluator[Mode.TRAIN] = evaluator_cls(**evaluator_kwargs)
            self.mode_2_psee_evaluator[Mode.VAL] = evaluator_cls(**evaluator_kwargs)
            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False
        elif stage == 'validate':
            mode = Mode.VAL
            self.mode_2_psee_evaluator[mode] = evaluator_cls(**evaluator_kwargs)
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        elif stage == 'test':
            mode = Mode.TEST
            self.mode_2_psee_evaluator[mode] = evaluator_cls(**evaluator_kwargs)
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError

    def forward(self,
                event_tensor: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                targets=None,
                distance_targets=None) \
            -> Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
        return self.mdl(x=event_tensor,
                        previous_states=previous_states,
                        retrieve_detections=retrieve_detections,
                        targets=targets,
                        distance_targets=distance_targets)

    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch['worker_id']

    def get_data_from_batch(self, batch: Any):
        return batch['data']

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        """Training step with distance estimation support."""
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            if batch_size > self.mode_2_batch_size[mode]:
                self.mode_2_batch_size[mode] = batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])
            else:
                token_masks = None

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors,
                                                                  previous_states=prev_states,
                                                                  token_mask=token_masks)
            prev_states = states

            current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                selected_indices=valid_batch_indices)
                obj_labels.extend(current_labels)
                ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                           selected_indices=valid_batch_indices)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        assert len(obj_labels) > 0
        
        # Batch the backbone features and labels
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, format_='yolox')
        labels_yolox = labels_yolox.to(dtype=self.dtype)

        # Extract distance targets if available
        distance_targets = None
        if self.distance_enabled:
            # Extract distances from each label if available
            all_distances = []
            for label in obj_labels:
                if hasattr(label, 'distance') and label.distance is not None:
                    all_distances.append(label.distance)
                elif hasattr(label, 'has_distance') and label.has_distance:
                    all_distances.append(label.distance)

            if all_distances:
                # Concatenate all distances into a single tensor
                distance_targets = th.cat([d if isinstance(d, th.Tensor) else th.tensor(d) 
                                           for d in all_distances])
                distance_targets = distance_targets.to(dtype=self.dtype, device=labels_yolox.device)

        # Forward through detection (and distance) head
        predictions, losses = self.mdl.forward_detect(
            backbone_features=selected_backbone_features,
            targets=labels_yolox,
            distance_targets=distance_targets
        )

        if self.mode_2_sampling_mode[mode] in (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
            predictions = predictions[-batch_size:]
            obj_labels = obj_labels[-batch_size:]

        pred_processed = postprocess(prediction=predictions,
                                     num_classes=self.mdl_config.head.num_classes,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

        assert losses is not None
        assert 'loss' in losses

        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-batch_size:],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-batch_size:],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-batch_size),
            ObjDetOutput.SKIP_VIZ: False,
            'loss': losses['loss']
        }

        # Logging
        prefix = f'{mode_2_string[mode]}/'
        log_dict = {f'{prefix}{k}': v for k, v in losses.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        if mode in self.mode_2_psee_evaluator:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            if self.train_metrics_config.detection_metrics_every_n_steps is not None and \
                    step > 0 and step % self.train_metrics_config.detection_metrics_every_n_steps == 0:
                self.run_psee_evaluator(mode=mode)

        return output
    
    def _extract_distances_from_labels(self, obj_labels: list) -> list:
        """
        Extract distance values from object labels.
        
        Args:
            obj_labels: List of ObjectLabels instances (ObjectLabels from labels.py)
            
        Returns:
            List of distance arrays
        """
        distances = []
        for label in obj_labels:
            # ObjectLabels class uses has_distance property and distance property
            if hasattr(label, 'has_distance') and label.has_distance:
                dist = label.distance
                if dist is not None:
                    # Convert tensor to numpy if needed
                    if hasattr(dist, 'cpu'):
                        distances.append(dist.cpu().numpy())
                    else:
                        distances.append(np.array(dist))
                else:
                    distances.append(np.array([]))
            elif hasattr(label, 'distance') and label.distance is not None:
                dist = label.distance
                if hasattr(dist, 'cpu'):
                    distances.append(dist.cpu().numpy())
                else:
                    distances.append(np.array(dist))
            else:
                # No distance data available
                distances.append(np.array([]))
        return distances
    
    def _extract_distances_from_predictions(self, predictions: list, pred_processed: list) -> list:
        """
        Extract distance values from model predictions.
        
        Args:
            predictions: Raw model predictions
            pred_processed: Processed predictions after NMS
            
        Returns:
            List of distance arrays per sample
        """
        distances = []
        
        for i, pred in enumerate(pred_processed):
            if pred is None or len(pred) == 0:
                distances.append(np.array([]))
                continue
            
            # Check if distance channel is present (8th column, index 7)
            # Format after postprocess: (x1, y1, x2, y2, obj_conf, class_conf, class_pred, distance)
            if isinstance(pred, th.Tensor) and pred.shape[-1] > 7:
                dist_vals = pred[:, 7].cpu().numpy()
                distances.append(dist_vals)
            elif isinstance(pred, np.ndarray) and pred.shape[-1] > 7:
                distances.append(pred[:, 7])
            else:
                # No distance in predictions, use zeros
                distances.append(np.zeros(len(pred)))
        
        return distances

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        """Validation/test step with distance support."""
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            if batch_size > self.mode_2_batch_size[mode]:
                self.mode_2_batch_size[mode] = batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        
        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or \
                                  (self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM)
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors, previous_states=prev_states)
            prev_states = states

            if collect_predictions:
                current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                if len(current_labels) > 0:
                    backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                    selected_indices=valid_batch_indices)
                    obj_labels.extend(current_labels)
                    ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                               selected_indices=valid_batch_indices)
                                                               
        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        
        if len(obj_labels) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}
            
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        predictions, _ = self.mdl.forward_detect(backbone_features=selected_backbone_features)

        # Postprocess predictions
        # Note: with distance enabled, predictions have distance channel appended
        pred_processed = postprocess(prediction=predictions,
                                     num_classes=self.mdl_config.head.num_classes,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

        # For visualization
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-1],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-1],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-1)[0],
            ObjDetOutput.SKIP_VIZ: False,
        }

        if self.started_training:
            evaluator = self.mode_2_psee_evaluator[mode]
            
            if self.distance_enabled and isinstance(evaluator, DistancePropheseeEvaluator):
                # Add labels and predictions with distance
                gt_distances = self._extract_distances_from_labels(obj_labels)
                pred_distances = self._extract_distances_from_predictions(predictions, pred_processed)
                
                evaluator.add_labels_with_distance(loaded_labels_proph, gt_distances)
                evaluator.add_predictions_with_distance(yolox_preds_proph, pred_distances)
            else:
                # Standard evaluation without distance
                evaluator.add_labels(loaded_labels_proph)
                evaluator.add_predictions(yolox_preds_proph)

        return output

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.VAL)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST)

    def run_psee_evaluator(self, mode: Mode):
        """Run evaluation with distance metrics if enabled."""
        psee_evaluator = self.mode_2_psee_evaluator[mode]
        batch_size = self.mode_2_batch_size[mode]
        hw_tuple = self.mode_2_hw[mode]
        
        if psee_evaluator is None:
            warn(f'psee_evaluator is None in {mode=}', UserWarning, stacklevel=2)
            return
            
        assert batch_size is not None
        assert hw_tuple is not None
        
        if psee_evaluator.has_data():
            metrics = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0],
                                                     img_width=hw_tuple[1])
            assert metrics is not None

            prefix = f'{mode_2_string[mode]}/'
            step = self.trainer.global_step
            log_dict = {}
            
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError
                    
                assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}'
                log_dict[f'{prefix}{k}'] = value.to(self.device)
            
            self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                for k, v in log_dict.items():
                    dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0:
                        log_dict[k] /= dist.get_world_size()
                        
            if self.trainer.is_global_zero:
                add_hack = 2
                self.logger.log_metrics(metrics=log_dict, step=step + add_hack)
                
                # Log distance metrics prominently if available
                if self.distance_enabled:
                    self._log_distance_summary(metrics, prefix)

            psee_evaluator.reset_buffer()
        else:
            warn(f'psee_evaluator has not data in {mode=}', UserWarning, stacklevel=2)
    
    def _log_distance_summary(self, metrics: dict, prefix: str):
        """Log distance metrics summary."""
        print("\n" + "="*70)
        print(f"Distance Estimation Metrics ({prefix.strip('/')})")
        print("="*70)
        
        ade = metrics.get('distance/ADE', -1)
        if ade >= 0:
            print(f"  ADE (Avg Dist Err): {ade:.3f} m  <- Compare with YOLOX")
            print(f"  RMSE:               {metrics.get('distance/RMSE', 0):.3f} m")
            print(f"  Matched Samples:    {metrics.get('distance/NumSamples', 0)}")
            print(f"  Acc @ 0.5m:         {metrics.get('distance/Acc@0.5m', 0)*100:.1f}%")
            print(f"  Acc @ 1.0m:         {metrics.get('distance/Acc@1.0m', 0)*100:.1f}%")
            print(f"  Acc @ 2.0m:         {metrics.get('distance/Acc@2.0m', 0)*100:.1f}%")
        else:
            print("  No valid distance samples collected")
        print("="*70 + "\n")

    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        if self.started_training:
            assert self.mode_2_psee_evaluator[mode].has_data()
            self.run_psee_evaluator(mode=mode)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        assert self.mode_2_psee_evaluator[mode].has_data()
        self.run_psee_evaluator(mode=mode)

    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = th.optim.AdamW(self.mdl.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
