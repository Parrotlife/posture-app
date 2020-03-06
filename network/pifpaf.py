
import glob

import numpy as np
import torchvision
import torch
from PIL import Image, ImageFile
from openpifpaf.network import nets
from openpifpaf import decoder

class PifPaf:
    def __init__(self, args):
        """Instanciate the model"""
        model_pifpaf, _ = nets.factory_from_args(args)
        model_pifpaf = model_pifpaf.to(args.device)
        self.processor = decoder.factory_from_args(args, model_pifpaf)
        self.keypoints_whole = []

        # Scale the keypoints to the original image size for printing (if not webcam)
        if not args.webcam:
            self.scale_np = np.array([args.scale, args.scale, 1] * 17).reshape(17, 3)
        else:
            self.scale_np = np.array([1, 1, 1] * 17).reshape(17, 3)

    def fields(self, processed_images):
        """Encoder for pif and paf fields"""
        fields_batch = self.processor.fields(processed_images)
        return fields_batch

    def forward(self, image, processed_image_cpu, fields):
        """Decoder, from pif and paf fields to keypoints"""
        self.processor.set_cpu_image(image, processed_image_cpu)
        keypoint_sets, scores = self.processor.keypoint_sets(fields)

        if keypoint_sets.size > 0:
            self.keypoints_whole.append(np.around((keypoint_sets / self.scale_np), 1)
                                        .reshape(keypoint_sets.shape[0], -1).tolist())

        pifpaf_out = [
            {'keypoints': np.around(kps / self.scale_np, 1).reshape(-1).tolist(),
             'bbox': [np.min(kps[:, 0]) / self.scale_np[0, 0], np.min(kps[:, 1]) / self.scale_np[0, 0],
                      np.max(kps[:, 0]) / self.scale_np[0, 0], np.max(kps[:, 1]) / self.scale_np[0, 0]]}
            for kps in keypoint_sets
        ]
        return keypoint_sets, scores, pifpaf_out
