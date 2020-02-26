import time

import numpy as np

import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from .printer import Printer
from ..network import PifPaf, MonoLoco, Pose3d
from ..network.process import preprocess_pifpaf_1, preprocess_pifpaf_2, image_transform

LOGI_KK = [[0.7477614068838029, -0.0008111695246478873,  0.5134789751408451],
           [0.0,                      1.33178042553125, 0.45684016428124996],
           [0.                ,                     0.,                  1.]]

body_props = {'neck': 0.07729765064733189,
                'shoulder': 0.13887964509275452,
                'biceps': 0.1903499496025787,
                'forearm': 0.1856855142453098,
                'quad': 0.2250013372092681,
                'tibias': 0.2199742522543633,
                'hip': 0.16409048728199763,
                'back': 0.22711310836812693}
face_pos = {'nose': [0.0, 0.0, 0.07331183671724711],
            'eye': [0.02339558783308983, 0.02517361111111116, 0.058635183635287635],
            'ear': [0.05771397604574112, 0.0, 0.0]}

def webcam(args):


    # add args.device
    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    
    # load models
    args.camera = True
    pifpaf = PifPaf(args)
    monoloco = MonoLoco(model=args.mono_model, device=args.device)
    pose3d = Pose3d(model=args.pose3d_model, device=args.device)

    # # # Start recording
    cam = cv2.VideoCapture(0)
    visualizer = None

    while True:
        start = time.time()
        end = start+10000
        ret, frame = cam.read()
        image = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        height, width, _ = image.shape
        #print('resized image size: {}'.format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image_cpu = image_transform(image.copy())
        processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)
        fields = pifpaf.fields(torch.unsqueeze(processed_image, 0))[0]
        _, _, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)
        #print(pifpaf_out)

        if not ret:
            break
        key = cv2.waitKey(1)

        if key % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        pil_image = Image.fromarray(image)

        kk = [[x*width  for x in LOGI_KK[0]],
              [x*height for x in LOGI_KK[1]],
               LOGI_KK[2]]  # intrinsics for Logi webcams
        if visualizer is None:
           visualizer = Visualizer(kk, args)(pil_image)
           visualizer.send(None)

        if pifpaf_out:
            mono_boxes, mono_keypoints = preprocess_pifpaf_1(pifpaf_out, (width, height))

            pose3d_inputs = preprocess_pifpaf_2(pifpaf_out, body_props, face_pos, kk, arms = True)


            mono_outputs, mono_varss = monoloco.forward(mono_keypoints, kk)
            #print(mono_outputs)
            pose3d_outputs = pose3d.forward(pose3d_inputs).cpu().detach().numpy()
            #print('angles are:', pose3d_outputs)
            dic_out_mono = monoloco.post_process(mono_outputs, mono_varss, mono_boxes, mono_keypoints, kk)

            dic_out_pose3d = pose3d.post_process(pose3d_outputs, body_props, face_pos)

            
            visualizer.send((pil_image, dic_out_mono, dic_out_pose3d, pose3d_inputs[:,:42]))
        #     end = time.time()
        # print("run-time: {:.2f} ms".format((end-start)*1000))

    cam.release()

    cv2.destroyAllWindows()


class Visualizer:
    def __init__(self, kk, args, epistemic=False):

        self.kk = kk
        self.args = args
        self.z_max = args.z_max
        self.epistemic = epistemic
        

    def __call__(self, first_image, fig_width=4.0, **kwargs):
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (fig_width, fig_width * first_image.size[0] / first_image.size[1])

        printer = Printer(first_image, kk=self.kk,
                          z_max=self.z_max, epistemic=self.epistemic)
        figures, axes = printer.factory_axes()
        
        for fig in figures:
            fig.show()

        while True:
            image, dict_mono, dict_pose3d, pifpaf_out = yield
            
            while axes and (axes[1] and axes[1].patches):  # for front -1==0, for bird/combined -1 == 1
                
                if axes[0].patches:
                    for i in range(len(axes[0].patches)):
                        del axes[0].patches[-1]
                    del axes[0].texts[-1]
                    
                    if axes[1].patches:
                        del axes[1].patches[0]  # the one became the 0
                        if len(axes[1].lines) > 2:
                            del axes[1].lines[2]
                            if axes[1].texts:  # in case of no text
                                del axes[1].texts[0]
             
                    axes[2].clear()
                    axes[2] = printer.set_axes(axes[2], 2)
                        #del axes[2].lines[i]

            printer.draw(figures, axes, dict_mono, dict_pose3d, image, pifpaf_out)
            mypause(0.01)


def mypause(interval):
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)