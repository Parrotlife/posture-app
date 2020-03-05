import argparse

from openpifpaf.network import nets
from openpifpaf import decoder

def cli():
    predict_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Predict (3D pose from images)

    ##Here we can add arguments to pass to the software.

    # Pifpaf
    nets.cli(predict_parser)
    decoder.cli(predict_parser, force_complete_pose=True, instance_threshold=0.15)
    predict_parser.add_argument('--scale', default=0.2, type=float, help='change the scale of the image to preprocess')
    #predict_parser.add_argument('--pose3d_model', help='path of 3Dpose model to load', default='data/models/pose3d-real-v0_3_9.pkl')
    predict_parser.add_argument('--pose3d_model', help='path of 3Dpose model to load', default='data/models/pose3d-v0_8.pkl')
    predict_parser.add_argument('--webcam', help='streaming', action='store_true')

    args = predict_parser.parse_args()
    return args

def main():
    
    args = cli()

    if args.webcam:
        from .visuals.pose3d_viz_arms import webcam
        webcam(args)

    

if __name__ == '__main__':
    main()