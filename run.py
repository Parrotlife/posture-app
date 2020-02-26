import argparse

from openpifpaf.network import nets
from openpifpaf import decoder

def cli():
    predict_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Predict (2D pose and/or 3D location from images)
    # General
    predict_parser.add_argument('--networks', nargs='+', help='Run pifpaf and/or monoloco', default=['monoloco'])
    predict_parser.add_argument('images', nargs='*', help='input images')
    predict_parser.add_argument('--glob', help='glob expression for input images (for many images)')
    predict_parser.add_argument('-o', '--output-directory', help='Output directory')
    predict_parser.add_argument('--show', help='to show images', action='store_true')

    # Pifpaf
    nets.cli(predict_parser)
    decoder.cli(predict_parser, force_complete_pose=True, instance_threshold=0.15)
    predict_parser.add_argument('--scale', default=0.2, type=float, help='change the scale of the image to preprocess')

    # Monoloco
    predict_parser.add_argument('--mono_model', help='path of MonoLoco model to load', default='data/models/monoloco-pretrained.pkl')
    #predict_parser.add_argument('--pose3d_model', help='path of 3Dpose model to load', default='data/models/pose3d-real-v0_3_9.pkl')
    predict_parser.add_argument('--pose3d_model', help='path of 3Dpose model to load', default='data/models/pose3d-v0_8.pkl')
    predict_parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=512)
    predict_parser.add_argument('--transform', help='transformation for the pose', default='None')
    predict_parser.add_argument('--draw_box', help='to draw box in the images', action='store_true')
    predict_parser.add_argument('--predict', help='whether to make prediction', action='store_true')
    predict_parser.add_argument('--z_max', type=int, help='maximum meters distance for predictions', default=5)
    predict_parser.add_argument('--n_dropout', type=int, help='Epistemic uncertainty evaluation', default=0)
    predict_parser.add_argument('--dropout', type=float, help='dropout parameter', default=0.2)
    predict_parser.add_argument('--webcam', help='monoloco streaming', action='store_true')

    args = predict_parser.parse_args()
    return args

def main():
    
    args = cli()

    if args.webcam:
        from .visuals.pose3d_viz_arms import webcam
        webcam(args)

    

if __name__ == '__main__':
    main()