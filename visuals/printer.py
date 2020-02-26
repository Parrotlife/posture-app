import math
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils import pixel_to_camera

class Printer:
    """
    Print results on images: birds eye view and computed distance
    """
    FONTSIZE_BV = 16
    FONTSIZE = 18
    TEXTCOLOR = 'darkorange'
    COLOR_KPS = 'yellow'

    def __init__(self, image, kk, epistemic=False, z_max=30, fig_width=10):

        self.im = image
        self.kk = kk
        self.epistemic = epistemic
        self.z_max = z_max  # To include ellipses in the image
        self.y_scale = 1
        self.width = self.im.size[0]
        self.height = self.im.size[1]
        self.fig_width = fig_width
        self.cmap = cm.get_cmap('jet')


        # Define variables of the class to change for every image
        self.mpl_im0 = self.stds_ale = self.stds_epi = self.xx_pred = self.zz_pred =\
            self.uv_centers = self.uv_shoulders = self.uv_kps = self.boxes  = \
            self.uv_camera = self.radius = None

    def _process_mono_results(self, dic_ann):
        # Include the vectors inside the interval given by z_max
        self.stds_ale = dic_ann['stds_ale']
        self.stds_epi = dic_ann['stds_epi']
        
        self.xx_pred = [xx[0] for xx in dic_ann['xyz_pred']]
        self.zz_pred = [xx[2] if xx[2] < self.z_max - self.stds_epi[idx] else 0
                        for idx, xx in enumerate(dic_ann['xyz_pred'])]
        
        self.uv_shoulders = dic_ann['uv_shoulders']
        self.boxes = dic_ann['boxes']


        self.uv_camera = (int(self.im.size[0] / 2), self.im.size[1])
        self.radius = 11 / 1600 * self.width
    
    def _process_pose3d_results(self, dic_ann):
        # Include the vectors inside the interval given by z_max
        self.angles = dic_ann['angles']
        self.poses = dic_ann['pose']
        self.poses_2d = dic_ann['2d pose']
        

    def factory_axes(self):
        """Create axes for figures: front bird combined"""
        axes = []
        figures = []

        #  Initialize combined figure, resizing it for aesthetic proportions

        self.y_scale = self.width / (self.height * 1.8)  # Defined proportion
        if self.y_scale < 0.95 or self.y_scale > 1.05:  # allows more variation without resizing
            self.im = self.im.resize((self.width, round(self.height * self.y_scale)))
        self.width = self.im.size[0]
        self.height = self.im.size[1]
        fig_width = self.fig_width #+ 0.6 * self.fig_width
        fig_height = self.fig_width * self.height / self.width

        fig_ar_1 = 2
        width_ratio = 1.9

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=False, gridspec_kw={'width_ratios': [1, 1, 1]},
                                        figsize=(fig_width, fig_height))
        ax1.set_aspect(fig_ar_1)
        fig.set_tight_layout(True)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

        figures.append(fig)
        
        # Create front figure axis
        ax0 = self.set_axes(ax0, axis=0)

        divider = make_axes_locatable(ax0)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        bar_ticks = self.z_max // 5 + 1
        norm = matplotlib.colors.Normalize(vmin=0, vmax=self.z_max)
        scalar_mappable = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        scalar_mappable.set_array([])
        plt.colorbar(scalar_mappable, ticks=np.linspace(0, self.z_max, bar_ticks),
                        boundaries=np.arange(- 0.05, self.z_max + 0.1, .1), cax=cax, label='Z [m]')

        axes.append(ax0)

        # Initialize bird-eye-view figure
        #fig1, ax1 = plt.subplots(1, 1)
        #fig1.set_tight_layout(True)
        #figures.append(fig1)
           
        ax1 = self.set_axes(ax1, axis=1)  # Adding field of view
        axes.append(ax1)

            # Initialize side-view figure
        #fig2, ax2 = plt.subplots(1, 1)
        #fig2.set_tight_layout(True)
        #figures.append(fig2)
        
        ax2 = self.set_axes(ax2, axis=2)  # Adding side view
        axes.append(ax2)

        return figures, axes

    def draw(self, figures, axes, dic_out_mono, dic_out_pose3d, image, pifpaf_out, draw_text=True, legend=True, draw_box=False):

        # Process the annotation dictionary of monoloco
        self._process_mono_results(dic_out_mono)
        self._process_pose3d_results(dic_out_pose3d)
        

        # Draw the front figure
        num = 0
        self.mpl_im0.set_data(image)
        for idx, uv in enumerate(self.uv_shoulders):

            print('2d projection', min(self.poses_2d[idx].reshape((21, 2)).transpose()[0]),min(self.poses_2d[idx].reshape((21, 2)).transpose()[1]),\
                max(self.poses_2d[idx].reshape((21, 2)).transpose()[0]), max(self.poses_2d[idx].reshape((21, 2)).transpose()[1]))

            print('hip', self.poses_2d[idx].reshape((21, 2))[18])
            
            print('pifpaf', pifpaf_out[idx])
            
            if self.zz_pred[idx] > -5:
                color = self.cmap((self.zz_pred[idx] % self.z_max) / self.z_max)
                self.draw_circle(axes, uv, color)

                for point in self.poses_2d[idx].reshape((21, 2)):
                    self.draw_circle(axes, point, color)

                # for point in np.array(pifpaf_out[idx]['keypoints']).reshape((17, 2)):
                #     self.draw_circle(axes, point, color)

                self.draw_boxes(axes, idx, color)

                if draw_text:
                    self.draw_text_front(axes, uv, num)
                    num += 1

        # Draw the bird figure
        num = 0
        for idx, _ in enumerate(self.xx_pred):
            
            if self.zz_pred[idx] > -1:

                # Draw ground truth and predicted ellipses
                self.draw_ellipses(axes, idx)

                # Draw bird eye view text
                if draw_text:
                    self.draw_text_bird(axes, idx, num)
                    num += 1

        # Draw the side figure
        num = 0
        #connections = [(0,20),(20,17),(17,19),(19,18),(0,13),(0,14),(13,15),(14,16),(1,4),(18,7),(18,10)]
            
        for idx, _ in enumerate(self.xx_pred):
            
            pose = self.poses_2d[idx].reshape((21,2)).transpose()
            pose2 = pifpaf_out[idx].reshape((21,2)).transpose()
            shift = 2*num
            #we find xyz for each tuple of joints to draw

            # axes[2].scatter(pose[0],-pose[1], color="black")
            # axes[2].scatter(pose2[0],-pose2[1], color="blue")

            for i in range(21):
                axes[2].scatter([pose[0][i], pose2[0][i]],[-pose[1][i], -pose2[1][i]])
                
            num += 1
        
        # Add the legend
        if legend:
            draw_legend(axes)

        # Draw, save or/and show the figures
        for idx, fig in enumerate(figures):
            fig.canvas.draw()
            
            fig.show()

    def draw_ellipses(self, axes, idx):
        """draw uncertainty ellipses"""
        angle = get_angle(self.xx_pred[idx], self.zz_pred[idx])
        ellipse_ale = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_ale[idx] * 2,
                              height=1, angle=angle, color='b', fill=False, label="Aleatoric Uncertainty",
                              linewidth=1.3)
        ellipse_var = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_epi[idx] * 2,
                              height=1, angle=angle, color='r', fill=False, label="Uncertainty",
                              linewidth=1, linestyle='--')

        axes[1].add_patch(ellipse_ale)
        if self.epistemic:
            axes[1].add_patch(ellipse_var)

        axes[1].plot(self.xx_pred[idx], self.zz_pred[idx], 'ro', label="Predicted", markersize=3)

    def draw_boxes(self, axes, idx, color):
        ww_box = self.boxes[idx][2] - self.boxes[idx][0]
        hh_box = (self.boxes[idx][3] - self.boxes[idx][1]) * self.y_scale

        rectangle = Rectangle((self.boxes[idx][0], self.boxes[idx][1] * self.y_scale),
                              width=ww_box, height=hh_box, fill=False, color=color, linewidth=3)
        
        axes[0].add_patch(rectangle)

    def draw_text_front(self, axes, uv, num):
        axes[0].text(uv[0] + self.radius, uv[1] * self.y_scale - self.radius, str(num),
                     fontsize=self.FONTSIZE, color=self.TEXTCOLOR, weight='bold')

    def draw_text_bird(self, axes, idx, num):
        """Plot the number in the bird eye view map"""

        std = self.stds_epi[idx] if self.stds_epi[idx] > 0 else self.stds_ale[idx]
        theta = math.atan2(self.zz_pred[idx], self.xx_pred[idx])

        delta_x = std * math.cos(theta)
        delta_z = std * math.sin(theta)

        axes[1].text(self.xx_pred[idx] + delta_x, self.zz_pred[idx] + delta_z,
                     str(num), fontsize=self.FONTSIZE_BV, color='darkorange')
        

    def draw_circle(self, axes, uv, color):

        circle = Circle((uv[0], uv[1] * self.y_scale), radius=self.radius, color=color, fill=True)
        axes[0].add_patch(circle)

    def set_axes(self, ax, axis):
        assert axis in (0, 1, 2)

        if axis == 0:
            ax.set_axis_off()
            ax.set_xlim(0, self.width)
            ax.set_ylim(self.height, 0)
            self.mpl_im0 = ax.imshow(self.im)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        elif axis == 1:
            uv_max = [0., float(self.height)]
            xyz_max = pixel_to_camera(uv_max, self.kk, self.z_max)
            x_max = abs(xyz_max[0])  # shortcut to avoid oval circles in case of different kk
            ax.plot([0, x_max], [0, self.z_max], 'k--')
            ax.plot([0, -x_max], [0, self.z_max], 'k--')
            ax.set_ylim(0, self.z_max+1)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Z [m]")
        else:
            ax.set_ylim(-0.2, 1.2)
            ax.set_xlim(-1.2, 1.2)
            ax.set_xlabel("Y [m]")
            ax.set_ylabel("Z [m]")


        return ax


def draw_legend(axes):
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys())


def get_angle(xx, zz):
    """Obtain the points to plot the confidence of each annotation"""

    theta = math.atan2(zz, xx)
    angle = theta * (180 / math.pi)

    return angle
