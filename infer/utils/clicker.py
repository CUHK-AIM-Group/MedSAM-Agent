import numpy as np
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import os


def ensure_intermediate_results_dir(dataset_name='default', output_dir='./intermediate_results'):
    """Ensure the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class Clicker(object):
    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1, click_indx_offset=0,
                 dataset_name='default', output_dir='./intermediate_results'):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()
        self.visualize = False
        self.dataset_name = dataset_name
        self.visualize_dir = ensure_intermediate_results_dir(dataset_name, output_dir)
        self.object_id = "object"
        self.file_name = "unknown"

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask, file_name=None):
        assert self.gt_mask is not None
        self.file_name = file_name
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def get_last_click(self):
        """Get the last added point for iterative segmentation."""
        if len(self.clicks_list) > 0:
            return [self.clicks_list[-1]]
        return []

    def _get_next_click(self, pred_mask, padding=True,save_dir='./visualizations'):
        visualize = self.visualize
        # Ensure the save directory exists
        if self.visualize:
            save_dir =self.visualize_dir
        if visualize and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        click_idx = len(self.clicks_list) + 1
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("False Negative Mask (fn_mask)")
            plt.imshow(fn_mask, cmap='gray')

            plt.subplot(1, 2, 2)
            plt.title("False Positive Mask (fp_mask)")
            plt.imshow(fp_mask, cmap='gray')
            plt.savefig(os.path.join(save_dir, f'{self.file_name}_{str(self.object_id)}_fn_fp_masks_{click_idx}.png'))
            plt.close()

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Distance Transform of False Negative (fn_mask_dt)")
            plt.imshow(fn_mask_dt, cmap='jet')

            plt.subplot(1, 2, 2)
            plt.title("Distance Transform of False Positive (fp_mask_dt)")
            plt.imshow(fp_mask_dt, cmap='jet')
            plt.savefig(f'{save_dir}/{self.file_name}_{str(self.object_id)}_distance_transforms_{click_idx}.png')
            plt.close()

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("False Negative Distance Transform (Filtered)")
            #plt.imshow(fn_mask_dt, cmap='jet')

            plt.subplot(1, 2, 2)
            plt.title("False Positive Distance Transform (Filtered)")
            #plt.imshow(fp_mask_dt, cmap='jet')
            plt.savefig(f'{save_dir}/{self.file_name}_{str(self.object_id)}_filtered_distance_transforms_{click_idx}.png')
            plt.close()

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        if visualize:
            plt.figure(figsize=(5, 5))
            plt.title("Next Click Position")
            plt.imshow(pred_mask, cmap='hot')  # Background image is the predicted mask
            if is_positive:
                plt.scatter(coords_x[0], coords_y[0], color='red', s=100)  # The s parameter controls point size
            else:
                plt.scatter(coords_x[0], coords_y[0], color='blue', s=100)
            plt.savefig(os.path.join(save_dir, f'{self.file_name}_{str(self.object_id)}_next_click_position_{click_idx}.png'))
            plt.close()


        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click,radius=0):
        coords = click.coords
        if radius > 0:
            # Consider exclusion radius
            x1, y1 = click.coords
            p1 = click.is_positive
            for prev_click in self.clicks_list:
                x2, y2 = prev_click.coords
                p2 = prev_click.is_positive
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < radius and p1 != p2:
                    self.clicks_list.remove(prev_click)
                    if prev_click.is_positive:
                        self.num_pos_clicks -= 1
                    else:
                        self.num_neg_clicks -= 1
        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)

class Clicker_sampler(object):
    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1, click_indx_offset=0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()
        self.visualize = False

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask, file_name=None):
        assert self.gt_mask is not None
        self.file_name = file_name
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]
        
    def _get_next_click(self, pred_mask, padding=True,save_dir='./visualizations'):
        visualize = self.visualize
        # Ensure the save directory exists
        if self.visualize:
            save_dir =self.visualize_dir
        if visualize and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        click_idx = len(self.clicks_list) + 1
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("False Negative Mask (fn_mask)")
            plt.imshow(fn_mask, cmap='gray')

            plt.subplot(1, 2, 2)
            plt.title("False Positive Mask (fp_mask)")
            plt.imshow(fp_mask, cmap='gray')
            plt.savefig(os.path.join(save_dir, f'{self.file_name}_{str(self.object_id)}_fn_fp_masks_{click_idx}.png'))
            plt.close()

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Distance Transform of False Negative (fn_mask_dt)")
            plt.imshow(fn_mask_dt, cmap='jet')

            plt.subplot(1, 2, 2)
            plt.title("Distance Transform of False Positive (fp_mask_dt)")
            plt.imshow(fp_mask_dt, cmap='jet')
            plt.savefig(f'{save_dir}/{self.file_name}_{str(self.object_id)}_distance_transforms_{click_idx}.png')
            plt.close()

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("False Negative Distance Transform (Filtered)")
            #plt.imshow(fn_mask_dt, cmap='jet')

            plt.subplot(1, 2, 2)
            plt.title("False Positive Distance Transform (Filtered)")
            #plt.imshow(fp_mask_dt, cmap='jet')
            plt.savefig(f'{save_dir}/{self.file_name}_{str(self.object_id)}_filtered_distance_transforms_{click_idx}.png')
            plt.close()

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        if visualize:
            plt.figure(figsize=(5, 5))
            plt.title("Next Click Position")
            plt.imshow(pred_mask, cmap='hot')  # Background image is the predicted mask
            if is_positive:
                plt.scatter(coords_x[0], coords_y[0], color='red', s=100)  # The s parameter controls point size
            else:
                plt.scatter(coords_x[0], coords_y[0], color='blue', s=100)
            plt.savefig(os.path.join(save_dir, f'{self.file_name}_{str(self.object_id)}_next_click_position_{click_idx}.png'))
            plt.close()


        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def random_sample_click(self,pos_num,neg_num):
        pos_clicks = []
        neg_clicks = []
        # Sample positive clicks from the gt_mask
        pos_coords = np.where(self.gt_mask)
        for i in range(pos_num):
            idx = np.random.randint(len(pos_coords[0]))
            pos_clicks.append(Click(is_positive=True,coords=(pos_coords[0][idx],pos_coords[1][idx])))
        
        # Sample negative clicks from the gt_mask
        neg_coords = np.where(np.logical_not(self.gt_mask))
        for i in range(neg_num):
            idx = np.random.randint(len(neg_coords[0]))
            neg_clicks.append(Click(is_positive=False,coords=(neg_coords[0][idx],neg_coords[1][idx])))
        
        # add clicks to the clicker object
        for click in pos_clicks:
            self.add_click(click)
        
        for click in neg_clicks:
            self.add_click(click)

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Click:
    def __init__(self, is_positive, coords, indx=None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy
