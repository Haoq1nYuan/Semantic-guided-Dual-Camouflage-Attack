import os.path
from colorsys import hsv_to_rgb
from math import sqrt, ceil
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.vq import kmeans, whiten
from tools.train_utils import *


class ColorExtractor:
    def __init__(self, img_idx):
        self.new_subfolder = set_save('results/extract_results')
        os.mkdir(os.path.join(self.new_subfolder, 'intermediate_results'))
        self.cluster_size = 640
        self.size = 1000
        self.compacted_map = np.zeros((self.cluster_size, self.cluster_size, 4))
        self.img_idx = img_idx
        self.last_idx = -1

        for idx in range(0, len(self.img_idx)):
            image_path = f'data/unity/{self.img_idx[idx]}.png'
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Required path does not exist: {image_path}")

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            edges = cv2.Canny(image, 100, 200)
            cv2.imwrite(os.path.join(self.new_subfolder, f"intermediate_results/{idx + 1}_Edges.png"), edges)

            image_rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(self.new_subfolder, f"intermediate_results/{idx + 1}_Original_bg.png"), image_rgba)

            for i in range(self.cluster_size):
                for j in range(self.cluster_size):
                    if edges[i][j] == 0:
                        image_rgba[i][j][3] = 0

            cv2.imwrite(os.path.join(self.new_subfolder, f"intermediate_results/{idx + 1}_Selected_color.png"), image_rgba)
            if not self.cluster_non_transparent_pixels(image_rgba):
                break

        cv2.imwrite(os.path.join(self.new_subfolder, "Compacted_color.png"), self.compacted_map)

        print("Analysis saved as:", self.new_subfolder)

    def cluster_non_transparent_pixels(self, array_image):
        """merge all high-frequency regions into a single image"""

        # get positions of non-transparent pixels
        non_transparent_coords = np.column_stack(np.where(array_image[:, :, 3] > 0))

        # fill high-frequency pixels sequentially
        flag = True
        for i, coord in enumerate(non_transparent_coords):
            self.last_idx += 1
            if self.last_idx >= 640 * 640:
                flag = False
                break
            self.compacted_map[self.last_idx // array_image.shape[1], self.last_idx % array_image.shape[1], :] = array_image[coord[0], coord[1], :]

        return flag

    def run_cluster(self, cluster_num, rounds):
        """execute color clustering pipeline: analyze, visualize and save results"""

        # perform k-means clustering on compacted image
        means, match_counts = self.cluster_analyze(os.path.join(self.new_subfolder, "Compacted_color.png"), cluster_num, rounds)

        # prepare visualization figure
        figure, ax = self.prep_figure(self.size)
        self.draw_color_patches(means, match_counts, self.size, ax)

        self.save_file(figure, self.size, os.path.join(self.new_subfolder, "Extracted_colors.png"))

    def cluster_analyze(self, filename, num_means, rounds):
        """
        Perform k-means clustering on image to extract dominant colors.

        Returns a tuple of two objects:
          * A list of the means in the form [(h, s, v), ...].  Each of the
            (h, s, v) values are in the range [0, 1].
          * A list of the same length containing the number of pixels that
            are closest to the mean at the same index in the first list.
        """

        # load and convert the img from rgb to hsv (all values in range [0, 1])
        img = Image.open(filename).convert('RGB')
        flat_img = np.asarray(img)
        flat_img = flat_img / 255.0
        flat_img = matplotlib.colors.rgb_to_hsv(flat_img)

        # reshape to a Nx3 array
        img = np.reshape(flat_img, (len(flat_img) * len(flat_img[0]), 3))

        # perform k-means clustering
        stdev = self.get_stdev(img)
        whitened = whiten(img)
        means, _ = kmeans(whitened, num_means, iter=rounds)
        unwhitened = means * stdev

        unwhitened = list(map(tuple, unwhitened))
        unwhitened.sort()

        # count the number of pixels that are closest to each centroid
        match_counts = [0] * len(unwhitened)
        for i, row in enumerate(flat_img):
            for a in row:
                distances = [self.dist(a, b) for b in unwhitened]
                min_index = distances.index(min(distances))
                match_counts[min_index] += 1

        return unwhitened, match_counts

    def prep_figure(self, size):
        """initialize matplotlib figure with transparent background"""

        bgcolor = (1.0, 1.0, 1.0)
        fig = plt.figure(facecolor=(0.5, 0.5, 0.5), linewidth=0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        bg = self.background(size, bgcolor)
        ax.imshow(bg.astype('uint8'), origin='lower')
        return fig, ax

    def draw_color_patches(self, means, match_counts, size, ax):
        """
        visualize color clusters as rectangles with size proportional to frequency

        Args:
        means: List of HSV color values
        match_counts: List of pixel counts for each color
        size: Figure size
        ax: Matplotlib axes object
        """

        max_count = max(match_counts)
        num_rows = int(ceil(sqrt(len(means))))
        width = size / num_rows
        height = size / num_rows

        # draw colored rectangle for each cluster
        for i, mean in enumerate(means):
            rgb_mean = hsv_to_rgb(*mean)
            rgb_mean = [x * 256.0 for x in rgb_mean]
            x_coord = width * (i % num_rows)
            y_coord = height * ((num_rows - 1) - (i // num_rows))

            # scale rectangle size based on cluster frequency
            count = match_counts[i]
            count_ratio = count / float(max_count)
            adjusted_size = (width * 0.9) * count_ratio

            rect_coords = (x_coord, y_coord + (width * 0.1))
            ax.add_patch(Rectangle(rect_coords, adjusted_size, adjusted_size,
                                   facecolor=self.rgb(*rgb_mean), edgecolor="none"))

            # add HSV values and frequency ratio as text label
            adjusted_hsv = (mean[0] * 360.0, mean[1] * 100.0, mean[2] * 100.0)
            ax.text(x_coord, y_coord, ",".join("%d" % int(x) for x in adjusted_hsv) + f" {count_ratio:.2f}")

    @staticmethod
    def background(size, color):
        """create solid color background image of given size"""

        color = [int(c * 255) for c in color]
        bg = np.array([[color]])
        # expand single pixel to full size
        bg = np.repeat(bg, size, 0)
        bg = np.repeat(bg, size, 1)
        return bg

    @staticmethod
    def rgb(red, green, blue):
        return red / 255.0, green / 255.0, blue / 255.0

    @staticmethod
    def get_stdev(array):
        return np.std(array, axis=0)

    @staticmethod
    def dist(a, b):
        (xa, ya, za) = a
        (xb, yb, zb) = b
        return sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

    @staticmethod
    def save_file(fig, size, filename):
        bgcolor = (1.0, 1.0, 1.0)
        dpi = 100.0
        fig.set_size_inches(size / dpi, size / dpi)
        plt.savefig(filename, facecolor=bgcolor, dpi=dpi)
