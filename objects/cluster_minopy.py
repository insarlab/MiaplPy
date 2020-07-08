#!/usr/bin/env python3
#############################################################
# Program is part of MiNoPy                                 #
# Author: Sara Mirzaee 2020                                 #
#############################################################
# Recommend import:
#     from minopy.objects import cluster_minopy as cluster


import time
from mintpy.objects.cluster import DaskCluster

# supported / tested clusters
CLUSTER_LIST = ['lsf', 'pbs', 'slurm', 'local']

class MDaskCluster(DaskCluster):
    """
        Generic dask cluster wrapper for parallel processing in blocks.

        This object takes in a computing function for one block in space.
        For the computing function:
            1. the output is always several matrices and one box.
            2. the number of matrices may vary for different applications/functions.
            3. all matrices will be in 2D in size of (len, wid) or 3D in size of (n, len, wid),
               thus, the last two dimension (in space) will be the same.
        This charateristics allows the automatic result collection without prior knowledge
            of the computing funciton, thus being a generic wrapper.

        Check phase_inversion.py as an example.

    """
    def run(self, func, func_data):
        """Wrapper function encapsulating submit_workers and compile_workers.

        For a generic result collection without prior knowledge of the computing function,
        we assume that the output of "func" is: several 2D or 3D matrices + a box

        :param func: function, a python function to run in parallel
        :param func_data: dict, a dictionary of the argument to pass to the function
        :param results: list[numpy.nd.array], arrays of the appropriate structure representing
               the final output of processed box (need to be in the same order as the function passed in
               submit_workers returns in)
        :param dimlimits: [image_length, image_width]
        :return:
        """
        from dask.distributed import Client

        # This line needs to be in a function or in a `if __name__ == "__main__":` block. If it is in no function
        # or "main" block, each worker will try to create its own client (which is bad) when loading the module
        print('initiate Dask client')
        self.client = Client(self.cluster)

        # split the primary box into sub boxes for each worker
        box = func_data["box"]
        range_window = func_data['range_window']
        azimuth_window = func_data['azimuth_window']

        sub_boxes = self.split_box2sub_boxes(box, range_window=range_window, azimuth_window=azimuth_window,
                                             num_split=self.num_worker, dimension='x')

        print('split patch into {} sub boxes in x direction for workers to process'.format(len(sub_boxes)))

        # submit job for each worker
        futures, submission_time = self.submit_job(func, func_data, sub_boxes)

        # assemble results from all workers
        return self.collect_result(futures, submission_time)

    @staticmethod
    def split_box2sub_boxes(box, range_window, azimuth_window, num_split, dimension='x'):
        """Divide the input box into `num_split` different sub_boxes.

        :param box: [x0, y0, x1, y1]: list[int] of size 4
        :param range_window: range window size for shp finding
        :param azimuth_window: azimuth window size for shp finding
        :param num_split: int, the number of sub_boxes to split a box into
        :param dimension: str = 'y' or 'x', the dimension along which to split the boxes
        :return: sub_boxes: list(list(4 int)), the splited sub boxes
        """

        x0, y0, x1, y1 = box
        length, width = y1 - y0, x1 - x0

        sub_boxes = []
        if dimension == 'y':
            if (length // num_split) < (2 * azimuth_window):
                num_split = length // (2 * azimuth_window)
            for i in range(num_split):
                start = (i * length) // num_split + y0
                end = ((i + 1) * length) // num_split + y0
                if i == num_split - 1:
                    end = y1
                sub_boxes.append([x0, start, x1, end])

        else:
            if (width // num_split) < (2 * range_window):
                num_split = width // (2 * range_window)
            for i in range(num_split):
                start = (i * width) // num_split + x0
                end = ((i + 1) * width) // num_split + x0
                if i == num_split - 1:
                    end = x1
                sub_boxes.append([start, y0, end, y1])

        return sub_boxes

    def collect_result(self, futures, submission_time):
        """Compile results from completed workers and recompiles their sub outputs into the output
        for the complete box being worked on.
        :param futures: list(dask.Future), list of futures representing future dask worker calculations
        :param results: list[numpy.nd.array], arrays of the appropriate structure representing
               the final output of processed box (need to be in the same order as the function passed in
               submit_workers returns in)
        :param box: numpy.nd.array, the initial complete box being processed
        :param submission_time: time, the time of submission of the dask workers (used to determine worker
               runtimes as a performance diagnostic)
        :return: results: tuple(numpy.nd.arrays), the processed results of the box
        """
        from dask.distributed import as_completed

        num_future = 0
        for future, sub_results in as_completed(futures, with_results=True):

            # message
            num_future += 1
            sub_t = time.time() - submission_time
            print("FUTURE #{} complete. Time used: {:.0f} seconds".format(num_future, sub_t))

        return
