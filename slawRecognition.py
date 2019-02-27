import numpy as np
from math import sin, cos
import math
import cv
import cv2
#import rospy
#import tf.transformations
#from slaw_msgs.msg import ObjectPoseConfidence, ObjectPoseConfidenceArray
#import vision_constants
#from slaw_recognition.j48 import j48_robocup_2016, j48_robocup_2016_cont
#from vision_helpers import filter_contours, merge_contours, get_closest_rgb
#from metric_contour import MetricContour, MerticContourList
#import multiprocessing as mp

from itertools import cycle

import vision_constants
from slaw_recognition.vision_constants import CONTOUR_MAX_AREA, CONTOUR_MIN_AREA, MIN, USE_JAVA_CLASSIFIER


class MetricContour:
    def __init__(self, pixel_contour, depth, cam_params):
        self.CX = cam_params[0]
        self.CY = cam_params[1]
        self.FX = cam_params[2]
        self.FY = cam_params[3]
        self.pixel_contour = pixel_contour
        self.metric_contour = []
        self.depth = depth
        self.dx = self.depth / self.FX
        self.dy = self.depth / self.FY
        for pt in self.pixel_contour:
            X = (pt[0][0] - self.CX) * self.dx
            Y = (pt[0][1] - self.CY) * self.dy
            self.metric_contour.append([X, Y, depth])

    def metric_centre_dist(self, pix1, pix2):
        X1 = (pix1[0] - self.CX) * self.dx
        Y1 = (pix1[1] - self.CY) * self.dy
        X2 = (pix2[0] - self.CX) * self.dx
        Y2 = (pix2[1] - self.CY) * self.dy
        return np.linalg.norm(np.array([X1,Y1]) - np.array([X2,Y2]))

    def reprojectContour(self, depth):
        self.depth = depth
        self.dx = self.depth / self.FX
        self.dy = self.depth / self.FY
        for pt in self.pixel_contour:
            X = (pt[0][0] - self.CX) * self.dx
            Y = (pt[0][1] - self.CY) * self.dy
            self.metric_contour.append([X, Y, depth])

    def getMetricArea(self):
        x = []
        y = []
        for mcont in self.metric_contour:
            x.append(mcont[0])
            y.append(mcont[1])
        return MetricContour.PolyArea(np.array(x), np.array(y))

    def getPerimeter(self):
        return sum(MetricContour.get_distances(self.metric_contour))

    def projectBox(self, box):
        mbox = []
        for point in box:
            mx = (point[0] - self.CX) * self.dx
            my = (point[1] - self.CY) * self.dx
            mbox.append([mx, my])
        return np.array(mbox)

    def getMinMaxAxis(self):
        rect = cv2.minAreaRect(self.pixel_contour)
        box = cv2.cv.BoxPoints(rect)
        box = np.int32(box)
        mbox = self.projectBox(box)
        vec1 = mbox[1] - mbox[2]
        vec2 = mbox[2] - mbox[3]
        axis2 = (np.linalg.norm(vec1), np.linalg.norm(vec2))
        return min(axis2), max(axis2)

    @staticmethod
    def get_distances(points):
        circular_buffer = cycle(points)
        previous_point = circular_buffer.next()

        for i in range(len(points)):
            point = circular_buffer.next()
            yield MetricContour.get_distance(previous_point, point)
            previous_point = point

    @staticmethod
    def get_distance(point1, point2):
        a = point1[0] - point2[0]
        b = point1[1] - point2[1]
        return math.sqrt(a ** 2 + b ** 2)

    @staticmethod
    def PolyArea(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class MerticContourList:
    def __init__(self, pixel_contours, depth, cam_params):
        self.pixel_contours = pixel_contours
        self.metric_contours = []
        self.depth = depth
        self.cam_params = cam_params
        self.project_pixel_contours(depth)

    def project_pixel_contours(self, depth):
        if len(self.metric_contours) > 0:
            for cont in self.metric_contours:
                cont.reprojectContour(depth)
        else:
            for cont in self.pixel_contours:
                mcont = MetricContour(cont, depth, self.cam_params)
                self.metric_contours.append(mcont)

    def filter_by_area_metric(self, min_area, max_area):
        enum = reversed(list(enumerate(self.metric_contours)))
        for idx, cont in enum:
            area = cont.getMetricArea()
            if area < min_area or area > max_area:
                self.pixel_contours.pop(idx)
                self.metric_contours.pop(idx)

    def filter_by_perimeter_metric(self, min_perim, max_perim):
        enum = reversed(list(enumerate(self.metric_contours)))
        for idx, cont in enum:
            area = cont.getPerimeter()
            if area < min_perim or area > max_perim:
                self.pixel_contours.pop(idx)
                self.metric_contours.pop(idx)

    def filter_by_aspect_ratio_metric(self, maxratio):
        enum = reversed(list(enumerate(self.metric_contours)))
        for idx, cont in enum:
            min_axis, max_axis = cont.getMinMaxAxis()
            aspect_ratio = max_axis / min_axis
            if aspect_ratio > maxratio:
                self.pixel_contours.pop(idx)
                self.metric_contours.pop(idx)

    def filter_by_aspect_ratio_pixel(self, maxratio):
        enum = reversed(list(enumerate(self.pixel_contours)))
        for idx, cont in enum:
            rect = cv2.minAreaRect(cont)
            box = cv2.cv.BoxPoints(rect)
            box = np.int32(box)
            vec1 = box[1] - box[2]
            vec2 = box[2] - box[3]
            axis2 = (np.linalg.norm(vec1), np.linalg.norm(vec2))
            if not min(axis2) == 0:
                aspect_ratio = max(axis2) / min(axis2)
            else:
                aspect_ratio = 0
            if aspect_ratio > maxratio:
                self.pixel_contours.pop(idx)
                self.metric_contours.pop(idx)

    def metric_centre_dist(self, pix1, pix2):
        X1 = (pix1[0] - self.cam_params[0]) * (self.depth / self.cam_params[2])
        Y1 = (pix1[1] - self.cam_params[1]) * (self.depth / self.cam_params[3])
        X2 = (pix2[0] - self.cam_params[0]) * (self.depth / self.cam_params[2])
        Y2 = (pix2[1] - self.cam_params[1]) * (self.depth / self.cam_params[3])
        return np.linalg.norm(np.array([X1,Y1]) - np.array([X2,Y2]))

    def merge_contours(self, input_contours, dist_between=0.01, max_area_merge=0.4):
        non_merged = False
        centers = []
        res_conts = []
        merge_conts = []
        for idx, cnt in enumerate(input_contours):
            try:
                hull = cv2.convexHull(cnt)
            except Exception as e:
                print e
                continue
            M = cv2.moments(hull)
            hull_m = MetricContour(hull, self.depth, self.cam_params)
            if hull_m.getMetricArea() > max_area_merge:
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                aprroxc = cv2.approxPolyDP(cnt, epsilon, True)
                res_conts.append(aprroxc)
                continue

            if M['m00'] == 0.0:
                if len(cnt) > 4:
                    ellipse = cv2.fitEllipse(cnt)
                    center = np.int32(ellipse[0])
                    print 'fitted ellipse'
                else:
                    continue
            else:
                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])
                center = np.array([centroid_x, centroid_y])
            centers.append(center)
            merge_conts.append(cnt)
        # print len(res_conts)
        # print len(centers)
        # ret  = [x for x in centers]
        while not non_merged:
            non_merged = True
            # rem_x = None
            # rem_y = None
            idx = 0
            idy = 0
            for idx, cen1 in enumerate(centers):
                for idy, cen2 in enumerate(centers):
                    # print idy, "test"
                    if idy == idx:
                        continue

                    dist = self.metric_centre_dist(cen1, cen2)
                    #print dist#, pixel_to_m(dist)
                    #print "\n"
                    if abs(dist) < dist_between:
                        non_merged = False

                        # print "Found one", idx, idy, dist
                        # rem_x = idx
                        # rem_y = idy
                        break
                if not non_merged:
                    break
            if non_merged:
                continue
            # print 'merge', idx, idy
            merged = np.append(merge_conts[idx], merge_conts[idy], axis=0)
            # print test

            if idy > idx:
                centers.pop(idy)
                centers.pop(idx)
                merge_conts.pop(idy)
                merge_conts.pop(idx)
            else:
                centers.pop(idx)
                centers.pop(idy)
                merge_conts.pop(idy)
                merge_conts.pop(idx)

            try:
                cnt = cv2.convexHull(merged)
            except Exception as e:
                print e
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                print 'merged has no area'
                continue

            merge_conts.append(cnt)

            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            center = np.array([centroid_x, centroid_y])
            centers.append(center)

        for cnt in merge_conts:
            res_conts.append(cnt)

        return res_conts, centers

def pre_process_depth(depth_img, visualization_level):
    if visualization_level > 1:
        input_depth = depth_img.copy()
        cv2.imshow("input", input_depth)
        cv2.waitKey(0)

    # filter NaN's and Infs
    idx = np.isnan(depth_img)
    depth_img[idx] = 0.0
    idx = np.isinf(depth_img)
    depth_img[idx] = 1.2

    # compute median depth
    median_depth = np.median(depth_img)

    # find indexes of pixels below 0.2 and above 1.0
    low_ids = depth_img < 0.2
    high_ids = depth_img > 1.0

    # set the low indexes to median depth - 0.02
    depth_img[low_ids] = median_depth - 0.02
    # set the high indexes to 1.2
    depth_img[high_ids] = 1.2

    # normalize to between 0 and 255 and convert to unit8
    cv2.normalize(depth_img, depth_img, 0, 255, cv2.NORM_MINMAX)
    depth_img = np.uint8(depth_img)
    
    # apply blur to remove noise either median blur ot gaussian
    depth_img = cv2.medianBlur(depth_img, 5)
    # depth_img = cv2.GaussianBlur(depth_img, (5, 5), 0)
    depth_blurred = depth_img.copy()

    if visualization_level > 1:
        cv2.imshow("filtered and blurred", depth_img)
        cv2.waitKey(0)

    # apply and adaptive filter to the blurred image
    frame_filter = cv2.adaptiveThreshold(depth_img,
                                         255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         # cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         15,  # neighbourhood
                                         2)

    # invert Colors
    cv2.bitwise_not(frame_filter, frame_filter)

    if visualization_level > 1:
        cv2.imshow("post adaptive filter", frame_filter)
        cv2.waitKey(0)

    # Dilate to merge shapes
    kernel = np.ones((7, 7), 'uint8')
    frame_filter = cv2.dilate(frame_filter, kernel)

    kernel = np.ones((3, 3), 'uint8')
    frame_filter = cv2.erode(frame_filter, kernel)

    kernel = np.ones((3,3), 'uint8')
    frame_filter = cv2.morphologyEx(frame_filter, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5,5), 'uint8')
    frame_filter = cv2.morphologyEx(frame_filter, cv2.MORPH_CLOSE, kernel)

    # Dilate to merge shapes
    #kernel_dilate = np.ones((15, 15), 'uint8')
    kernel_dilate2 = np.ones((15, 15), 'uint8')

    #frame_filter = cv2.dilate(frame_filter, kernel_dilate2)

    if visualization_level > 1:
        cv2.imshow("threshold image", frame_filter)
        cv2.waitKey(0)

    return median_depth, depth_blurred, frame_filter

def process_rgb(rgb_img):

    frame_gray = cv2.cvtColor(rgb_img, cv.CV_RGB2GRAY)
    gray_blurred = cv2.medianBlur(frame_gray, 5)

    gray_filter = cv2.adaptiveThreshold(gray_blurred,
                                        255.0,
                                        cv.CV_ADAPTIVE_THRESH_MEAN_C,
                                        cv.CV_THRESH_BINARY,
                                        31,  # neighbourhood, was 15 # sort of line thickness
                                        4)
    cv2.bitwise_not(gray_filter, gray_filter)

    #cv2.imshow("rgbplusdepth", gray_filter)
    #cv2.waitKey(0)
    
    gray_dilate = gray_filter

    kernel_morph = np.ones((10, 10), 'uint8')
    gray_dilate = cv2.morphologyEx(gray_dilate, cv2.MORPH_CLOSE, kernel_morph)
    
    #cv2.imshow("rgbplusdepth", gray_dilate)
    #cv2.waitKey(0)

    kernel = np.ones((8, 8), 'uint8') # was 24x24 - too blown up
    gray_dilate = cv2.dilate(gray_dilate, kernel)
    
    #cv2.imshow("rgbplusdepth", gray_dilate)
    #cv2.waitKey(0)
    
    kernel = np.ones((8, 8), 'uint8') # was 24x24 - too blown up
    gray_dilate = cv2.erode(gray_dilate, kernel)
    
    #cv2.imshow("rgbplusdepth", gray_dilate)
    #cv2.waitKey(0)
    
    return gray_dilate

def extract_contours_m(depth_image, input_image, media_depth, cam_params, cross_w, cross_h, visualization_level):
        size = depth_image.shape
        size = (size[1] - 1, size[0] - 1)

        cv2.rectangle(depth_image, (0, 0), size,
                      0,  # color
                      20,  # thickness
                      8,  # line-type ???
                      0)  # random shit

        contours, hierarchy = cv2.findContours(depth_image,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

        approx = []
        for cont in contours:
            if(len(cont)>2):
                approx.append(cv2.convexHull(cont))

        contour_list = MerticContourList(approx, media_depth, cam_params)

        if visualization_level > 1:
            raw_img = input_image.copy()
            cv2.drawContours(raw_img, contours, -1, (255, 0, 0), 3)
            cv2.imshow("raw contours", raw_img)
            cv2.waitKey(0)

        contour_list.filter_by_aspect_ratio_pixel(5)

        if visualization_level > 1:
            raw_img = input_image.copy()
            cv2.drawContours(raw_img, contour_list.pixel_contours, -1, (255, 0, 0), 3)
            cv2.imshow("aspect filtered contours", raw_img)
            cv2.waitKey(0)

            contour_list.filter_by_area_metric(vision_constants.CONTOUR_MIN_AREA, vision_constants.CONTOUR_MAX_AREA)

        if visualization_level > 1:
            raw_img = input_image.copy()
            cv2.drawContours(raw_img, contour_list.pixel_contours, -1, (255, 0, 0), 3)
            cv2.imshow("area filtered contours", raw_img)
            cv2.waitKey(0)

        return contour_list
        
def preproces_rgb_and_detect_contours(rgb_image, depth_image, contour_list, visualization_level, container_mode=False):
    # convert image to greyscale
    frame_gray = cv2.cvtColor(rgb_image, cv.CV_RGB2GRAY)

    gray_blurred = cv2.medianBlur(frame_gray, 19)

    # apply bilateral filter
    gray_blurred = cv2.bilateralFilter(gray_blurred, 8, 16, 4)

    redetected_contours = []
    counter2 = 0
    for idx, cont in enumerate(contour_list.pixel_contours):
        leftmost = tuple(cont[cont[:, :, 0].argmin()][0])
        rightmost = tuple(cont[cont[:, :, 0].argmax()][0])
        topmost = tuple(cont[cont[:, :, 1].argmin()][0])
        bottommost = tuple(cont[cont[:, :, 1].argmax()][0])
        eps = 5
        #print leftmost, rightmost, topmost, bottommost
        # get object from gray image with smaller mask
        object_image = frame_gray[topmost[1] - eps:bottommost[1] + eps, leftmost[0] - eps:rightmost[0] + eps].copy()

        # create the object mask
        topleft = np.array([leftmost[0]-eps, topmost[1] - eps])
        #sc = cont.copy()
        for p in cont:
            p -= topleft
        sc = cv2.convexHull(cont)
        mask = np.zeros(object_image.shape).astype(object_image.dtype)
        cv2.fillPoly(mask, [sc], [255, 255, 255])

        kernel_erode = np.ones((15, 15), 'uint8')
        mask_erode = cv2.erode(mask, kernel_erode)

        mean_int = cv2.mean(object_image, mask_erode)
        mcont = contour_list.metric_contours[idx]
        area = mcont.getMetricArea()
        if mean_int[0] < 80 or (mean_int[0] < 90 and area < 0.0025): #130

            _,frame_bin = cv2.threshold(object_image,30,255,cv2.THRESH_BINARY)

            frame_filter = cv2.adaptiveThreshold(object_image,
                                     255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     #cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     9,  # neighbourhood
                                     15)

            kernel_morph = np.ones((3, 3), 'uint8')
            frame_filter = cv2.morphologyEx(frame_filter, cv2.MORPH_OPEN, kernel_morph, iterations=2)
            frame_filter = cv2.bitwise_and(frame_filter, frame_bin)
            if visualization_level > 8:
                #cv2.drawContours(frame_filter, [conts[idx]], 0, (255, 255, 255), 2)
                string = 'object' + str(counter2)
                counter2 += 1
                cv2.imshow(string, frame_filter)

        elif mean_int[0] > 90 and area < 0.003:
                object_image = cv2.bitwise_and(mask, object_image)
                # apply and adaptive filter to the blurred image
                frame_filter = cv2.adaptiveThreshold(object_image,
                                                     255,
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     #cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     cv2.THRESH_BINARY,
                                                     15,  # neighbourhood
                                                     6)

                object_image_depth = depth_image[topmost[1] - eps:bottommost[1] + eps,
                                     leftmost[0] - eps:rightmost[0] + eps].copy()
                frame_filter_depth = cv2.adaptiveThreshold(object_image_depth,
                                                     255,
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     #cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     cv2.THRESH_BINARY,
                                                     15,  # neighbourhood
                                                     2)
                frame_filter = cv2.bitwise_and(frame_filter, frame_filter_depth)
                if visualization_level > 8:
                    #cv2.drawContours(frame_filter, [conts[idx]], 0, (255, 255, 255), 2)
                    string = 'object' + str(counter2)
                    counter2 += 1
                    cv2.imshow(string, frame_filter)

        else:
            object_image = cv2.bitwise_and(mask, object_image)
            # apply and adaptive filter to the blurred image
            frame_filter = cv2.adaptiveThreshold(object_image,
                                                 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 #cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY,
                                                 15,  # neighbourhood
                                                 6)

            if mcont.getMetricArea() > 0.002:
                object_image_depth = depth_image[topmost[1] - eps:bottommost[1] + eps,
                                     leftmost[0] - eps:rightmost[0] + eps].copy()
                frame_filter_depth = cv2.adaptiveThreshold(object_image_depth,
                                                     255,
                                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     #cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     cv2.THRESH_BINARY,
                                                     15,  # neighbourhood
                                                     2)
                #cv2.bitwise_not(frame_filter_depth, frame_filter_depth)
                frame_filter = cv2.bitwise_and(frame_filter, frame_filter_depth)


            if visualization_level > 8:
                #cv2.drawContours(frame_filter, [conts[idx]], 0, (255, 255, 255), 2)
                string = 'object' + str(counter2)
                counter2 += 1
                cv2.imshow(string, frame_filter)


        # invert Colors
        cv2.bitwise_not(frame_filter, frame_filter)

        frame_filter = cv2.bitwise_and(frame_filter, mask)

        # Dilate to merge shapes
        kernel_morph = np.ones((15, 15), 'uint8')
        frame_filter = cv2.morphologyEx(frame_filter, cv2.MORPH_CLOSE, kernel_morph)

        #frame_filter_otsu = cv2.morphologyEx(frame_filter_otsu, cv2.MORPH_OPEN, kernel_morph)
        kernel_dilate = np.ones((3, 3), 'uint8')
        frame_filter = cv2.dilate(frame_filter, kernel_dilate)

        size = frame_filter.shape
        size = (size[1] - 1, size[0] - 1)

        # rectangle to get correctly order contours
        cv2.rectangle(frame_filter, (0, 0), size,
                      0,  # color
                      20,  # thickness
                      8,  # line-type ???
                      0)  # random shit

        ##Find second contour in smaller image
        conts, _ = cv2.findContours(frame_filter,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        if len(conts) > 0:
            ### If more than one contour take largest
            idx = 0
            if not len(conts) == 1:
                min_area = 0
                for i, cont in enumerate(conts):
                    try:
                        h = cv2.convexHull(cont)
                    except Exception as e:
                        print e
                        continue
                    area_tmp = cv2.contourArea(h)
                    if area_tmp > min_area:
                        min_area = area_tmp
                        idx = i

            ### Crop to full image
            topleft = np.array([leftmost[0] - eps, topmost[1] - eps])
            if visualization_level > 2:
                cv2.drawContours(frame_filter, [conts[idx]], 0, (255, 255, 255), 2)
                string = 'object' + str(counter2)
                counter2 += 1
                cv2.imshow(string, frame_filter)

            for p in conts[idx]:
                p += topleft

            c = conts[idx]
            if len(c) > 2:
              redetected_contours.append(c)
            
    return redetected_contours
