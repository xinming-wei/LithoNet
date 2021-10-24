# Transform a 1-channel nparrray mask to GDS file
# Mask: 1-channel nparray, mostly binary(0,255) with OPC and SRAF
# GDS file: {layer, datatype}
#* Note that origL, OPC and SRAF can be put in one single layer(whatever the layer name is)
#* All of these are defined in .slo file
## {0,0} orig mask, OPC and SRAF

import numpy as np
import cv2
import gdspy
import sys
import os

class Array2Gds:
    
    def __init__(self, mask_array, orig_layout):
        """
        mask_array, orig_layout are both nparrays
        with the same size(e.g. 250x250)
        """
        self.mask_array = mask_array
        self.orig_layout = orig_layout
        assert len(mask_array.shape) == 2 and len(orig_layout.shape) == 2, "Mask and Layout should be 2-dim"
        assert mask_array.shape[1] == orig_layout.shape[1], "Mask and Layout should have the same size!"
        self.width = mask_array.shape[1]
        self.gdsUnit = 1e-3   # GDSII file unit in um
        self.cell = "TOP"
        self.mask_gds = None
        self.orig_rects = []
        self.opc_and_sraf_polygons = []
        self.layer = 0
        self.datatype = 0
        # self.index = i

    def find_OrigRects(self):
        orig_layout = np.array(self.orig_layout, dtype=np.uint8)
        contours, _ = cv2.findContours(orig_layout, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            rect = [x, y, x+w, y+h]
            rect = [r / self.width * 2 for r in rect]
            self.orig_rects.append(rect)

    def find_OPC_SRAF_polygons(self):
        mask_array = np.array(self.mask_array, dtype=np.uint8)
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # test_img = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)
        # approxes = []
        for cntr in contours:
            # x,y,w,h = cv2.boundingRect(cntr)
            # Filter redundent features
            # if cv2.contourArea(cntr) <= 4.0:
            #     continue
            eps = 0.01 * cv2.arcLength(cntr, True)
            approx = cv2.approxPolyDP(cntr, eps, True)
            r = 2 / self.width
            points = [(point[0][0]*r, point[0][1]*r) for point in approx]
            # approxes.append(approx)
            # rect = [x, y, x+w, y+h]
            # rect = [r / self.width * 2 for r in rect]
            self.opc_and_sraf_polygons.append(points)
        # cv2.polylines(test_img, approxes, True, (0,0,255), 1)
        # cv2.imwrite('%04d.jpg' % self.index, test_img)            
    
    def convert(self, gdsPath):
        self.find_OrigRects()
        self.find_OPC_SRAF_polygons()
        gdsii = gdspy.GdsLibrary()
        #* Must overwrite current_lirary, otherwise we cannot add new cell with the same name 
        gdspy.current_library = gdsii 
        cell = gdsii.new_cell(self.cell)

        for rect in self.orig_rects:
            cell.add(gdspy.Rectangle(rect[:2], rect[2:], layer=self.layer, datatype=self.datatype))
        for polygon_points in self.opc_and_sraf_polygons:
            cell.add(gdspy.Polygon(polygon_points, layer=self.layer, datatype=self.datatype))
        
        gdsii.write_gds(gdsPath)
        
                

        