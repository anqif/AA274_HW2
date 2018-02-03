#!/usr/bin/python

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from PlotFunctions import *


############################################################
# functions
############################################################

#-----------------------------------------------------------
# ExtractLines
#
# This function implements a split-and-merge line
# extraction algorithm
#
# INPUT: RangeData - (x_r, y_r, theta, rho)
#                x_r - robot's x position (m)
#                y_r - robot's y position (m)
#              theta - (1D) np array of angle 'theta' from data (rads)
#                rho - (1D) np array of distance 'rho' from data (m)
#           params - dictionary of parameters for line extraction
#
# OUTPUT: (alpha, r, segend, pointIdx)
#         alpha - (1D) np array of 'alpha' for each fitted line (rads)
#             r - (1D) np array of 'r' for each fitted line (m)
#        segend - np array (N_lines, 4) of line segment endpoints.
#                 each row represents [x1, y1, x2, y2]
#      pointIdx - (N_lines,2) segment's first and last point index

def ExtractLines(RangeData, params):

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=np.int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    for endIdx in LineBreak:
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, rho, startIdx, endIdx, params)
        N_lines = r_seg.size

        ### Merge Lines ###
        if (N_lines > 1):
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx


#-----------------------------------------------------------
# SplitLineRecursive
#
# This function executes a recursive line-splitting algorithm,
# which recursively sub-divides line segments until no further
# splitting is required.
#
# INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
#           rho - (1D) np array of distance 'rho' from data (m)
#      startIdx - starting index of segment to be split
#        endIdx - ending index of segment to be split
#        params - dictionary of parameters
#
# OUTPUT: alpha - (1D) np array of 'alpha' for each fitted line (rads)
#             r - (1D) np array of 'r' for each fitted line (m)
#           idx - (N_lines,2) segment's first and last point index

def SplitLinesRecursive(theta, rho, startIdx, endIdx, params):

    ##### TO DO #####
    # Implement a recursive line splitting function
    # It should call 'FitLine()' to fit individual line segments
    # It should call 'FindSplit()' to find an index to split at
    #################
    theta_s = theta[startIdx:endIdx]
    rho_s = rho[startIdx:endIdx]

    # No more splits possible if single point
    if startIdx == endIdx:
        return np.empty((0)), np.empty((0)), np.empty((0,2))

    # Attempt to split the fitted line segment
    alpha, r = FitLine(theta_s, rho_s)
    splitIdx_s = FindSplit(theta_s, rho_s, alpha, r, params)
    if splitIdx_s == -1:
        idx = np.array([[startIdx, endIdx]])
    else:
        # Fit line segments to each of the split data sets
        splitIdx = splitIdx_s + startIdx
        alpha_l, r_l, idx_l = SplitLinesRecursive(theta, rho, startIdx, splitIdx, params)
        alpha_r, r_r, idx_r = SplitLinesRecursive(theta, rho, splitIdx, endIdx, params)

        alpha = np.hstack((alpha_l, alpha_r))
        r = np.hstack((r_l, r_r))
        idx = np.vstack((idx_l, idx_r))
    return alpha, r, idx


#-----------------------------------------------------------
# FindSplit
#
# This function takes in a line segment and outputs the best
# index at which to split the segment
#
# INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
#           rho - (1D) np array of distance 'rho' from data (m)
#         alpha - 'alpha' of input line segment (1 number)
#             r - 'r' of input line segment (1 number)
#        params - dictionary of parameters
#
# OUTPUT: SplitIdx - idx at which to split line (return -1 if
#                    it cannot be split)

def FindSplit(theta, rho, alpha, r, params):

    ##### TO DO #####
    # Implement a function to find the split index (if one exists)
    # It should compute the distance of each point to the line.
    # The index to split at is the one with the maximum distance
    # value that exceeds 'LINE_POINT_DIST_THRESHOLD', and also does
    # not divide into segments smaller than 'MIN_POINTS_PER_SEGMENT'
    # return -1 if no split is possiple
    #################
    # Calculate distance from line and apply threshold filters
    n = len(theta)
    num_pts = np.array(range(0,n))   # Assume split point assigned to left segment
    d = np.abs(np.multiply(rho, np.cos(theta-alpha)) - r)
    filt = (d > params["LINE_POINT_DIST_THRESHOLD"]) & \
           (num_pts >= params["MIN_POINTS_PER_SEGMENT"]) & \
           ((n-num_pts) >= params["MIN_POINTS_PER_SEGMENT"])

    # Choose largest distance that satisfies thresholds
    if np.any(filt):
        org_to_max = np.argsort(d)
        idx = np.where(filt[org_to_max])[0][-1]  # Max element is last since sorted in ascending order
        splitIdx = org_to_max[idx]   # Map back to unsorted data index
    else:
        splitIdx = -1
    return splitIdx


#-----------------------------------------------------------
# FitLine
#
# This function outputs a best fit line to a segment of range
# data, expressed in polar form (alpha, r)
#
# INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
#           rho - (1D) np array of distance 'rho' from data (m)
#
# OUTPUT: alpha - 'alpha' of best fit for range data (1 number) (rads)
#             r - 'r' of best fit for range data (1 number) (m)

def FitLine(theta, rho):

    ##### TO DO #####
    # Implement a function to fit a line to polar data points
    # based on the solution to the least squares problem (see Hw)
    #################
    n = len(theta)
    alpha_num = 0.0
    alpha_den = 0.0
    for i in range(0, n):
        for j in range(0, n):
            alpha_num += rho[i]*rho[j]*np.cos(theta[i])*np.sin(theta[j])
            alpha_den += rho[i]*rho[j]*np.cos(theta[i] + theta[j])
    alpha_num = np.sum(np.multiply(rho**2, np.sin(2.0*theta))) - (2.0/n)*alpha_num
    alpha_den = np.sum(np.multiply(rho**2, np.cos(2.0*theta))) - (1.0/n)*alpha_den

    alpha = 0.5*np.arctan2(alpha_num, alpha_den) + np.pi/2
    r = (1.0/n)*np.sum(np.multiply(rho, np.cos(theta-alpha)))
    return alpha, r


#---------------------------------------------------------------------
# MergeColinearNeigbors
#
# This function merges neighboring segments that are colinear and outputs
# a new set of line segments
#
# INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
#           rho - (1D) np array of distance 'rho' from data (m)
#         alpha - (1D) np array of 'alpha' for each fitted line (rads)
#             r - (1D) np array of 'r' for each fitted line (m)
#      pointIdx - (N_lines,2) segment's first and last point indices
#        params - dictionary of parameters
#
# OUTPUT: alphaOut - output 'alpha' of merged lines (rads)
#             rOut - output 'r' of merged lines (m)
#      pointIdxOut - output start and end indices of merged line segments

def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):

    ##### TO DO #####
    # Implement a function to merge colinear neighboring line segments
    # HINT: loop through line segments and try to fit a line to data
    #       points from two adjacent segments. If this line cannot be
    #       split, then accept the merge. If it can be split, do not merge.
    #################
    alphaOut = alpha.copy()
    rOut = r.copy()
    pointIdxOut = pointIdx.copy()

    i = 0
    N_lines = pointIdxOut.shape[0]
    while i < (N_lines - 1):
        startIdx = pointIdx[i,0]   # Start of first segment
        endIdx = pointIdx[i+1,1]   # End of next segment
    
        # Fit line to data from adjacent segments and attempt to split
        alpha_fit, r_fit = FitLine(theta[startIdx:endIdx], rho[startIdx:endIdx])
        splitIdx = FindSplit(theta[startIdx:endIdx], rho[startIdx:endIdx], alpha_fit, r_fit, params)

        # Accept merge if fitted line cannot be split
        if splitIdx == -1:
            # Update with new segment
            alphaOut[i] = alpha_fit
            rOut[i] = r_fit
            pointIdxOut[i,1] = endIdx
        
            # Delete old segment that was merged
            alphaOut = np.delete(alphaOut, i+1)
            rOut = np.delete(rOut, i+1)
            pointIdxOut = np.delete(pointIdxOut, i+1, axis=0)
            N_lines = N_lines - 1
        else:
            i = i + 1

    return alphaOut, rOut, pointIdxOut


#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.05  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = 0.025  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 3  # minimum number of points per line segment
    MAX_P2P_DIST = 0.54  # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    filename = 'rangeData_5_5_180.csv'   # Best params = [0.05, 0.065, 2, 0.80]
    # filename = 'rangeData_4_9_360.csv'     # Best params = [0.05, 0.025, 3, 0.54]
    # filename = 'rangeData_7_2_90.csv'    # Best params = [0.05, 0.165, 2, 0.42]

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    plt.show(ax)

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
