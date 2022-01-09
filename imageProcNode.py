#!/usr/bin/env python
import sys
sys.path.append('.')

import socket
import struct
import io
import cv2
import numpy as np
import time
import datetime
import threading
import multiprocessing
import statistics
import math
import numpy.ma as ma

import rospy
import ros
import std_msgs.msg

from std_msgs.msg import Int32, Float32
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from geometry_msgs.msg import Point, Point32
from sensor_msgs.msg import PointCloud

from threading import Thread
from multiprocessing import Pipe, Process, Event

from src.utils.templates.workerprocess import WorkerProcess
#from src.utils.templates.workerprocess import WorkerProcess
from src.hardware.camera.cameraprocess               import CameraProcess
from src.hardware.serialhandler.serialhandler        import SerialHandler

from src.utils.camerastreamer.camerastreamer       import CameraStreamer
from src.utils.cameraspoofer.cameraspooferprocess  import CameraSpooferProcess
from src.utils.remotecontrol.remotecontrolreceiver import RemoteControlReceiver
from numpy.polynomial import polynomial as P

# SWITCHES
DEBUG_MODE = 1
DRAW_MODE = 1

# PARAMETERS, DEFINES, MACROS
ROADSTATE_STRAIGHT = 10
ROADSTATE_NO_LEFTLANE = 11
ROADSTATE_NO_RIGHTLANE = 12
ROADSTATE_CURVE = 20
ROADSTATE_THRESHOLD = 10

TSDSTATE_NONE = 0
TSDSTATE_STOP = 1
TSDSTATE_MAIN = 2
TSDSTATE_PEDASTRIAN = 3
TSDSTATE_PARK = 4

LD_WHITE_PERCENTAGE = 10.0
#LD_WHITE_SAT_TH_VALS = np.linspace(0, 255, 8)
LD_WHITE_SAT_TH_VALS = [30, 90, 160]

HISTARRAY = np.full_like(LD_WHITE_SAT_TH_VALS, 0) 

class ImageProcessing(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self, inPs, outPs):
        ### Reading the parameters
        # general
        self.rate = float(rospy.get_param('~rate', '30.0'))
        # TSD node
        self.trafficSignDetStatusTopicName = rospy.get_param('~trafficSignDetStatusTopicName', 'trafficSignDetStatus')
        # LD node
        self.lateralErrorTopicName = rospy.get_param('~lateralErrorTopicName', 'errorLateral')
        self.headingErrorTopicName = rospy.get_param('~headingErrorTopicName', 'errorHeading')
        self.intersectionStateTopicName = rospy.get_param('~intersectionStateTopicName', 'intersectionState')
        self.roadStateTopicName = rospy.get_param('~roadStateTopicName', 'roadState')
        
        ### Publisher messages
        # TSD node
        self.trafficSignDetStatus = Int32()

        # LD node
        self.lateralErrorMsg = Int32()
        self.headingErrorMsg = Float32()
        self.roadStateMsg = Int32()
        self.intersectionStateMsg = Int32()
        self.intersectionStateMsg.data = 0
        ### Create publihers & subscribers
        # TSD node
        self.trafficSignDetStatusPub = rospy.Publisher(self.trafficSignDetStatusTopicName, Int32, queue_size=2)
        # LD node
        self.lateralErrorPub = rospy.Publisher(self.lateralErrorTopicName, Int32, queue_size=2)
        self.headingErrorPub = rospy.Publisher(self.headingErrorTopicName, Float32, queue_size=2)
        self.roadStateMsgPub = rospy.Publisher(self.roadStateTopicName, Int32, queue_size=2)
        self.intersectionStatePub = rospy.Publisher(self.intersectionStateTopicName, Int32, queue_size=2)
        

        ### Private variables
        ## LD Node

        
        # trackbar var's:
        # [0-Yellow_TH_low, 1-Yellow_TH_high, 2-Red_TH_low, 3-Red_TH_high, 4-Blue_TH_low, 5-Blue_TH_high, 6-White_TH_sat_low, 7-White_TH_Sat_high,
        #  8-SaturationLow, 9-ValueLow, 10-Canny_Low, 11-Canny_High, 12-polyArcLenghtParam, 13-whiteSatTh
        # 14 - preproc Value lower, 15 - preproc Value upper]
        # default values: 

        self.Yellow_TH_low = 28
        self.Yellow_TH_high = 43
        self.Yellow_Sat_low = 54
        self.Yellow_Sat_high = 158
        self.Yellow_Value_low = 91
        self.Yellow_Value_high = 255
        
        self.Red_TH_low = 0
        self.Red_TH_high = 121
        self.Red_Sat_low = 61
        self.Red_Sat_high = 225
        self.Red_Value_low = 60
        self.Red_Value_high = 155
        
        self.Blue_TH_low = 87
        self.Blue_TH_high = 122
        self.Blue_Sat_low = 118
        self.Blue_Sat_high = 255
        self.Blue_Value_low = 15
        self.Blue_Value_high = 200
        
        self.CannyWhiteTH_Low = 128
        self.CannyWhiteTH_High = 255
        self.CannyParamLow = 80
        self.CannyParamHigh = 255
        self.polyArcLengthParam = 3

        #self.trackbarValues = [0, 35, 140, 180, 95, 130, 130, 255, 90, 75, 80, 255, 2]
        self.trackbarValues = [
            self.Yellow_TH_low, # 0
            self.Yellow_TH_high, # 1
            self.Yellow_Sat_low, # 2 
            self.Yellow_Sat_high, # 3
            self.Yellow_Value_low, # 4
            self.Yellow_Value_high, # 5
            self.Red_TH_low, # 6...
            self.Red_TH_high,
            self.Red_Sat_low,
            self.Red_Sat_high,
            self.Red_Value_low,
            self.Red_Value_high,            
            self.Blue_TH_low,
            self.Blue_TH_high,
            self.Blue_Sat_low,
            self.Blue_Sat_high,
            self.Blue_Value_low,
            self.Blue_Value_high,            
            self.CannyWhiteTH_Low,
            self.CannyWhiteTH_High,
            self.CannyParamLow,
            self.CannyParamHigh,
            self.polyArcLengthParam
        ]
        rospy.Subscriber("trackBarValues", Int32MultiArray, self.LD_trackbarCallback)

        # Preprocessing white filter limits
        self.lowerWhiteLimit = np.array([0, 0, self.CannyWhiteTH_Low])
        self.upperWhiteLimit = np.array([180, 100, self.CannyWhiteTH_High])

        # Region of interest img: 640 x 480
        # lower rectangle
        self.LD_roiTopLeftRect = (0, 360)        #x-y val
        self.LD_roiTopRightRect = (640, 360)        #x-y val
        self.LD_roiBottomLeftRect = (0, 480)        #x-y val
        self.LD_roiBottomRightRect = (640, 480)        #x-y val
        #upper trapeze
        self.LD_roiTopLeftTrapeze = (80, 240)        #x-y val
        self.LD_roiTopRightTrapeze = (560, 240)        #x-y val
        self.LD_roiBottomLeftTrapeze = (0, 360)        #x-y val
        self.LD_roiBottomRightTrapeze = (640, 360)        #x-y val
        roiMaskBox = self.MakeRoiMaskImage(self.LD_roiTopLeftRect, self.LD_roiTopRightRect, self.LD_roiBottomLeftRect, self.LD_roiBottomRightRect)
        roiMaskTrapeze = self.MakeRoiMaskImage(self.LD_roiTopLeftTrapeze, self.LD_roiTopRightTrapeze, self.LD_roiBottomLeftTrapeze, self.LD_roiBottomRightTrapeze)
        self.LD_roiMask = cv2.bitwise_or(roiMaskBox, roiMaskTrapeze)
        

        self.LD_roiMaskCarTopLeft = (140, 440)
        self.LD_roiMaskCarTopRight = (500, 440)
        self.LD_roiMaskCarBottomLeft = (140, 480)
        self.LD_roiMaskCarBottomRight = (500, 480)
        self.LD_roiMaskCar = self.MakeRoiMaskImage(self.LD_roiMaskCarTopLeft, self.LD_roiMaskCarTopRight, self.LD_roiMaskCarBottomLeft, self.LD_roiMaskCarBottomRight)

        #Intersection exit
        self.LD_IntersectionLatch = 0
        self.LD_ExitIntersection = 0
        self.LD_IntersectionTimeStamp = 0

        # others
        self.leftLine = 0
        self.rightLine = 640
        self.middleLine = 320
        self.middleLinePrev = 320
        self.alfaExp = 70 / 100
        self.windowsNumber = 5
        
        self.leftWindowLine =  np.full((self.windowsNumber, 1), 0)
        self.rightWindowLine = np.full((self.windowsNumber, 1), 640)
        self.middleWindowLine = np.full((self.windowsNumber, 1), 320)
        ## TSD node
        self.plausabilityPedastrian = 0
        self.plausabilityMain = 0
        self.plausabilityPark = 0
        self.plausabilityStop = 0
         # Region of interest
        self.TSD_roiTopLeft = (430, 60)        #x-y val
        self.TSD_roiTopRight = (640, 60)        #x-y val
        self.TSD_roiBottomLeft = (430, 240)        #x-y val
        self.TSD_roiBottomRight = (640, 240)        #x-y val
        self.TSD_roiMask = self.MakeRoiMaskImage(self.TSD_roiTopLeft, self.TSD_roiTopRight, self.TSD_roiBottomLeft, self.TSD_roiBottomRight)


        super(ImageProcessing,self).__init__( inPs, outPs)



         # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        rospy.loginfo("ImageProcessing is running")
        r = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            # general
            start_time = time.time()
            stamp, image = self.inPs[0].recv()
            
            image2 = np.copy(image)
            # cv2.imshow("image", image2)
            # cv2.waitKey(0)
            # LD node
            LD_imageWithAllLine, LD_image, warped_image = self.LD_run(image)
            # print("time diff:" + str(time.time()-start_time))
            #sending out the pictures
            #self.outPs[0].send([[stamp], image])
            self.outPs[0].send([[stamp], LD_image])
            self.outPs[1].send([[stamp], LD_imageWithAllLine])
            # #self.outPs[2].send([[stamp], image])
            # # TSD node
            temp, ProcedImage = self.TSD_run(image2)
            # # #sending out the pictures
            self.outPs[2].send([[stamp], warped_image])
            self.outPs[3].send([[stamp], temp])
            r.sleep()




############################ LANE DETECTION ############################
    def LD_trackbarCallback(self, trackbarVals):
            self.Yellow_TH_low = trackbarVals.data[0]
            self.Yellow_TH_high = trackbarVals.data[1]
            self.Yellow_Sat_low = trackbarVals.data[2]
            self.Yellow_Sat_high = trackbarVals.data[3]
            self.Yellow_Value_low = trackbarVals.data[4]
            self.Yellow_Value_high = trackbarVals.data[5] 
            self.Red_TH_low = trackbarVals.data[6]
            self.Red_TH_high = trackbarVals.data[7]
            self.Red_Sat_low = trackbarVals.data[8]
            self.Red_Sat_high = trackbarVals.data[9]
            self.Red_Value_low = trackbarVals.data[10]
            self.Red_Value_high = trackbarVals.data[11]            
            self.Blue_TH_low = trackbarVals.data[12]
            self.Blue_TH_high = trackbarVals.data[13]
            self.Blue_Sat_low = trackbarVals.data[14]
            self.Blue_Sat_high = trackbarVals.data[15]
            self.Blue_Value_low = trackbarVals.data[16]
            self.Blue_Value_high = trackbarVals.data[17]            
            self.CannyWhiteTH_Low = trackbarVals.data[18]
            self.CannyWhiteTH_High = trackbarVals.data[19]
            self.CannyParamLow = trackbarVals.data[20]
            self.CannyParamHigh = trackbarVals.data[21]
            self.polyArcLengthParam = trackbarVals.data[22]


    def LD_run(self, image):
        #reciveing pictures and timestamp
        #Processing pipeline
        #Original image -> Preprocessed image
        prePorcessedImage = self.LD_pre_proc(image)
        # Preproccessed image -> car mask
        fullImg = np.full_like(prePorcessedImage, 255)
        invCarMask = fullImg - self.LD_roiMaskCar
        carMaskedImage = self.maskRegionOfInterest(prePorcessedImage, invCarMask)

        # Car masked image -> ROI mask image
        maskedImage = self.maskRegionOfInterest(carMaskedImage, self.LD_roiMask)

        # Fully masked image -> Warped image
        warpedMaskedImage = self.LD_perspective_warp(maskedImage, 1)

        #Calculate and draw middle line
        # warpedMaskedImage -> image with middle line
        middleLineImage = self.LD_GetMiddleLine(warpedMaskedImage)

        #Detect cross line in histogrammed image
        histogramImage = self.LD_CrossLineDetection(middleLineImage)

        #Calculate error
        errorImg = self.LD_getError(image)

        # publish messages
        self.LD_PublishMsgs()
       
        if DEBUG_MODE == 1:
            car_masked_image_rgb = self.maskRegionOfInterest(image, invCarMask)
            masked_image_rgb = self.maskRegionOfInterest(car_masked_image_rgb, self.LD_roiMask)
            masked_image_color = self.maskRegionOfInterest(masked_image_rgb, self.LD_roiMask)
            warped_image_rgb = self.LD_perspective_warp(masked_image_rgb, 1)

        #Streaming the images
        return prePorcessedImage, errorImg, histogramImage


    def LD_PublishMsgs(self):
        if (self.headingErrorMsg.data > ROADSTATE_THRESHOLD or self.headingErrorMsg.data < -ROADSTATE_THRESHOLD):
            self.roadStateMsg.data = ROADSTATE_CURVE
        else:
            self.roadStateMsg.data = ROADSTATE_STRAIGHT

        self.lateralErrorPub.publish(self.lateralErrorMsg)
        if(self.LD_ExitIntersection == 1):
            self.roadStateMsgPub.publish(30)
            self.LD_ExitIntersection = 0
        else:
            self.roadStateMsgPub.publish(self.roadStateMsg)
        self.headingErrorPub.publish(self.headingErrorMsg) 
        self.intersectionStatePub.publish(self.intersectionStateMsg)


    def MakeRoiMaskImage(self, roiTopLeft, roiTopRight, roiBottomLeft, roiBottomRight):
        polygons = np.array([ 
            [roiBottomLeft, roiBottomRight, roiTopRight, roiTopLeft] 
            ]) 
        mask = np.zeros((480, 640),dtype=np.uint8) 
         # Fill poly-function deals with multiple polygon 
        cv2.fillPoly(mask, polygons, (255, 255, 255))  
        return mask
                

    ## inv = -1
    ## normal = 1 or else
    def LD_perspective_warp(self, img, inv):
        dst_size=(640,480)
        src=np.float32([(0.15,0.2), (0.85,0.2), (-0.7,1), (1.7,1)])
        dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])
        if (inv == -1):
            temp = src
            src = dst
            dst = temp
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        dst = dst * np.float32(dst_size)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped

    def LD_GetPartialHistSum(self, imgHeight, fromIndex, untilIndex):
        windowsSUM = np.full_like(self.windowsHist[0], 0)
        for index in range(fromIndex, untilIndex):
            windowsSUM = windowsSUM + self.windowsHist[index]
        windowsSUM = windowsSUM/(windowsSUM.max()/255.0)
        return windowsSUM


        

    def LD_CrossLineDetection(self, img):
        lowerHistSum = self.LD_GetPartialHistSum(img.shape[0], 0, self.windowsNumber)
        crossLineWindowIndicies = [None] * self.windowsNumber 
        for windowIndex in range(self.windowsNumber):
            histCounter = 0
            laneWidth = self.rightWindowLine[windowIndex] - self.leftWindowLine[windowIndex]
            for columnIndex in range(self.leftWindowLine[windowIndex][0], self.rightWindowLine[windowIndex][0]):
                if self.windowsHist[windowIndex][columnIndex] > 1:
                    histCounter = histCounter + 1
            if(histCounter/laneWidth > 0.5):
                # print("Crosslane detected in", windowIndex)
                # print(histCounter/laneWidth) 
                crossLineWindowIndicies[windowIndex] = 1
        
        self.intersectionStateMsg.data = 0

        if 1 in crossLineWindowIndicies:
            self.intersectionStateMsg.data = 10

        if crossLineWindowIndicies[0] == 1 or crossLineWindowIndicies[1] == 1:
            self.intersectionStateMsg.data = 20
            self.LD_IntersectionLatch = 1
            self.LD_IntersectionTimeStamp = rospy.get_time()
        
    
        if DRAW_MODE == 1:
            lowerHistSum = img.shape[0] - lowerHistSum    
            x = np.linspace(0, img.shape[1]-1, img.shape[1] )
            pts = np.vstack((x, lowerHistSum)).astype(np.int32).T
            cv2.polylines(img, [pts], isClosed=False, color=(255,0,255), thickness=2)

        return img

    def LD_GetLineCount(self, leftLine, rightLine):
        #wait 2 sec
        
        if ((rospy.get_time() - self.LD_IntersectionTimeStamp) > 4.5) and self.LD_IntersectionLatch == 1:
            if leftLine != 0 and rightLine != 0:
                self.LD_IntersectionLatch = 0
                self.LD_ExitIntersection = 1


    def LD_GetMiddleLine(self, img):
        height = img.shape[0]
        height_roi = img.shape[0] - (img.shape[0] - self.LD_roiBottomLeftRect[1]) 
        width = img.shape[1]
        widthMiddle = width // 2

        tImg = np.copy(img)
        self.windowsHist = np.full((self.windowsNumber, img.shape[1]), 0)
        windowHeight = height//5//self.windowsNumber
        for idx in range(0, self.windowsNumber):
            windowTop = height - windowHeight * (idx + 1)
            windowBottom = height - windowHeight * (idx) - 1
            self.windowsHist[idx] = np.sum(img[windowTop:windowBottom,:], axis=0) #sums by rows
            if (self.windowsHist[idx].max() != 0):
                self.windowsHist[idx] = self.windowsHist[idx]/(self.windowsHist[idx].max()/255.0)
            windowIndices = np.argwhere(self.windowsHist[idx] > 0)
            #deadzone in the middle, so false points won't be added
            deadZone = 40
            leftWindowIndices = windowIndices[windowIndices < (widthMiddle-deadZone)]
            rightWindowIndices = windowIndices[windowIndices > (widthMiddle+deadZone)]
            newLeftWindowLine = self.leftWindowLine[idx]
            newRightWindowLine = self.rightWindowLine[idx]
            if leftWindowIndices.size > 3:
                newLeftWindowLine = int(np.average(leftWindowIndices))
            if rightWindowIndices.size > 3:
                newRightWindowLine = int(np.average(rightWindowIndices))
            
            if idx == 0:
                self.LD_GetLineCount(leftWindowIndices.size, rightWindowIndices.size)

            LD_MAX_ALIGNMENT_TH = 30
            if(idx > 0):
                if (newLeftWindowLine - self.leftWindowLine[idx - 1]) > LD_MAX_ALIGNMENT_TH:
                        newLeftWindowLine = self.leftWindowLine[idx - 1] + LD_MAX_ALIGNMENT_TH
                if (newLeftWindowLine - self.leftWindowLine[idx - 1]) < -LD_MAX_ALIGNMENT_TH:
                    if self.leftWindowLine[idx - 1] > LD_MAX_ALIGNMENT_TH:
                        newLeftWindowLine = self.leftWindowLine[idx - 1] - LD_MAX_ALIGNMENT_TH
                    else:
                        newLeftWindowLine = 0

                if (newRightWindowLine - self.rightWindowLine[idx - 1]) > LD_MAX_ALIGNMENT_TH:
                    if self.rightWindowLine[idx - 1] + LD_MAX_ALIGNMENT_TH < img.shape[1]:
                        newRightWindowLine = self.rightWindowLine[idx - 1] + LD_MAX_ALIGNMENT_TH
                    else:
                        newRightWindowLine = img.shape[1]
                if (newRightWindowLine - self.rightWindowLine[idx - 1]) < -LD_MAX_ALIGNMENT_TH:
                    newRightWindowLine = self.rightWindowLine[idx - 1] - LD_MAX_ALIGNMENT_TH
            
            if abs(newLeftWindowLine - newRightWindowLine) > 40:
                self.leftWindowLine[idx] = newLeftWindowLine
                self.rightWindowLine[idx] = newRightWindowLine
            else:
                # print(self.rightWindowLine[idx])
                pass

            
            cv2.line(tImg, (self.leftWindowLine[idx], windowBottom), (self.leftWindowLine[idx], windowTop), (255, 255, 0), 2)
            cv2.line(tImg, (self.rightWindowLine[idx], windowBottom), (self.rightWindowLine[idx], windowTop), (255, 255, 0), 2)
            
        self.middleLine = int((self.leftWindowLine[0] + self.rightWindowLine[0]) // 2)
        self.middleWindowLine = np.mean([self.leftWindowLine, self.rightWindowLine], axis=0)
        y = self.middleWindowLine[-1]-self.middleWindowLine[0]
        x = self.windowsNumber*windowHeight
        self.headingErrorMsg.data = math.degrees(math.atan2(y, x))

        for idx in range(0, self.windowsNumber):
           windowTop = height - windowHeight * (idx + 1)
           windowBottom = height - windowHeight * (idx) - 1
           cv2.line(tImg, (self.middleWindowLine[idx], windowBottom), (self.middleWindowLine[idx], windowTop), (255, 255, 0), 2)

        self.LD_filterMiddleLine() # exponential filter

        if DRAW_MODE:
            cv2.line(tImg, (self.leftLine, 0), (self.leftLine,480), (255,255,0), 1)
            cv2.line(tImg, (self.rightLine, 0), (self.rightLine,480), (255,255,0), 1)
            cv2.line(tImg, (self.middleLine, 0), (self.middleLine,480), (255,255,0), 1)

        return tImg
    
    def LD_filterMiddleLine(self):
        self.middleLine = int(self.middleLine * self.alfaExp + (1-self.alfaExp) * self.middleLinePrev)
        self.middleLinePrev = self.middleLine


    def LD_GetWhitePercentage(self, image):
        maskHistTopLeft = (0, 300)
        maskHistTopRight = (640, 300)
        maskHistBottomLeft = (0, 480)
        maskHistBottomRight = (640, 480)
        LD_roiMaskHist = self.MakeRoiMaskImage(maskHistTopLeft, maskHistTopRight, maskHistBottomLeft, maskHistBottomRight)
        maskedImageRectTrap = cv2.bitwise_and(image, LD_roiMaskHist)
        #mask out the car
        fullImg = np.full_like(image, 255)
        invCarMask = fullImg - self.LD_roiMaskCar
        maskedImageRectTrapCar = cv2.bitwise_and(maskedImageRectTrap, invCarMask)

        deadzoneTopLeft = (260, 240)
        deadzoneTopRight = (380, 240)
        deadzoneBottomLeft = (260, 480)
        deadzoneBottomRight = (380, 480)
        deadzoneMask = self.MakeRoiMaskImage(deadzoneTopLeft, deadzoneTopRight, deadzoneBottomLeft, deadzoneBottomRight)
        invDeadZone = fullImg - deadzoneMask
        maskedImageRectTrapCarDeadzone = cv2.bitwise_and(maskedImageRectTrapCar, invDeadZone)
        

        # cv2.imshow("image", maskedImageRectTrapCarDeadzone)
        # cv2.waitKey(0)

        # mask the first rectangle
        percent = np.average(maskedImageRectTrapCarDeadzone)
        return percent
        

    def LD_pre_proc(self, img, s_thresh=(50, 255), sx_thresh=(80, 255)):
        # if DEBUG_MODE == 1:
        #     img=np.copy(img)
        imageHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_threshold = cv2.inRange(imageHSV, (0, 0, 65), (180, LD_WHITE_SAT_TH_VALS[0], 255))
        percentArray = np.array([])
        percent = self.LD_GetWhitePercentage(white_threshold)
        for satVal in LD_WHITE_SAT_TH_VALS:
            white_threshold = cv2.inRange(imageHSV, (0, 0, 65), (180, satVal, 255))
            percent = self.LD_GetWhitePercentage(white_threshold)
            percentArray = np.append(percentArray, percent)
            
        minDiff = abs(percentArray[0] - LD_WHITE_PERCENTAGE)
        goodIdx = 0
        for index in range(0, len(LD_WHITE_SAT_TH_VALS)):
            diff = abs(percentArray[index] - LD_WHITE_PERCENTAGE)
            if (minDiff > diff):
                goodIdx = index
                minDiff = diff

        if percentArray[goodIdx] < 1:
            goodIdx = len(LD_WHITE_SAT_TH_VALS)//2

        white_threshold = cv2.inRange(imageHSV, (0, 0, self.CannyWhiteTH_Low), (180, LD_WHITE_SAT_TH_VALS[goodIdx], self.CannyWhiteTH_High))

        HISTARRAY[goodIdx] = HISTARRAY[goodIdx] + 1 

        # print(LD_WHITE_SAT_TH_VALS[goodIdx])
        # print(percentArray[goodIdx])    

        edges = cv2.Canny(white_threshold, self.CannyParamLow, self.CannyParamHigh)
        return edges


    def LD_getError(self, img):
        height = img.shape[0]
        width = img.shape[1]
       
        errorLateral = self.middleLine - (width // 2)
        errorLateral = -errorLateral

        if DRAW_MODE == 1:
            cv2.line(img, (320, 480), (self.middleLine,480), (22,255,123), 5)
            cv2.line(img, (self.middleLine, 200), (self.middleLine,480), (0,255,0), 3)
            cv2.line(img, (self.middleWindowLine[-1], 200), (self.middleLine,480), (0,255,255), 3)

        self.lateralErrorMsg.data = errorLateral
        return img

    def maskRegionOfInterest(self, image, roiMask): 
        # Bitwise operation between canny image and mask image
        if (image.shape != roiMask.shape):
            roiMask = np.stack([roiMask, roiMask, roiMask], axis = 2)
        masked_image = cv2.bitwise_and(image, roiMask)
        return masked_image 


   
############################ ----------------- ############################


############################ TRAFFIC SIGN DETECTION ############################
    def TSD_run(self, image):
        #doing stuff with the picture
        masked_image = self.maskRegionOfInterest(image, self.TSD_roiMask)
        temp, ProcedImage, detectedShapesNR, boundingBox = self.TSD_preProcessing(masked_image)
        #tLaneStatus = self.TSD_classification(detectedShapesNR)
        tLaneStatus = self.TSD_classification_boudingBox(boundingBox)

        #publish messages
        self.trafficSignDetStatus.data = int(tLaneStatus[0])
        self.trafficSignDetStatusPub.publish(self.trafficSignDetStatus)

        return temp, ProcedImage
        
        #rospy.loginfo(rospy.is_shutdown())    

    def TSD_sumPixelVals(self, channel, colorIndex, x1, y1, width, height):

        #yellow
        #magic threshold: 0.1
        if colorIndex == 0:
            whitePixelCount = (channel[y1:y1+height, x1:x1 + width] <  110).sum()
    
        #red
        #nincs szukseg magicre
        elif colorIndex == 1:
            whitePixelCount = (channel[y1:y1+height, x1:x1 + width] <  160).sum()

        #blue
        #magic threshold: 0.6 (park és pedestrian kozott)
        elif colorIndex == 2 or colorIndex == 3:
            whitePixelCount = (channel[y1:y1+height, x1:x1 + width] <  85).sum()
        
        overallPixelCount = (width*height)
        print(whitePixelCount / overallPixelCount)
        heuristicClassificationMAGIC = whitePixelCount / overallPixelCount 
        return heuristicClassificationMAGIC


    def TSD_preProcessing(self, img):
        tempImg=np.copy(img)
        font = cv2.FONT_HERSHEY_COMPLEX
        #hvs=cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype('uint8')
        hvs=cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('uint8')
        # separate channels
        h_channel = hvs[:,:,0]
        s_channel = hvs[:,:,1]
        v_channel = hvs[:,:,2]
        
        # histogram correction
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10,10))
        v_channel = clahe.apply(v_channel)

        # yellow mask
        yellow_threshold = cv2.inRange(hvs, (self.Yellow_TH_low, self.Yellow_Sat_low, self.Yellow_Value_low), (self.Yellow_TH_high, self.Yellow_Sat_high, self.Yellow_Value_high))

        # red masks
        red_threshold1 = cv2.inRange(hvs, (0, self.Red_Sat_low, self.Red_Value_low), (self.Red_TH_low, self.Red_Sat_high, self.Red_Value_high))
        red_threshold2 = cv2.inRange(hvs, (self.Red_TH_high, self.Red_Sat_low, self.Red_Value_low), (180, self.Red_Sat_high, self.Red_Value_high))
        red_threshold = cv2.bitwise_or(red_threshold1, red_threshold2)

        # blue masks ~
        blue_threshold = cv2.inRange(hvs, (self.Blue_TH_low, self.Blue_Sat_low, self.Blue_Value_low), (self.Blue_TH_high, self.Blue_Sat_high, self.Blue_Value_high))
        colors = ["yellow", "red", "blue"]

        # combine yellow-red-blue masks
        combined_threshold = cv2.bitwise_or(yellow_threshold, red_threshold)
        
        combined_threshold = cv2.bitwise_or(combined_threshold, blue_threshold)
        
        combinedColorThreshold = np.array((yellow_threshold, red_threshold, blue_threshold))

        _, combined_contours, _ = cv2.findContours(combined_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        detectedShapes = []
        detectedShapesNR = []
        boundingBoxNR = []
        detectedShapesPoint = np.array([])

        # iterate through every mask
        for idx, colorMaps in enumerate(combinedColorThreshold): # idx: 0-yellow, 1-red, 2-blue
            _, tempContour, _ = cv2.findContours(colorMaps, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            arcLengthMax = 0.0
            cntMax = None
            for cnt in tempContour:
                arcLength = cv2.arcLength(cnt, True)
                # draws the max arc length bounding box for EACH color, if only one is necessary, indent <-- backwards
                if (arcLength > 80.0) and (arcLength < 250.0):
                    approx = cv2.approxPolyDP(cnt, 0.02*arcLength, True)
                    cv2.drawContours(tempImg, [approx], 0, (255), 5)
                    x = approx.ravel()[0]
                    y = approx.ravel()[1]
                    boundRect = cv2.boundingRect(approx)
                    boundRectRatio = boundRect[2]/boundRect[3]
                    if boundRectRatio > 0.9 and boundRectRatio < 1.1 and boundRect[2] > 25 and boundRect[2] < 65:
                        magicClassVal = self.TSD_sumPixelVals(v_channel, idx, boundRect[0], boundRect[1], boundRect[2], boundRect[3])
                        cv2.rectangle(tempImg, (int(boundRect[0]), int(boundRect[1])), \
                            (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (255, 255, 0), 2)
                        if idx == 2 or idx == 3: #blue
                            if magicClassVal < 0.6:
                                idx = 3 # PARK
                            elif magicClassVal > 0.08:
                                idx = 2 # PEDESTRIAN 
                            boundingBoxNR.append(idx)
                        if idx == 0 and magicClassVal < 0.12: #yellow
                            boundingBoxNR.append(idx)
                        if idx == 1: #red
                            boundingBoxNR.append(idx)
                        detectedShapesPoint = np.append(detectedShapesPoint, [x,y])            
            else:
                pass

        return tempImg, combined_threshold, detectedShapesNR, boundingBoxNR

    def TSD_classification_boudingBox(self, boudingBoxes):
        tLaneStatus = []
        tLaneStatusString = []
    
        if 0 in boudingBoxes:
            self.plausabilityMain += 1
        else:
            self.plausabilityMain -= 1

        if self.plausabilityMain > 10:
            self.plausabilityMain = 10
        elif self.plausabilityMain < 0:
            self.plausabilityMain = 0

        if self.plausabilityMain > 5:
            tLaneStatus.append(TSDSTATE_MAIN)
            tLaneStatusString.append("Main road")



        # check for pedestrian sign
        if 2 in boudingBoxes: #blue 
            self.plausabilityPedastrian += 1
        else:
            self.plausabilityPedastrian -= 0.5
        
        if self.plausabilityPedastrian < 0:
            self.plausabilityPedastrian = 0
        elif self.plausabilityPedastrian > 15:
            self.plausabilityPedastrian = 15
        
        if self.plausabilityPedastrian > 5:
            tLaneStatus.append(TSDSTATE_PEDASTRIAN)
            tLaneStatusString.append("Pedastrian")

        # check for park sign
        if 3 in boudingBoxes: #blue
            self.plausabilityPark += 1
        else:
            self.plausabilityPark -= 1
        
        if self.plausabilityPark < 0:
            self.plausabilityPark = 0
        elif self.plausabilityPark > 10:
            self.plausabilityPark = 10
        
        if self.plausabilityPark > 5:
            tLaneStatus.append(TSDSTATE_PARK)
            tLaneStatusString.append("Park")

        # check for STOP road sign
        if 1 in boudingBoxes: #blue
            self.plausabilityStop += 1
        else:
            self.plausabilityStop -= 1
        
        if self.plausabilityStop < 0:
            self.plausabilityStop = 0
        elif self.plausabilityStop > 10:
            self.plausabilityStop = 10
        
        if self.plausabilityStop > 5:
            tLaneStatus.append(TSDSTATE_STOP)
            tLaneStatusString.append("STOP")


        if (tLaneStatus == []):
            tLaneStatus.append(TSDSTATE_NONE)
            tLaneStatusString.append("None")
        
        print(tLaneStatus)
        print(tLaneStatusString)
        return tLaneStatus


    def TSD_classification(self, detectedShapesNR):
        tLaneStatus = []
        tLaneStatusString = []

        # check for pedestrian sign
        if 21 in detectedShapesNR: #blue triangle
            temp = np.array([22, 23, 24, 25])
            if np.any(np.in1d(temp,detectedShapesNR)):
                self.plausabilityPedastrian += 1
            else:
                pass
        else:
            self.plausabilityPedastrian -= 1
        
        if self.plausabilityPedastrian < 0:
            self.plausabilityPedastrian = 0
        elif self.plausabilityPedastrian > 10:
            self.plausabilityPedastrian = 10
        
        if self.plausabilityPedastrian > 7:
            tLaneStatus.append(TSDSTATE_PEDASTRIAN)
            tLaneStatusString.append("Pedastrian")

        # check for park sign
        temp_bluecontour = np.array([24, 25])
        if np.any(np.in1d(temp_bluecontour, detectedShapesNR)): #blue circle or blue ellipse
            temp = np.array([22, 23, 24, 25])
            if np.any(np.in1d(temp,detectedShapesNR)):
                self.plausabilityPark += 1
            else:
                pass
        else:
            self.plausabilityPark -= 1
        
        if self.plausabilityPark < 0:
            self.plausabilityPark = 0
        elif self.plausabilityPark > 10:
            self.plausabilityPark = 10
        
        if self.plausabilityPark > 7:
            tLaneStatus.append(TSDSTATE_PARK)
            tLaneStatusString.append("Park")
        self.plausabilityMain += 1


        # check for main road sign
        temp_yellow_e_c = np.array([4, 5])
        temp = np.array([1, 2, 3]) # any other yellow is OK, but not good or bad
        if np.any(np.in1d(temp_yellow_e_c, detectedShapesNR)): # yellow ellipse or circle
            self.plausabilityMain += 2
        elif np.any(np.in1d(temp, detectedShapesNR)):
            pass
        else:
            self.plausabilityMain -= 1
        
        if self.plausabilityMain < 0:
            self.plausabilityMain = 0
        elif self.plausabilityMain > 10:
            self.plausabilityMain = 10
        
        if self.plausabilityMain > 6:
            tLaneStatus.append(TSDSTATE_MAIN)
            tLaneStatusString.append("Main road")

        # check for STOP road sign -- NAGYON kezdetleges
        temp = np.array([11, 12, 13, 14, 15, 16])
        if np.any(np.in1d(temp, detectedShapesNR)): # any red contour
            self.plausabilityStop += 1
        else:
            self.plausabilityStop -= 2
        
        if self.plausabilityStop < 0:
            self.plausabilityStop = 0
        elif self.plausabilityStop > 10:
            self.plausabilityStop = 10
        
        if self.plausabilityStop > 6:
            tLaneStatus.append(TSDSTATE_STOP)
            tLaneStatusString.append("STOP")

        
        if (tLaneStatus == []):
            tLaneStatus.append(TSDSTATE_NONE)
            tLaneStatusString.append("None")
        print(tLaneStatus)
        print(tLaneStatusString)
        return tLaneStatus

############################ ----------------- ############################

if __name__ == '__main__':
    ##### INIT #####
    rospy.init_node('ImageProcessorNode')

    ##### CONFIG #####
    # config nodes
    enableLaneDetection = False
    enableTrafficSignDetection = False
    enableCameraSpoof = True
    
    # camera stream pipes
    camSt1R, camSt1S = Pipe(duplex = False)           # camera  ->  node
    camSt2R, camSt2S = Pipe(duplex = False)           # node  ->  streamer
    camSt3R, camSt3S = Pipe(duplex = False)           # node  ->  streamer2
    camSt4R, camSt4S = Pipe(duplex = False)           # node2  ->  streamer
    camSt5R, camSt5S = Pipe(duplex = False)           # node2 ->  streamer2

    allProcesses = list()
    #####-#####

    ##### LOGIC #####
    if enableCameraSpoof:
        #camSpoofer = CameraSpooferProcess([],[camSt1S],'/home/pi/catkin_ws/src/vid') # REL PATH NEEDED
        #camSpoofer = CameraSpooferProcess([],[camSt1S],'/home/bence/catkin_ws/src/demo') # REL PATH NEEDED
        #camSpoofer = CameraSpooferProcess([],[camSt1S],'/home/antalmarton/BFMC_catkin_ws/src/videos') # REL PATH NEEDED
        #camSpoofer = CameraSpooferProcess([],[camSt1S],'/home/antalmarton/BFMC_catkin_ws/src/vid2') # REL PATH NEEDED
        #camSpoofer = CameraSpooferProcess([],[camSt1S],'/home/antalmarton/BFMC_catkin_ws/src/vid_tsd') # REL PATH NEEDED
        #camSpoofer = CameraSpooferProcess([],[camSt1S],'/home/antalmarton/BFMC_catkin_ws/src/vid_keresztkep') # REL PATH NEEDED
        #camSpoofer = CameraSpooferProcess([],[camSt1S],'/home/antalmarton/BFMC_catkin_ws/src/bosch_vid') # REL PATH NEEDED
        #camSpoofer = CameraSpooferProcess([],[camSt1S],'/home/antalmarton/BFMC_catkin_ws/src/exam_vid') # REL PATH NEEDED
        camSpoofer = CameraSpooferProcess([],[camSt1S],'/home/pi/catkin_ws/src/exam_vid') # REL PATH NEEDED
        #camSpoofer = CameraSpooferProcess([],[camSt1S],'/home/antalmarton/BFMC_catkin_ws/src/vid_tsd_exam') # REL PATH NEEDED
        allProcesses.append(camSpoofer)
    else:
        camProc = CameraProcess([],[camSt1S])
        allProcesses.append(camProc)

    streamProc = CameraStreamer([camSt2R], [], 2244)
    streamProc2 = CameraStreamer([camSt3R], [], 2245)
    streamProc3 = CameraStreamer([camSt4R], [], 2246)
    streamProc4 = CameraStreamer([camSt5R], [], 2247)

    imageProcNode = ImageProcessing([camSt1R],[camSt2S, camSt3S, camSt4S, camSt5S])

    # append nodes
    # allProcesses.append(camProc)
    # allProcesses.append(camSpoofer)
     allProcesses.append(streamProc)
    #allProcesses.append(streamProc2)
    #allProcesses.append(streamProc3)
    #allProcesses.append(streamProc4)

    ## START PROCESSES
    for proc in allProcesses:
        proc.deamon = True
        proc.start()

    imageProcNode.run()

    print("IMG PROC Node shut down")

    # ~KeyBoard Interrupt
    for proc in allProcesses:
        proc.stop()
        proc.terminate()
        proc.join()

