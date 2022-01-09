import rospy
import ros
import std_msgs.msg
import cv2
import numpy as np
import math
from std_msgs.msg import Int32, Float32

# PARAMETERS, DEFINES, MACROS
ROADSTATE_STRAIGHT = 10
ROADSTATE_NO_LEFTLANE = 11
ROADSTATE_NO_RIGHTLANE = 12
ROADSTATE_CURVE = 20
ROADSTATE_THRESHOLD = 10
DRAW_MODE = 1

LD_WHITE_PERCENTAGE = 10.0
#LD_WHITE_SAT_TH_VALS = np.linspace(0, 255, 8)
LD_WHITE_SAT_TH_VALS = [30, 90, 160]

HISTARRAY = np.full_like(LD_WHITE_SAT_TH_VALS, 0) 

class LaneDetection:
    def __init__(self, defaultValues):
        self.DEBUG_MODE = 1;
        self.SetTrackbarValues(defaultValues)
        self.lateralErrorTopicName = rospy.get_param('~lateralErrorTopicName', 'errorLateral')
        self.headingErrorTopicName = rospy.get_param('~headingErrorTopicName', 'errorHeading')
        self.intersectionStateTopicName = rospy.get_param('~intersectionStateTopicName', 'intersectionState')
        self.roadStateTopicName = rospy.get_param('~roadStateTopicName', 'roadState')

        self.lateralErrorMsg = Int32()
        self.headingErrorMsg = Float32()
        self.roadStateMsg = Int32()
        self.intersectionStateMsg = Int32()
        self.intersectionStateMsg.data = 0

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
                # LD node
        self.lateralErrorPub = rospy.Publisher(self.lateralErrorTopicName, Int32, queue_size=2)
        self.headingErrorPub = rospy.Publisher(self.headingErrorTopicName, Float32, queue_size=2)
        self.roadStateMsgPub = rospy.Publisher(self.roadStateTopicName, Int32, queue_size=2)
        self.intersectionStatePub = rospy.Publisher(self.intersectionStateTopicName, Int32, queue_size=2)
        print("LaneDetection node inited")

    def SetTrackbarValues(self, trackbarVals):
        self.Yellow_TH_low = trackbarVals[0]
        self.Yellow_TH_high = trackbarVals[1]
        self.Yellow_Sat_low = trackbarVals[2]
        self.Yellow_Sat_high = trackbarVals[3]
        self.Yellow_Value_low = trackbarVals[4]
        self.Yellow_Value_high = trackbarVals[5] 
        self.Red_TH_low = trackbarVals[6]
        self.Red_TH_high = trackbarVals[7]
        self.Red_Sat_low = trackbarVals[8]
        self.Red_Sat_high = trackbarVals[9]
        self.Red_Value_low = trackbarVals[10]
        self.Red_Value_high = trackbarVals[11]            
        self.Blue_TH_low = trackbarVals[12]
        self.Blue_TH_high = trackbarVals[13]
        self.Blue_Sat_low = trackbarVals[14]
        self.Blue_Sat_high = trackbarVals[15]
        self.Blue_Value_low = trackbarVals[16]
        self.Blue_Value_high = trackbarVals[17]            
        self.CannyWhiteTH_Low = trackbarVals[18]
        self.CannyWhiteTH_High = trackbarVals[19]
        self.CannyParamLow = trackbarVals[20]
        self.CannyParamHigh = trackbarVals[21]
        self.polyArcLengthParam = trackbarVals[22]



    def run(self, image):
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
       
        if self.DEBUG_MODE == 1:
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

    def MakeRoiMaskImage(self, roiTopLeft, roiTopRight, roiBottomLeft, roiBottomRight):
        polygons = np.array([ 
        [roiBottomLeft, roiBottomRight, roiTopRight, roiTopLeft] 
        ]) 
        mask = np.zeros((480, 640),dtype=np.uint8) 
         # Fill poly-function deals with multiple polygon 
        cv2.fillPoly(mask, polygons, (255, 255, 255))  
        return mask
                