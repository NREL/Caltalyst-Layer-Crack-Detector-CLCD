import cv2
import numpy as np
import os
import scipy.signal
import sys
import time
import matplotlib.pyplot as plt


dir = "/Users/jpfeilst/Desktop/Giner/Giner keyence/200930 test coatings" # directory where .tif micrographs are saved

showFinal = 0 # show each final image
saveFinal = 1 # save each final image with crack percent and threshold values in the file's name

outputDirectoryDataFile = 0 # save all path, crack percent, and threshold values in a text file

autoThresh = 1 # set = 1 to automatically determine threshold value for each image
singleThresh = 90 # if autoThresh = 0 then this is the threshold value that will be used for all images (0 to 255)

saveLineCuts = 0 # save a figure of line cuts of the image that shows threshold and mean-standard deviations



cracksAreLight = 0 #if 0 looks for dark cracks, if 1 looks for light cracks

limFiles = 0
lowLim = 0
highLim = 1

showDeriveFig = 0
showAllDeriveFigs = 0



stdThresh1OrAccelThresh0 = 0



minSizeContours = 5  # 5 is standard

maskCorneredgeIn = True #True to mask off bottom right corner for scale bar False to mask nothing.

useBlur = 1
blurKernal1=1
blurKernal2=3

accelThreshNoBlur = 0.3
accelThreshBlur = 0.029  # 0.35 standard

showOrig = 0
showBlur = 0
showThresh = 0
saveThresh = 0

SGwindow = 11
SGdelThresh = 100
SGlim = 3

timeStart = time.time()


# find all the files in dir, even those nested in sub directories using "walk"
filePaths = []
for r, d, f in os.walk(dir):
    for file in f:
        # if it is an image file that has not been processed yet add it to the list
        if ".tif" in file and "Crack%" not in file and "lineCut" not in file:
            filePaths.append(os.path.join(r, file))


if limFiles == 1:
    filePaths = filePaths[lowLim:highLim]



allDeriveFigs = []



print("\n\n")
# function that does element wise division between two list of the same size, but returns 0 for pairs where denominator is 0
def divideIgnoreZeroBot(top,bot):
    out = []
    for topI,botI in zip(top,bot):
        if botI == 0:
            out.append(0.)
        else:
            out.append(topI/botI)
    return out

# performs SG filtering on a list succesively until the curent filtered result's error sqayred from the last filter is less than thresh or lim number of filterings have occured
def sucGolayToConv(list, window, order, thresh, lim):
    tempList1 = list
    tries = 0
    while True:
        tempList2 = scipy.signal.savgol_filter(tempList1, window, order)
        if np.sum((tempList2 - tempList1) ** 2) < thresh or tries > lim:
            break
        tempList1 = tempList2
        tries += 1
    return tempList2

# progress bar in console code. dont forget to sim terminal in pycharm
def progress(count, total, status=''):
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('\t[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

# define function that does all the work and will be performed with all the files from above (see bottom)
def crackIdent(ImagePath,thresh,maskCorneredge=maskCorneredgeIn,showFinal=0,saveFinal=0,showBlur=0,showThresh=0,saveThresh=0,blurKernal1=5,blurKernal2=5):

    crackThresh = thresh
    # open image file as a cv2 image object using the grayscale mode (the "0")
    img = cv2.imread(ImagePath,0)

    if cracksAreLight == 1:
        img = cv2.bitwise_not(img)

    # open image in normal mode (color) for displaying.
    imgColor = cv2.imread(ImagePath)
    # serially blur the image using two different kernal sizes
    im_gauss = 0
    if useBlur == 1:
        im_gauss = cv2.GaussianBlur(img, (blurKernal1, blurKernal1), 0)
        im_gauss = cv2.GaussianBlur(im_gauss, (blurKernal2, blurKernal2), 0)
    else:
        im_gauss = img




    totalArea = len(im_gauss)*len(im_gauss[0])
    totalL=len(im_gauss)
    totalW=len(im_gauss[0])

    if showOrig == 1:
        cv2.imshow("orig",img)
        cv2.waitKey()
    if showBlur == 1:
        cv2.imshow("blur",im_gauss)
        plt.show()
        cv2.waitKey()
        #cv2.imwrite(ImagePath[:-4] + "blur.tif", im_gauss)

    # Mask off the lower right corner (scale bar rom Keyence)
    if maskCorneredge == True:
        # width and height of the mask
        w = 300
        h = 100
        # dimension of the image in pixels to locate above sized rectangle
        y = int(len(im_gauss))
        x = int(len(im_gauss[0]))
        # create a matrix the size of the image with values of 255 (full 8 bits)
        mask = np.full((im_gauss.shape[0], im_gauss.shape[1]), 255, dtype=np.uint8)
        # draw a rectangle on the above full matrix filling it with zeros
        cv2.rectangle(mask,(x - w, y - h),(x, y), (0, 0, 0), cv2.FILLED)
        #bitwise and operator the blurred image with its self along with the mask. This nullifies pixels based on the mask
        imCopyMasked = cv2.bitwise_and(im_gauss, im_gauss, mask=mask)
        # put white around the edge pixels to be able to detect edge bound dark spots (cracks)
        cv2.rectangle(imCopyMasked,(0,0), (x, y), (255, 255, 0), 2)
        ret, thresh = cv2.threshold(imCopyMasked, crackThresh, 255, 0)
    else:
        w=0
        h=0
        y = int(len(im_gauss))
        x = int(len(im_gauss[0]))
        cv2.rectangle(im_gauss, (0, 0), (x, y), (255, 255, 0), 2)
        ret, thresh = cv2.threshold(im_gauss, crackThresh, 255, 0)


    # binary threshhold the image so that all pixels are put in two bins; those above a value of 100 and those below


    # show the binaried image if desired
    if showThresh == 1:
        cv2.imshow("thresh",thresh)
        cv2.waitKey(0)
    if saveThresh == 1:
        cv2.imwrite(ImagePath[:-4] + "_Thresh.tif", thresh)

    # using the binaried image, find all the closed dark pixeled contours
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #filter contours by area to get rid of small contours which are noise and super large ones incase image contains dark bound
    filteredCons = []
    CrackAreas = []
    maskPickedUp=0
    numUnderSized = 0
    for con in contours:
        area = cv2.contourArea(con)
        if minSizeContours < area <totalArea/1.5:
            if cv2.pointPolygonTest(con, (totalW - 1, totalL - 2), False) == 0:
                maskPickedUp = 1
            filteredCons.append(con)
            CrackAreas.append(area)
        elif minSizeContours > area:
            numUnderSized+=1

    # Draw the area filtered contours onto the color copy of the original image
    cv2.drawContours(imgColor,filteredCons,-1,(0,255,0),)
    # calculate the area percent of detected crack pixels and round.
    if maskPickedUp == 1:
        deadArea = 2*totalL + 2*totalW + (h*w-2) #from mask and dead edge
        mask = h*w-2
    else:
        deadArea = 2 * totalL + 2 * totalW
        mask = 0
    crackPercent = str(round(((np.sum(CrackAreas)-(mask))/ (totalArea-(deadArea)))*100,3))

    # show the orig with contours drawn on it (highliting the detected cracks) with the crack% as its tittle
    if showFinal == 1:
        fileName = ImagePath.split(os.sep)[-1][:-4]
        cv2.imshow(fileName+'; CP = '+ crackPercent+"%, Threshold = "+str(crackThresh),imgColor)
        cv2.waitKey()

    # save the processed image in the same directory as the original
    if saveFinal == 1:
        cv2.imwrite(ImagePath[:-4]+"_Crack%"+ crackPercent +"Thresh"+str(crackThresh)+".tif",imgColor)
        #print("\tSaved ",ImagePath[:-4]+"_Crack%"+ crackPercent +"Thresh"+str(crackThresh)+".tif")
    #elif cutLine == 1:

    #print("Detected Crack percentage = " + crackPercent + "%")
    # print("mean circularity = " + str(round(np.mean(circularities),4)))

    return ((np.sum(CrackAreas)-(mask))/ (totalArea-(deadArea)))*100,numUnderSized


def autoThreshProcess(imagePath):
    crackPercents = []
    undersizeds = []
    threshs = []
    print("\t"+imagePath)
    img = cv2.imread(imagePath, 0)

    if cracksAreLight == 1:
        img = cv2.bitwise_not(img)

    # run through all possible threshold values, and save the crack percent and number of undersized contours for each threshold value


    for i in range(256):
        progress(i,255,status='Calculating optimal threshold')
        threshs.append(i)
        cpi,undersizedi = crackIdent(imagePath,i)
        crackPercents.append(cpi)
        undersizeds.append(undersizedi)
    print("\n",end="")

    SGwindow = 13
    SGdelThresh = 500
    SGlim = 2

    # calculate the second derivative of the number of undersized contours with respect to threshold value and SG filter it


    undersizedsSmoothed = sucGolayToConv(undersizeds,SGwindow,3,SGdelThresh,SGlim)
    firstD = np.gradient(undersizedsSmoothed,threshs)

    secondD = np.gradient(firstD,threshs)
    SGsecondD = sucGolayToConv(secondD,SGwindow,3,SGdelThresh,SGlim)
    SGsecondNorm = np.array(secondD) / max(SGsecondD)


    if showAllDeriveFigs == 1:
        allDeriveFigs.append(np.array(undersizedsSmoothed)/max(undersizedsSmoothed))



    if showDeriveFig == 1:
        plt.plot()
        plt.show()

    window = 10

    if useBlur == 1:
        accelThresh = accelThreshBlur
    else:
        accelThresh = accelThreshNoBlur

    testThresh = accelThresh#.03
    filterList = SGsecondNorm #list used to perform undersized thresholding on
    autothreshoutput = 0

    # run through the second derivaive and detect once the average of a window's width of points excede testThresh
    # save the threshold value at which this undersized acceleration excedes testThresh
    for i in range(window,len(filterList)-window,1):
        if np.average(filterList[i:i+window]) > testThresh:  #all(xi > testThresh for xi in filterList[i:i+window]):

            accelthreshoutput = threshs[i]
            break

    stdev = np.std(img)
    mean = np.average(img)

    stdthreshOutput = int(mean - 1.5*stdev)

    if stdThresh1OrAccelThresh0 == 0:
        threshOutput = accelthreshoutput
    else:
        threshOutput = stdthreshOutput

    # use the above detected threshold value to perform a normal crack detection!
    crackPercent, underSized = crackIdent(imagePath,threshOutput,showFinal=showFinal,saveFinal=saveFinal,showThresh=showThresh,saveThresh=saveThresh,showBlur=showBlur)

    if saveLineCuts == 1:

        plt.figure(figsize=(18,12))
        for i  in range(1,len(img),int(len(img)/5)):
            if i > 1:
                plt.plot(img[i], '-', markerfacecolor='b', markeredgecolor='b', color='b',alpha=0.6, lw=0.5/3)
            else:
                plt.plot(img[i], '-', markerfacecolor='b', markeredgecolor='b', color='b', alpha=0.6, lw=0.5 / 3,label=imagePath.split(os.sep)[-1][:-4] + " H-lineCuts")

        stdev = np.std(img)
        mean = np.average(img)
        textSize = 20
        plt.ylim(0,255)
        plt.ylabel("Pixel Value",size=textSize)
        plt.xlabel("Pixel location",size=textSize)
        plt.plot([0,len(img[0])],[threshOutput,threshOutput],'k--',linewidth=1,label="Threshold = "+str(accelthreshoutput))
        plt.plot([0, len(img[0])], [mean-stdev, mean-stdev], 'k:', linewidth=1, label="1x STD = "+str(round(mean-stdev,1)))
        plt.plot([0, len(img[0])], [mean-stdev*1.5, mean-stdev*1.5], 'r:', linewidth=1, label="1.5x STD = "+str(round(mean-stdev*1.5,1)))
        plt.plot([0, len(img[0])], [mean-stdev*2, mean-stdev*2], 'g:', linewidth=1, label="2x STD = "+str(round(mean-stdev*2,1)))
        plt.legend(prop={'size': textSize})
        plt.tick_params(axis='both', which='major', labelsize=textSize)
        plt.tight_layout()
        plt.savefig(imagePath[:-4]+"_lineCuts.png")
        plt.close()


    print("\t" + "Calculated threshold = "+str(threshOutput)+"\tCrack area % = "+str(round(crackPercent,4))+"%")
    if saveFinal == 1:
        print("\tSaved processed image "+ imagePath[:-4]+"_Crack%"+ str(round(crackPercent,4))+"Thresh"+str(threshOutput)+".tif")


    if stdThresh1OrAccelThresh0 == 0:
        return crackPercent, accelthreshoutput
    else:
        return crackPercent, threshOutput




# do it for all the files in filePaths determined at the top
crackPercents = []
thresholds = []
numDone = 1
numToDo = len(filePaths)

fileTimes = []
for fp in filePaths:
    if numDone > 1:
        print("\nWorking on # " +str(numDone)+ "/" +str(numToDo)+". Estimated time remaining for entire set = "+str(round(np.average(fileTimes)*(numToDo-numDone+1),1))+" seconds:")
    else:
        print("\nWorking on # " + str(numDone) + "/" + str(numToDo))
    t0 = time.time()
    if autoThresh == 1:
        crackPercent, autoTheshVal = autoThreshProcess(fp)
        print("\n")
        crackPercents.append(crackPercent)
        thresholds.append(autoTheshVal)
    else:
        crackPercents = []
        crackIdent(fp,singleThresh,showFinal=showFinal,saveFinal=saveFinal)
    t1 = time.time()


    fileTimes.append(t1 - t0)
    numDone+=1

if outputDirectoryDataFile ==1:
    outData = open(dir+os.sep+"crackDataForDirectory.txt","w+")
    outData.write("file name\tcrack percent\tthreshold value used\trod size\n")
    for i in range(len(filePaths)):
        fileName = filePaths[i].split(os.sep)[-1]
        rodSize = ""
        if len(fileName.split("(")) == 1:
            rodSize = "Not autodetected"
        else:
            rodSize = fileName.split("(")[1].split(")")[0]
        outData.write(fileName+"\t"+str(crackPercents[i])+"\t"+str(thresholds[i])+"\t"+rodSize +"\n")


    if useBlur == 1:
        accelThresh = accelThreshBlur
    else:
        accelThresh = accelThreshNoBlur

    if stdThresh1OrAccelThresh0 == 0:
        outData.write("\n\nConfig params:\naccelThresh\tSGwindow\tSGdelThresh\tSGlim\tblurKernal1\tblurKernal2\n")
        outData.write(str(accelThresh)+"\t"+str(SGwindow)+"\t"+str(SGdelThresh)+"\t"+str(SGlim)+"\t"+str(blurKernal1)+"\t"+str(blurKernal2)+"\n")
    else:
        outData.write("\n\nConfig params:\nSTDthresh\tSGwindow\tSGdelThresh\tSGlim\tblurKernal1\tblurKernal2\n")
        outData.write(str("1.5 STD total image") + "\t" + str(SGwindow) + "\t" + str(SGdelThresh) + "\t" + str(SGlim) + "\t" + str(blurKernal1) + "\t" + str(blurKernal2) + "\n")
    outData.close()
timeEnd = time.time()
print("\n\nTotal run time = ",timeEnd-timeStart)

if showAllDeriveFigs == 1:
    for allDeriveFig in allDeriveFigs:
        plt.plot(allDeriveFig,'-',markersize=3)
        plt.xlabel("Threshold Value")
        plt.ylabel("Undersized Contours (normalized)")
    plt.show()
