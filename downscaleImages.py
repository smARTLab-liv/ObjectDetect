import os
import os.path
import sys
sys.dont_write_bytecode = True
import multiprocessing
from multiprocessing import Process, Queue
import time
from PIL import Image
import scipy.misc as misc
import numpy as np
import scipy.misc
import scipy.ndimage
from scipy.stats import norm

from classDefinitions import *
from random import random

#DEBUG = False
DEBUG = True

dirpath = list()

if(len(sys.argv) > 1):
  for i in range(1, len(sys.argv)):
    print sys.argv[i]
    dirpath.append(sys.argv[i])
else:
    dirpath.append("testfolder/")

MAX_ALLOWED_DEPTH = 600
MIN_AVG_DEPTH = 200

fromdirimg="color"
fromdirdep="depth"
todirimg="debug/color"
todirdep="debug/depth"
todirtarg="target"
coords_folder="coords"
output_folder="merged_out"

fromimgend=".png"
fromdepend=".npz"

toimgend=".png"
posoutend=".npy"

old_internal_name = "arr_0.npy"

original_image_size = (640, 480)
#image_size = np.array([64, 48])
#image_size = np.array([128, 96])
image_size = np.array([256, 192])

use_depth_channel = True
num_color_channels = 4 if use_depth_channel else 3

use_fake_distfunc = True

scalingFactor = np.divide(image_size.astype(np.float32), original_image_size)

loadFolders = ['Axis', 'Bearing', 'Bearing_Box', 'Distance_Tube', 'F20_20_B', 'F20_20_G', 'M20', 'M20_100', 'M30', 'Motor', 'R20', 'S40_40_B', 'S_40_40_G']

NUM_CLASSES = len(loadFolders)
NUM_CLASSES_AND_NOTHING = NUM_CLASSES + 1

def ensure_dir(file_path):
  if not os.path.exists(file_path):
    os.makedirs(file_path)

def handleSingleFile(f, subdirpath, cls):
  colorFold = os.path.join(subdirpath, fromdirimg)
  depthFold = os.path.join(subdirpath, fromdirdep)

  debugColorOut = os.path.join(subdirpath, todirimg)
  debugDepthOut = os.path.join(subdirpath, todirdep)

  outputPath = os.path.join(os.path.dirname(subdirpath), output_folder)
  ensure_dir(outputPath)
  coordOutputPath = os.path.join(os.path.dirname(subdirpath), coords_folder)
  ensure_dir(coordOutputPath)

  fname = os.path.splitext(os.path.basename(f))[0]
  iname = os.path.join(colorFold, f)
  dname = os.path.join(depthFold, fname+fromdepend)

  output_filename = os.path.join(outputPath, fname+toimgend)
  posfilename = os.path.join(coordOutputPath, fname+posoutend)

  if os.path.isfile(iname) and os.path.isfile(dname):
    isAKeeper = True
    #downsample image and save as numpy array
    image = scipy.misc.imread(iname)
    arrXd = np.array(image)

    arr = arrXd[:,:,0:3] # handle alpha channel separately - has pos info encoded!
    coordinateList = []
    if arrXd.shape[2] == 4: #we have an alpha channel
      alpha = arrXd[:,:,3]
      foundPos = np.where(alpha != classes['unspecified'])
      objectIndices = zip(*foundPos)
      values = [alpha[i] for i in objectIndices]
      coordinateList = zip(values, *foundPos)

      #scale down cordinates by same ratio
      coordinateList = [(c,y*scalingFactor[1],x*scalingFactor[0]) for (c,y,x) in coordinateList]#TODO 123 test test
      #print(coordinateList) # a list of (type, x, y) tuples

    else:
      return
    downscaled = scipy.misc.imresize(arr, (image_size[1], image_size[0]), 'lanczos')

    #now convert and downsample depth image
    dinp = np.load(dname)[old_internal_name]
    #if(np.amax(dinp) >= MAX_ALLOWED_DEPTH): #sort out weird files with spikes in depth channel. (could probably also just clamp at 600, but meh)
    #  isAKeeper = False
    dinp *= (255.0/MAX_ALLOWED_DEPTH) # scale from 0 to 600 to 0..255
    #replace depth with zeros
    #dinp = np.zeros(dinp.shape)
    #replace depth with random:
    #dinp = np.random.random_integers(0, 255, dinp.shape)
    downscaledD = scipy.misc.imresize(dinp, (image_size[1], image_size[0]), 'lanczos')

    avgDepthList = []
    #for i in range(1, len(sys.argv)):
    for idx in range(0, len(coordinateList)):
      centerCoordinates = np.array([coordinateList[idx][1], coordinateList[idx][2]])
      addedDepth = 0.0
      countedPixels = 0
      for y, x in np.ndindex(image_size[1], image_size[0]):
        distance = np.linalg.norm(centerCoordinates-np.array((y, x)))
        if distance < 15: # allows other classes not to be overwritten
          ix = int(round(x))
          iy = int(round(y))
          dvalue = downscaledD[iy, ix]
          if dvalue > 0:
            addedDepth += dvalue
            countedPixels += 1
      avgDepth = MIN_AVG_DEPTH if countedPixels == 0 else addedDepth / countedPixels
      if countedPixels != 0:
        avgDepth *= (MAX_ALLOWED_DEPTH/255.0) # scale back from 0..255 to 0..600
        avgDepth *= 0.1 # convert mm to cm
      avgDepthList.append(avgDepth)

    coordinateList = [(a,b,c,d) for (a,b,c),d in zip(coordinateList,avgDepthList)] # is now a 4-tuple, type, y, x, depth


      #debug image output
    if DEBUG:
      ensure_dir(debugColorOut)
      ensure_dir(debugDepthOut)
      debugColorName = os.path.join(debugColorOut, fname+toimgend)
      debugDepthName = os.path.join(debugDepthOut, fname+toimgend)

      img = Image.fromarray(downscaled.astype(np.uint8))
      img.save(debugColorName)

      img = Image.fromarray(downscaledD.astype(np.uint8))
      img.save(debugDepthName)
    #add "additional" empty dimension to concat later
    depthreshaped = np.reshape(downscaledD, ((image_size[1], image_size[0], 1)))

    rgbd = downscaled
    if(use_depth_channel):
      rgbd = np.concatenate((downscaled, depthreshaped), axis=2)

    if(not isAKeeper):
      print('File '+posfilename+' contained undesired depth value, dropping...')
      return
    if(not np.all(np.isfinite(rgbd))): # should not be triggered since values are ints now
      print('File '+posfilename+' contained NaN/inf, dropping...')
      return

    count = np.uint32(len(coordinateList))
    #if count > 1:
    #  print('Warning, image '+posfilename+' has more than one object already! skipping...') #TODO remove this when multi obj scans are available
    #  return
    if count < 1:
      #print('Warning, image '+posfilename+' has no object defined! skipping...') #TODO eventually we want to allow empty lists as well, but for now, it sorts out images we haven't tagged
      return

    if use_fake_distfunc:
      #mean = 0
      #standard_deviation = 10
      #valueAtZero = norm.pdf(0, mean, standard_deviation)
      fake = np.zeros((image_size[1], image_size[0]))
      fake.fill(classes['nothing']) #backdrop of "13"s - no obj here # TODO make it 0 at some point

      for v,ty,tx,_ in coordinateList:
        centerCoordinates = np.array([ty, tx])
        for y, x in np.ndindex(image_size[1], image_size[0]):
          distance = np.linalg.norm(centerCoordinates-np.array((y, x)))
          #value = (norm.pdf(distance, mean, standard_deviation) / valueAtZero) * 255
          if distance < 15: # allows other classes not to be overwritten
            fake[y, x] = v

      debugTargetOut = os.path.join(os.path.dirname(subdirpath), todirtarg)
      ensure_dir(debugTargetOut)
      debugTargetName = os.path.join(debugTargetOut, fname+toimgend)
      img = Image.fromarray(fake.astype(np.uint8))
      img.save(debugTargetName)

    coordsfile = open(posfilename,"w")
    #if(random() < 0.5):
    #  coordinateList.extend(coordinateList) #dirty hack to have more items in some cases, don't use - doesn't work on other end anyway, change to always X objects
    coordinateList = np.array(coordinateList)
    #print(coordinateList.shape) #check size
    np.save(coordsfile, coordinateList)

    img = Image.fromarray(rgbd.astype(np.uint8))
    img.save(output_filename)

    averagePixel = np.average(rgbd, axis=(0,1))
    return averagePixel

def handleListOfFiles(filesToProcess, resultqueue):
  result = {}
  totalNumber = 0
  summedPixel = np.zeros(num_color_channels)

  for [f, subdirpath, cls] in filesToProcess:
    meanPixel = handleSingleFile(f, subdirpath, cls)
    if (meanPixel is not None):
      totalNumber += 1
      summedPixel += meanPixel

  result['count'] = totalNumber
  result['summedAveragePixel'] = summedPixel

  resultqueue.put(result)

def chunker_list(seq, size):
  return list(seq[i::size] for i in range(size))

def generateListOfFiles(thepath):
  filesToProcess = []
  for fold in os.listdir(thepath):
    subdirpath = os.path.join(thepath, fold)
    if not os.path.isdir(subdirpath):
      continue
    if(fold == coords_folder or fold == todirtarg or fold == output_folder or fold == "MultipleDuplicateObjects" or fold == "NoObjects"): # TODO added the second 2 for now # and now removed -> or fold == "MultipleObjects" <-
      print('Skipping directory: ' + fold)
      continue

    print('Processing directory: ' + fold)

    if fold in classes:
      cls = np.uint8(classes[fold])
    else:
      cls = np.uint8(classes['unspecified'])

    folderToLoadImagesFrom = os.path.join(subdirpath, fromdirimg)

    for f in os.listdir(folderToLoadImagesFrom):
      filesToProcess.append([f, subdirpath, cls])
  return filesToProcess

def main():
  filesToProcess = []
  for folder in dirpath:
    filesToProcess.extend(generateListOfFiles(folder))

  numCores = multiprocessing.cpu_count()
  print "splitting up", len(filesToProcess), "files into", numCores, "work chunks"

  chunks = chunker_list(filesToProcess, numCores)
  threads = []
  results = [Queue() for x in range(numCores)]
  index = 0

  processedFiles = 0
  meanPixel = np.zeros(num_color_channels)

  for chunk in chunks:
    t = Process(target=handleListOfFiles, args=(chunk,results[index]))
    t.start()
    threads.append(t)
    index=index+1
    time.sleep(0.1) # just so not all threads access files synchronously

  print("joining in threads...")
  for x in threads:
     x.join()
  print("joined all threads.")

  for resultqueue in results:
    result = resultqueue.get()
    processedFiles += result['count']
    meanPixel += result['summedAveragePixel']
  meanPixel /= processedFiles

  print "out of", len(filesToProcess), "files,", processedFiles,"converted. (", len(filesToProcess)-processedFiles , "skipped)"
  print "average pixel:", meanPixel

if __name__ == "__main__":
  main()
