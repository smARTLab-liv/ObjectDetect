import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
import math
import downscaleImages
import random
from classDefinitions import *

image_size = downscaleImages.image_size

def cosineSimilaritiesFromCombinationOfVectors(inputVectors): # input is shape (Batch, VectorLength)
  normalized = tf.nn.l2_normalize(inputVectors, dim = 1)
  # multiply row i with row j using transpose
  # element wise product
  sim = tf.matmul(normalized, normalized,
                   adjoint_b = True # transpose second matrix
                   )
  return sim


def he_initializer(shape):
    fan_in = shape[-2]
    fan_out = shape[-1]
    n = (fan_in + fan_out)/2.0
    factor = 2
    initial = tf.truncated_normal(shape, stddev=math.sqrt(factor / n))
    return initial

def weight_variable(shape, stddev=0.02, name=None):
    #initial = tf.truncated_normal(shape, stddev=stddev)
    #He initialization
    initial = he_initializer(shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var

def process_image(image, mean_pixel):
    return image - mean_pixel

def unprocess_image(image, mean_pixel):
    return image + mean_pixel

def conv2d_basic(x, W, bias, pad = "SAME"):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=pad)
    return tf.nn.bias_add(conv, bias)

def conv2d_transpose_strided(x, W, b, output_shape=None, stride = (2,2), pad="SAME"):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride[0], stride[1], 1], padding=pad)
    return tf.nn.bias_add(conv, b)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def leaky_relu(x, alpha=0.02, name="leaky_relu"):
  #return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
  return tf.maximum(x, alpha * x, name=name)

def save_image(image, save_dir, name, mean=None):
    """
    Save image by unprocessing if mean given else just save
    :param mean:
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)

def normalizeWeights(x):
  current_min = tf.reduce_min(x)
  current_max = tf.reduce_max(x)
  x = (x - current_min) / (current_max - current_min)
  return replaceNaN(x, 1.0)

def normalizeWeightsExt(x):
  normalizedHeatmaps = []
  for i in range(0, BATCH_SIZE):
    normalizedHeatmap = normalizeWeights(x[i])
    normalizedHeatmaps.append(normalizedHeatmap)
  normalizedHeatmapsTensor = tf.stack(normalizedHeatmaps)
  return normalizedHeatmapsTensor

def weighedCenter(weights, points):
  weights_expanded = tf.expand_dims(weights, 2)
  weighedPoints = tf.multiply(weights_expanded, points)
  summedPoints = tf.reduce_sum(weighedPoints, axis=[0, 1])
  #todo if we want to strenghten the influence of the weights, do sth here with weights, before adding them up. square?
  summedWeights = tf.reduce_sum(weights, axis=[0, 1])
  mean = tf.divide(summedPoints, summedWeights)
  return mean

def replaceNaN(tensor, replaceBy):
  return tf.where(tf.is_nan(tensor), tf.ones_like(tensor) * replaceBy, tensor)

def calculateSquaredDistance(vfrom, vto, reductionDimension = 1):
  difference = tf.subtract(vfrom, vto)
  squaredElements = tf.square(difference)
  squaredDistances = tf.reduce_sum(squaredElements, reductionDimension)
  return squaredDistances

def calculateDistance(vfrom, vto, reductionDimension = 1):
  squaredDistances = calculateSquaredDistance(vfrom, vto)
  distances = tf.sqrt(squaredDistances)
  return distances

def weighedVariance(weights, points, centroid): # shapes (image_size[1], image_size[0]), (image_size[1], image_size[0], 2),  (2)
  distances = calculateSquaredDistance(points, centroid, reductionDimension=2)# shape (image_size[1], image_size[0])
  weighted_distances = tf.multiply(distances, weights)
  wVariance = tf.reduce_mean(weighted_distances) # shape (1)
  return wVariance

#TODO add ignoreObjectType(s) here and below
def read_dataset(data_dir, valid_dir, HOLDOUT_OBJECT_DURING_TRAINING):
  #filesToTrain = read_folder(data_dir)
  #filesToValidate = read_folder(valid_dir)

  filesToTrain, filesToValidate = read_folder(data_dir, valid_dir, HOLDOUT_OBJECT_DURING_TRAINING)

  return filesToTrain, filesToValidate

#TODO load these from downscaleImages
image_subdir = "merged_out"
coord_subdir = "coords"
target_subdir = "target"

def read_folder(folder1, folder2, HOLDOUT_OBJECT_DURING_TRAINING):

  HOLDOUT_OBJECT_DURING_TRAINING = None #TODO CHANGE THIS

  results = list()
  for folder in [folder1, folder2]:
    inputdir = os.path.join(folder, image_subdir)
    coorddir = os.path.join(folder, coord_subdir)
    targetdir = os.path.join(folder, target_subdir)

    for f in os.listdir(inputdir):
      filename = os.path.splitext(os.path.basename(f))[0]
      image = os.path.join(inputdir, f)

      annotation = os.path.join(targetdir, filename+".png")
      coordfile = os.path.join(coorddir, filename+".npy")

      if (os.path.isfile(image) and os.path.isfile(annotation) and os.path.isfile(coordfile)):
        coordlist = np.load(coordfile)
        entry = dict([('image', image), ('annotation', annotation), ('filename', filename), ('coordsfile', coordfile), ('coords', coordlist)])
        results.append(entry)

  #we now have both training and validation set files in results
  random.Random(14).shuffle(results)
  #repeat - "pre-peat" the split for multi/single objects, to make sure enough multi obj files land in both
  multipleObjs = list()
  singleObjs = list()

  for item in results:
    if len(item['coords']) > 1:
      multipleObjs.append(item)
    else:
      singleObjs.append(item)

  newTrainSet = list()
  newValidSet = list()

  for objList in [multipleObjs, singleObjs]:
    for i,elem in enumerate(objList):
      if HOLDOUT_OBJECT_DURING_TRAINING is None:
        transverseFlipped = zip(*elem['coords'])
        numDifferentObjects = len(set(transverseFlipped[0]))
        if(numDifferentObjects == 2): # all the 2 object images go in test set
          newValidSet.append(elem)
        elif numDifferentObjects == 1 and i % 8 == 0: # 1/8 of all 1 object images goes in test set
          newValidSet.append(elem)
        else:
          newTrainSet.append(elem) # all the rest (multi-obj and 7/8 of 1-object iamges go in training

      else: # HOLDOUT_OBJECT_DURING_TRAINING was set, so those objects are validation, rest ist training set
        transverseFlipped = zip(*elem['coords'])
        if(set(HOLDOUT_OBJECT_DURING_TRAINING) == set(transverseFlipped[0])): # this requires that ONLY the holdout objects are tagged, probably won't have enough samples
        #if(any((True for x in HOLDOUT_OBJECT_DURING_TRAINING if x in transverseFlipped[0]))): # if any (allows all) of HOLDOUT_OBJECT_DURING_TRAINING are present in the image...
          newValidSet.append(elem)
        elif len(set(transverseFlipped[0])) == 1 and any((True for x in HOLDOUT_OBJECT_DURING_TRAINING if x in transverseFlipped[0])): # single item images, in which the object is part of the holdout set
          newValidSet.append(elem)
        elif(not any((True for x in HOLDOUT_OBJECT_DURING_TRAINING if x in transverseFlipped[0]))): # hold out all holdout objects from training set
          newTrainSet.append(elem)

  print "new training set has", len(newTrainSet), "input images"
  print "new validation set has", len(newValidSet), "input images"

  filesToTrain = createDupletCombinations(newTrainSet)
  filesToValidate = createDupletCombinations(newValidSet)

  return filesToTrain, filesToValidate

def createDupletCombinations(filelist):
  result = list()

  multipleObjs = list()
  singleObjs = list()

  for item in filelist:
    if len(item['coords']) > 1:
      multipleObjs.append(item)
    else:
      singleObjs.append(item)
  print("multiple object set:", len(multipleObjs))
  print("single object set:", len(singleObjs))

  #we want to pad our multi-object dataset by a certain ratio, so we move x random single obj items into the multi obj set

  random.Random(18).shuffle(singleObjs)

  desiredNumObjs = 0 * len(multipleObjs) # TODO just removed all "single image" additions to multi image set. (could artificially lower the error...)
  if desiredNumObjs > len(singleObjs):
    desiredNumObjs = len(singleObjs) // 2

  multipleObjs.extend(singleObjs[:desiredNumObjs])
  singleObjs = singleObjs[desiredNumObjs:]

  print("after adjustment, multiple object set:", len(multipleObjs))
  print("after adjustment, single object set:", len(singleObjs))

  KEY_IMAGES_PER_DETECTION_IMAGE = 50
  print("Finding matching multi obj/single object pairs")
  for multiObject in multipleObjs:
    for specificObjType in multiObject['coords']: # for each object type featured in this image, search for 2 single image instances:
      #if specificObjType[0] not in [0., 1., 3., 4., 5., 8.]: # TODO remove, reduces the whole dataset to only specific object types
      #  continue
      numAdded = 0
      for singleObject in singleObjs:
        if(singleObject['coords'][0][0] == specificObjType[0] and numAdded < KEY_IMAGES_PER_DETECTION_IMAGE):
          result.append(dict([('image', multiObject['image']), ('annotation', multiObject['annotation']), ('filename', multiObject['filename']), ('coords', multiObject['coords']), #('coordsfile', multiObject['coordsfile']),
                              ('keyImage', singleObject['image']), ('keyCoords', singleObject['coords'][0]), ('keyAnnotation', singleObject['annotation']), ('type', specificObjType[0])]))
          numAdded += 1

  print("found object pairs:", len(result))
  return result

def pixelDistInCm(distance, depth):
  #that is distance in pixels.
  #fov in sr300 is 68 x 54 degs, i.e., one pixel varies...
  #downscaling brings us to 256 x 192 = 0.2734375 degrees per pixel, = 0.00477238467341 rads per pixel
  angularDistance = distance * 0.00477238467341
  #now, using the depth measure we have, and the fact that tan(alpha) = adjacent cathet. / x, where alpha is our angular distance, we get...
  distanceInCm = depth * np.tan(angularDistance)
  return distanceInCm

def fakePixelDistInCm(distance):
  #that is distance in pixels.
  #based on 16k measurements of pixel/cm readings, this is the best fit:
  #(on the validation set, that is - 10 cm platform. might vary wildly on training set!)
  return 0.1927323 * distance - 0.114962

def unfakePixelDistInCm(distanceinCm):
  return (distanceinCm+0.114962)/0.1927323

def maximumPossibleError():
  return np.linalg.norm(np.array((image_size[1], image_size[0])))

def printLossValues(itr, losses):
  mess = "At: "+itr+"\n"
  for loss in losses:
    mess += "Loss value: "+str(loss)+"\n"
  print(mess)

def EMAof(var):
  ema = tf.train.ExponentialMovingAverage(decay=0.9, zero_debias=False) # not zero-debiasing apparently is better...?
  m = ema.apply([var])
  av = ema.average(var)
  [retval] = tf.tuple([av], control_inputs=[m])
  return retval
