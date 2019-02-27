from __future__ import print_function
import sys
sys.dont_write_bytecode = True
import tensorflow as tf
import numpy as np
import Utils as utils
from classDefinitions import *
import downscaleImages

image_size = downscaleImages.image_size
NUM_CLASSES_AND_NOTHING = downscaleImages.NUM_CLASSES_AND_NOTHING

#TODO: testing something here, this should be off by 1, as 0 should not be in coords?
indexVectorArray = np.array([(y+1, x+1) for y, x in np.ndindex(image_size[1], image_size[0])]).reshape((image_size[1], image_size[0], 2))

PRINT_EVERYTHING = False

def distanceLoss(networkOutput, labelsAndIndex, evalMode=False):
  if (PRINT_EVERYTHING):
    averageInput = tf.reduce_mean(networkOutput)
    networkOutput = tf.Print(networkOutput,[averageInput], message="averageInput")
  labels = tf.slice(labelsAndIndex, [0,1], [BATCH_SIZE, 2])  # shape [BATCH_SIZE, 2]
  selectobjectLayerIndex = tf.ones([BATCH_SIZE], tf.int64) # always look at layer 1, 0 is not interesting (kept for cross entropy only)

  #reshaping and transposing for easier use
  reshapedResults = tf.reshape(networkOutput, [BATCH_SIZE, image_size[1], image_size[0], OUTPUT_LAYERS])
  transposed = tf.transpose(reshapedResults, [0, 3, 1, 2]) # to make it easier to select a
  batches = tf.cast(tf.range(BATCH_SIZE), tf.int64)

  final_idx = tf.reshape(tf.stack([batches, selectobjectLayerIndex], 1), [BATCH_SIZE, -1])
  heatmaps = tf.gather_nd(transposed, final_idx)

  #this holds all pixel coordinates for an image
  allpoints = tf.cast(tf.constant(indexVectorArray), tf.float32) # shape image_size[1], image_size[0], 2

  #for each image in batch, get the weighed centroid
  predictedPoints = []
  normalizedHeatmaps = []

  for i in range(0, BATCH_SIZE):
    normW = utils.normalizeWeights(heatmaps[i])
    normalizedHeatmaps.append(normW)
    if (PRINT_EVERYTHING):
      averageNormW = tf.reduce_mean(normW)
      normW = tf.Print(normW, [averageNormW], message="average of normW", summarize=BATCH_SIZE)

    predictedPoint = utils.weighedCenter(normW, allpoints)
    if (PRINT_EVERYTHING):
      predictedPoint = tf.Print(predictedPoint, [predictedPoint], message="predictedPoint", summarize=2*BATCH_SIZE)

    predictedPoints.append(predictedPoint)

  predictedPointsTensor = tf.stack(predictedPoints) # shape=(BATCH_SIZE, 2)

  if (PRINT_EVERYTHING):
    predictedPointsTensor = tf.Print(predictedPointsTensor, [predictedPointsTensor], message="predictedPointsTensor", summarize=2*BATCH_SIZE)

  #calculate the difference between predicted point and label
  distances = utils.calculateDistance(predictedPointsTensor, labels)

  if (PRINT_EVERYTHING):
    distances = tf.Print(distances, [distances], message="distances", summarize=BATCH_SIZE)

  #cleanup, remove NaN if in there
  cleanedDistances = utils.replaceNaN(distances, utils.maximumPossibleError()) # TODO test if required

  #before we average, we want to transform our values
  #fist, we transform them to fake centimeters:
  tempScaledError = utils.fakePixelDistInCm(cleanedDistances)

  #hyperbola:
  tempScaledError = -1/(tf.square(tempScaledError)+1) + 1

  pickup_error_mean = tf.reduce_mean(tempScaledError, name='averageBatchPickupError')
  error_mean  = tf.reduce_mean(cleanedDistances, name='averageBatchError')

  if(evalMode):
    normalizedHeatmapsTensor = tf.stack(normalizedHeatmaps)
    return(normalizedHeatmapsTensor, predictedPointsTensor, cleanedDistances, error_mean)

  return pickup_error_mean, error_mean


def vectorSpreadError(classificationLayer, keyObjectTypes):
  predictions = tf.squeeze(classificationLayer)

  crossCosineDistances = utils.cosineSimilaritiesFromCombinationOfVectors(predictions) # gets us a batch^2 matrix saying which vectors are really similar, and which ones aren't
  #crossCosineDistances = tf.Print(tf.reduce_mean(crossCosineDistances), [crossCosineDistances], message="crossCosineDistances")
  #keyObjectTypes are one-hot encoded... batch size x 13 (num objects)
  prod = tf.matmul(keyObjectTypes, keyObjectTypes, adjoint_b=True)  # multiply with itself, get batch^2 matrix saying: which images DO feature the same type? naturally, main diagonal is 1
  #prod = tf.Print(tf.reduce_mean(prod), [prod], message="prod")
  #prod is sort of our desired outcome: 1 (100%) similarity for all imgs with same obj, 0 for all those which don't

  difference = prod - crossCosineDistances #cosine dist can be -1..1 where -1 is furthest away from what we want (1) - add 1, div by 2 - same scale.
  #difference = tf.Print(tf.reduce_mean(difference), [difference], message="difference")
  #return tf.reduce_mean(tf.sqrt(tf.square(difference)))
  return tf.reduce_mean(difference*difference)


def binarizePredictionVector(classificationLayer): #aims to have all the values at 0 or 1
  predictions = tf.squeeze(classificationLayer)
  #predictions = tf.Print(predictions, [tf.reduce_mean(predictions)], message="predictions")

  errs0 = tf.abs(predictions) # distance to 0
  errs1 = tf.abs(1 - predictions) # distance to 1
  lowerErr = tf.minimum(errs0, errs1)

  return tf.reduce_mean(lowerErr, name='meanBinarizationError')

def ioULoss(networkOutput, desiredImage, labelsAndIndexToSelect):
  reshapedResults = tf.reshape(networkOutput, [BATCH_SIZE, image_size[1], image_size[0], OUTPUT_LAYERS])
  transposed = tf.transpose(reshapedResults, [0, 3, 1, 2])#transpose network output for easier indexing

  selectobjectLayerIndex = tf.ones([BATCH_SIZE], tf.int64) # always look at layer 1, 0 is not interesting (kept for cross entropy only)
  batches = tf.cast(tf.range(BATCH_SIZE), tf.int64)
  final_idx = tf.reshape(tf.stack([batches, selectobjectLayerIndex], 1), [BATCH_SIZE, -1])

  heatmap_raw = tf.gather_nd(transposed, final_idx) # shape (8, 192, 256)
  normalizedHeatmap = utils.normalizeWeightsExt(heatmap_raw) # TODO this was normalizeWeights for a long time, but really shouldn't ?? test later

  reshapedIdx = tf.cast(tf.reshape(labelsAndIndexToSelect, [BATCH_SIZE, 1, 1, NUM_CLASSES_AND_NOTHING]), tf.bool)
  oneHotObjectTypePerPixel = tf.cast(tf.one_hot(tf.cast(desiredImage, tf.int64), NUM_CLASSES_AND_NOTHING), tf.bool)

  binary_onehot_decision = tf.logical_and(oneHotObjectTypePerPixel, reshapedIdx) # select the pixels from each of the batches that are included in labelsAndIndexToSelect
  binary_decision = tf.reduce_sum(tf.cast(binary_onehot_decision, tf.float32), axis=-1)
  #binary_goal = tf.where(binary_decision, tf.ones_like(binary_decision, dtype=tf.float32), tf.zeros_like(binary_decision, dtype=tf.float32))

  '''
  now, logits is output with shape [batch_size x img h x img w x 1]
  and represents probability of class 1
  '''
  logits=tf.reshape(normalizedHeatmap, [-1])
  trn_labels=tf.reshape(binary_decision, [-1])

  '''
  Eq. (1) The intersection part - tf.mul is element-wise,
  if logits were also binary then tf.reduce_sum would be like a bitcount here.
  '''
  inter=tf.reduce_sum(tf.multiply(logits,trn_labels))

  '''
  Eq. (2) The union part - element-wise sum and multiplication, then vector sum
  '''
  union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)))

  # Eq. (4)
  loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
  return loss

def realIoU(networkOutput, desiredImage, labelsAndIndex):
  reshapedResults = tf.reshape(networkOutput, [BATCH_SIZE, image_size[1], image_size[0], OUTPUT_LAYERS])
  selectobjectLayerIndex = tf.ones([BATCH_SIZE], tf.int64) # always look at layer 1, 0 is not interesting (kept for cross entropy only)
  transposed = tf.transpose(reshapedResults, [0, 3, 1, 2])
  batches = tf.cast(tf.range(BATCH_SIZE), tf.int64)
  final_idx = tf.reshape(tf.stack([batches, selectobjectLayerIndex], 1), [BATCH_SIZE, -1])
  heatmap_raw = tf.gather_nd(transposed, final_idx) # shape (8, 192, 256)
  normalizedHeatmap = utils.normalizeWeights(heatmap_raw)

  binary_heatmap = tf.where(tf.greater(normalizedHeatmap, tf.ones_like(heatmap_raw)/2), tf.ones_like(normalizedHeatmap, dtype=tf.int64), tf.zeros_like(normalizedHeatmap, dtype=tf.int64))

  reshapedIdx = tf.reshape(selectobjectLayerIndex, [BATCH_SIZE, 1, 1])
  binary_goal = tf.cast(tf.equal(tf.cast(desiredImage, tf.int64), reshapedIdx), tf.int64) # select the pixels from each of the batches frame that corresponds to the index in selectobjectLayerIndex (1: have the object num, 0: no obj)

  math_sum = binary_heatmap + binary_goal # now the intersection will be 2, and the "xor" region 1. union = int+xor

  #go on and count 2s and 1s (and 0s)
  onehot = tf.one_hot(tf.reshape(math_sum, [BATCH_SIZE, -1]), 3)
  counts=tf.reduce_sum(onehot, axis=1) # sum up over all votes
  intersection = counts[:, 2]
  xor = counts[:, 1]
  union = intersection+xor
  iou = intersection / union
  mean_iou_loss = 1-tf.reduce_mean(iou)

  return mean_iou_loss
