from __future__ import print_function
import sys
sys.dont_write_bytecode = True
import tensorflow as tf
import numpy as np
import os
import Utils as utils
import datetime
import FileQueue as dataset
from six.moves import xrange
import downscaleImages
from classDefinitions import *
from distanceFunctions import *


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")

tf.flags.DEFINE_string("data_dir", "trainingData/allData/", "path to dataset")
tf.flags.DEFINE_string("valid_dir", "newEvaluationSet/", "path to evaluation dataset")

#tf.flags.DEFINE_string("data_dir", "trainingData/testMultiObj/", "path to dataset")
#tf.flags.DEFINE_string("valid_dir", "trainingData/testMultiObj/", "path to evaluation dataset")

#tf.flags.DEFINE_string("data_dir", "trainingData/testMultiObj2/", "path to dataset")
#tf.flags.DEFINE_string("valid_dir", "trainingData/testMultiObj2_valid/", "path to evaluation dataset")

tf.flags.DEFINE_float("learning_rate", "1e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('logProgress', "False", "evaluates every 1000 steps or so on the entire eval set, writes the accurracy into a file")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(100001)

image_size = downscaleImages.image_size
num_color_channels = downscaleImages.num_color_channels

learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
MID_LAYER_NAME = 'norm8'

def vgg_net(image, keyimage, keep_prob, useSelectionFilter=True):
    layers = []
    #TODO add more dropout layers
    #TODO original authors also used max pooling instead of avg

    #image is 192*256*4 = 196608 (~200k)

    #layer 1 uses 192*256*64 = 3145728 (3.14m)
    layers.extend(['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2'])
    if(useSelectionFilter):
        layers.extend(['sele1'])
    layers.extend(['avpl1'])

    #layer 2 uses 96*128*128 = 1572864 (1.57m)
    layers.extend(['conv2_1', 'relu2_1', 'conv2_2', 'relu2_2'])
    #if(useSelectionFilter):
    #    layers.extend(['sele2'])
    layers.extend(['avpl2'])

    #layer 3 uses 48*64*256 = 786432 (~800k)
    layers.extend(['conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4'])
    #if(useSelectionFilter):
    #    layers.extend(['sele3'])
    layers.extend(['avpl3'])

    #layer 4 uses 24*32*512 = 393216 (~300k)
    layers.extend(['conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4'])
    #if(useSelectionFilter):
    #    layers.extend(['sele4'])
    layers.extend(['avpl4'])

    #layer 5 uses 6*8*1024 = 49152 = (50k)
    layers.extend(['conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'])
    #if(useSelectionFilter):
    #    layers.extend(['sele5'])
    layers.extend(['pool5'])

    #layer 6 uses 6*8*2048 = 196608 (200k) # sudden incline - weird?
    #layers.extend(['conv6_1', 'relu6_1'])
    layers.extend(['conv6_1', 'relu6_1', 'conv6_2', 'relu6_2']) # one more conv/relu?
    #if(useSelectionFilter):
    #    layers.extend(['sele6'])
    layers.extend(['drop6'])
    layers.extend(['pool6'])#TODO actually, should be before dropout

    #layer 7 scales down to 1*1*4096 = 4096
    layers.extend(['conp7_0', 'relu7_0'])
    layers.extend(['conv7_1', 'relu7_1'])
    layers.extend(['full8_1', 'relu8_1', 'full8_2', 'relu8_2', 'norm8'])


    #layers.extend(['decp0'])
    #layers.extend(['decv1']) # scales up to 6*8*1024 = 49152 (50k) # is still 4k
    layers.extend(['decv2']) # scales up to 12*16*512 = 98304 (100k)
    layers.extend(['decv3']) # scales up to 24*32*256 = 196608 (200k)
    layers.extend(['decv4']) # scales up to 192*256*2 = 98304 (100k) (but technically, only half of the data/one layer is used.)


    kernels = {'conv1_1':[3, 3, 4, 64],
               'conv1_2':[3, 3, 64, 64],
               'conv2_1':[3, 3, 128, 128],
               'conv2_2':[3, 3, 128, 128],
               'conv3_1':[3, 3, 128, 256],
               'conv3_2':[3, 3, 256, 256],
               'conv3_3':[3, 3, 256, 256],
               'conv3_4':[3, 3, 256, 256],
               'conv4_1':[3, 3, 256, 512],
               'conv4_2':[3, 3, 512, 512],
               'conv4_3':[3, 3, 512, 512],
               'conv4_4':[3, 3, 512, 1024],
               'conv5_1':[3, 3, 1024, 1024],
               'conv5_2':[3, 3, 1024, 1024],
               'conv5_3':[3, 3, 1024, 1024],
               'conv5_4':[3, 3, 1024, 1024],
               'conv6_1':[3, 3, 1024, 2048],
               'conv6_2':[3, 3, 2048, 2048],
               'conp7_0':[3, 4, 2048, 4096],
               'conv7_1':[1, 1, 4096, 4096],
               'full8_1':[4096, 4096],
               'full8_2':[4096, 4096],
               'decp0':[3, 4, 2048, 4096],
               'decv1':[4, 4, 1024, 2048],
               'decv2':[4, 4, 1024, 2048],
               'decv3':[4, 4, 256, 1024],
               'decv4':[16, 16, OUTPUT_LAYERS, 256]}

    deconvolution_conf = {'decp0':((3,4), None, 'pool6', [BATCH_SIZE, 3, 4, 2048], True),
                          'decv1':((2,2), None, 'pool5', [BATCH_SIZE, 6, 8, 1024], True),
                          'decv2':((2,2), 'drop6', 'avpl4', [BATCH_SIZE, 12, 16, 1024], True),
                          'decv3':((2,2), None, 'avpl3', [BATCH_SIZE, 24, 32, 256], True),
                          'decv4':((8,8), None, 'output',[BATCH_SIZE, 192, 256, OUTPUT_LAYERS], False)}

    selectorWeightsIdx = {'sele1':(0, 64),
                          'sele2':(64, 192),
                          'sele3':(192, 448),
                          'sele4':(448, 960),
                          'sele5':(960, 1984),
                          'sele6':(1984, 4032)}

    net = {'input':keyimage}
    refinednet = {'input':image}

    networks = [net, refinednet]

    for idx, network in enumerate(networks):
        current = network['input']

        with tf.variable_scope('networks') as scope:
          #if(idx > 0):
          #  scope.reuse_variables() # TODO investigate where best
          #print("00", "input")
          #print(network['input'])
          for i, name in enumerate(layers):
              if(idx == 0 and i > 3): # hacky, stop first network from ever getting beyond first layer
                break
              #print(i, name)
              kind = name[:4]
              if kind in ['conv', 'conp']:
                  padding = 'VALID' if kind == 'conp' else 'SAME'

                  numOutputFilters = kernels[name][3]
                  kernel = utils.weight_variable(kernels[name], name=name+str(idx)+"_weights")
                  biases = utils.bias_variable([numOutputFilters], name=name+str(idx)+"_biases")
                  current = utils.conv2d_basic(current, kernel, biases, pad=padding)
              elif kind == 'relu':
                  current = utils.leaky_relu(current, name=name+str(idx))
              elif kind == 'avpl':
                  current = utils.avg_pool_2x2(current)
              elif kind == 'pool':
                  current = utils.max_pool_2x2(current)
              elif kind == 'drop':
                  current = tf.nn.dropout(current, keep_prob=keep_prob)
              elif kind == 'full':
                  kernelshape = kernels[name]
                  weights = utils.weight_variable(kernelshape, name=name+"_weights")
                  biases = utils.bias_variable([kernelshape[1]], name=name+"_biases")
                  reshaped = tf.reshape(current, [BATCH_SIZE, -1])
                  current = tf.matmul(reshaped, weights) + biases
                  current = tf.reshape(current, [BATCH_SIZE,1,1,-1]) # shaped back
              elif kind == 'sele' and idx > 0: #only refined network gets selector weights
                  #classificationOutput = net[MID_LAYER_NAME] # TODO was 7_1
                  #lower, upper = selectorWeightsIdx[name];
                  #selectorWeights = classificationOutput[:,:,:,lower:upper]
                  #current = selectorWeights * current
                  current = tf.concat([current, net["relu1_2"]], 3)
              elif kind == 'norm':
                  current = tf.nn.l2_normalize(current, dim=-1,  name=name+"_normalized")
              elif kind in ['decv', 'decp']:
                  padding = 'VALID' if kind == 'decp' else 'SAME'
                  stride, inputLayerName, mergeLayer, target_shape, shouldFuse = deconvolution_conf[name]
                  kernelshape = kernels[name]
                  inputLayer = network[inputLayerName] if inputLayerName is not None else current
                  kernel = utils.weight_variable(kernelshape, name=name+"_weights")
                  biases = utils.bias_variable([kernelshape[2]], name=name+"_biases")
                  current = utils.conv2d_transpose_strided(inputLayer, kernel, biases, output_shape=target_shape, stride=stride, pad=padding)

                  if shouldFuse:
                    current = tf.add(current, network[mergeLayer], name=name+"_fuse")

              #print(current)
              network[name] = current
              network['lastLayer'] = current

    return net, refinednet


def inference(image, keyimage, keep_prob):
    mean_pixel = tf.constant([120.4281724, 121.60578141, 118.14762266, 172.40523575])
    #measured from 11951 of the 31713 input files: 120.4281724   121.60578141  118.14762266  172.40523575

    processed_image = utils.process_image(image, mean_pixel)
    processed_keyimage = utils.process_image(keyimage, mean_pixel)

    with tf.variable_scope("inference"):
        image_net, refined_image_net = vgg_net(processed_image, processed_keyimage, keep_prob)
        network_output = refined_image_net['lastLayer']
        #key_network_output = image_net['lastLayer']
        key_network_output = refined_image_net['lastLayer']

        #classificationLayerKey = image_net[MID_LAYER_NAME]
        classificationLayerKey = refined_image_net[MID_LAYER_NAME]
        classificationLayer = refined_image_net[MID_LAYER_NAME]

    return utils.normalizeWeightsExt(network_output), network_output, utils.normalizeWeightsExt(key_network_output), key_network_output, classificationLayerKey, classificationLayer


def allLossValues(logits, labels, coords, logits2, keylabels, keycoords, classificationLayer, classificationLayerKey):

  meanCrossEntropyLoss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="entropy")))
  meanCrossEntropyKeyLoss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=keylabels, name="entropyKey")))

  meanPickupError, meanDistanceError = distanceLoss(logits, coords)
  meanKeyPickupError, meanKeyDistanceError = distanceLoss(logits2, keycoords)
  scaledMeanDistanceError = meanDistanceError # / 100 # very rough scaling to 0..1
  scaledMeanKeyDistanceError = meanKeyDistanceError # / 100 # very rough scaling to 0..1

  meanIoULoss = ioULoss(logits, labels, coords)
  meanKeyIoULoss = ioULoss(logits2, keylabels, keycoords)

  #meanCrossEntropyLossAverage = utils.EMAof(meanCrossEntropyLoss)
  #meanDistanceErrorAverage = utils.EMAof(scaledMeanDistanceError)
  #meanPickupErrorAverage = utils.EMAof(meanPickupError)
  #meanIoULossAverage = utils.EMAof(meanIoULoss)

  #meanCrossEntropyLossVariance = utils.EMAof(tf.square(meanCrossEntropyLossAverage-meanCrossEntropyLoss))
  #meanDistanceErrorVariance = utils.EMAof(tf.square(meanDistanceErrorAverage-scaledMeanDistanceError))
  #meanPickupErrorVariance = utils.EMAof(tf.square(meanPickupErrorAverage-meanPickupError))
  #meanIoULossVariance = utils.EMAof(tf.square(meanIoULossAverage-meanIoULoss))


  # first row is for debug and plotting, second row is actually used as error, 3rd and 4rth are average/variance, currently unused
  # TODO second can be meanDistanceError->meanPickupError for pickup error
  #return [meanCrossEntropyLoss, meanDistanceError, meanIoULoss],\
  #       [meanCrossEntropyLoss, scaledMeanDistanceError, meanIoULoss],\
  #       [meanCrossEntropyLossAverage, meanDistanceErrorAverage, meanIoULossAverage],\
  #       [meanCrossEntropyLossVariance, meanDistanceErrorVariance, meanIoULossVariance]
  return [meanCrossEntropyLoss, meanDistanceError, meanIoULoss, meanCrossEntropyKeyLoss, meanKeyDistanceError, meanKeyIoULoss],\
         [meanCrossEntropyLoss, scaledMeanDistanceError, meanIoULoss, meanCrossEntropyKeyLoss, scaledMeanKeyDistanceError, meanKeyIoULoss]


def loss_function_selector(logits, labels, coords, logits2, keylabels, keycoords, classificationLayer, classificationLayerKey, iteration):
  _, [meanCrossEntropyLoss, scaledMeanDistanceError, meanIoULoss, meanCrossEntropyKeyLoss, scaledMeanKeyDistanceError, meanKeyIoULoss] = allLossValues(logits, labels, coords, logits2, keylabels, keycoords, classificationLayer, classificationLayerKey)

  #return [[scaledMeanDistanceError+scaledMeanKeyDistanceError, tf.constant(1.0)]]
  return [[scaledMeanDistanceError, tf.constant(1.0)]]


def train(losses_val, var_list):
    returns = []
    for loss_val, loss_weight in losses_val:
      optimizer = tf.train.AdamOptimizer(learning_rate_placeholder * loss_weight)
      grads = optimizer.compute_gradients(loss_val, var_list=var_list)
      returns.append(optimizer.apply_gradients(grads))
    return returns

def main(argv=None):
    #placeholders
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    iteration = tf.placeholder(tf.int64, name="iteration")

    image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_size[1], image_size[0], num_color_channels], name="input_image")
    keyimage = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_size[1], image_size[0], num_color_channels], name="key_image")

    annotation = tf.placeholder(tf.int32, shape=[BATCH_SIZE, image_size[1], image_size[0], 1], name="annotation")
    keyannotation = tf.placeholder(tf.int32, shape=[BATCH_SIZE, image_size[1], image_size[0], 1], name="keyannotation")
    labels = tf.squeeze(annotation, squeeze_dims=[3])
    keylabels = tf.squeeze(keyannotation, squeeze_dims=[3])

    coords = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 4], name="coordinates")
    keycoords = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 4], name="keycoordinates")

    norm_networkOutput, logits, norm_networkOutput2, logits2, classificationLayerKey, classificationLayer = inference(image, keyimage, keep_probability)

    allErrorMeasures = distanceLoss(logits, coords, evalMode=True)
    pickupError, distanceError = distanceLoss(logits, coords)


    allLosses, _ = allLossValues(logits, labels, coords, logits2, keylabels, keycoords, classificationLayer, classificationLayerKey)
    losses = loss_function_selector(logits, labels, coords, logits2, keylabels, keycoords, classificationLayer, classificationLayerKey, iteration)

    realIouValue = realIoU(logits, labels, coords)

    trainable_vars = tf.trainable_variables()
    train_op = train(losses, trainable_vars)


    print("Reading dataset dimensions...")
    train_records, valid_records = utils.read_dataset(FLAGS.data_dir, FLAGS.valid_dir)

    print(str(len(train_records)) + " records for training")
    print(str(len(valid_records)) +  " records for validation")
    if len(valid_records) == 0  or len(train_records) == 0:
      print("insufficient training or validation data")
      raise SystemExit


    print("Loading entire dataset to memory...")
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.FileQueue(train_records)
    validation_dataset_reader = dataset.FileQueue(valid_records)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        printLossEvery = 10
        printEvalLossEvery = 200
        saveModelEvery = 500
        runCompleteEvalEvery = 500 if FLAGS.logProgress else MAX_ITERATION

        if(FLAGS.logProgress):
          accFile = open('progressWhileTraining.csv', 'w')
          accFile.write("step,CEBatch,CELabelBatch,iouBatch,distanceBatch,CEEval,CELabelEval,classificationAccuracyVote,classificationAccuracyLayer,iouTrainEval,iouRealEval,distancePix,distanceCm\n")

        for itr in xrange(1, MAX_ITERATION):
            (train_images, train_annotations, train_coords, train_keyimages, train_keyannotations, train_keycoords) = train_dataset_reader.next_batch(BATCH_SIZE)

            feed_dict = {image: train_images, keyimage:train_keyimages, annotation: train_annotations, keyannotation:train_keyannotations, keycoords:train_keycoords, keep_probability: 0.85, coords: train_coords, iteration: int(itr), learning_rate_placeholder: FLAGS.learning_rate}
            sess.run(train_op, feed_dict=feed_dict)

            if itr % printLossEvery == 0:
                train_losses, train_error_mean = sess.run([losses, distanceError], feed_dict=feed_dict)
                if(np.any(np.isnan(np.array(train_losses)))):
                  print("Stopping - NaN loss.")
                  raise SystemExit
                utils.printLossValues(str(itr), train_losses)

            if itr % printEvalLossEvery == 0 and not FLAGS.logProgress:
                (valid_images, valid_annotations, valid_coords, valid_keyimages, valid_keyannotations, valid_keycoords) = validation_dataset_reader.next_batch(BATCH_SIZE)
                feed_dict = {image: valid_images, keyimage:valid_keyimages, annotation: valid_annotations, keyannotation:valid_keyannotations, keycoords:valid_keycoords, coords: valid_coords, keep_probability: 1.0, iteration: int(itr), learning_rate_placeholder: FLAGS.learning_rate}
                valid_losses, valid_error_mean = sess.run([losses, distanceError], feed_dict=feed_dict)
                utils.printLossValues(str(datetime.datetime.now()), valid_losses)

            if itr % saveModelEvery == 0:
              saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

            if itr % runCompleteEvalEvery == 0 and FLAGS.logProgress:
              #get the losses from the current batch
              trainCELoss, trainDistance, trainIoU, trainKeyCELoss, trainKeyDistance, trainKeyIoU = sess.run(allLosses, feed_dict=feed_dict)

              #now, run eval on the complete eval set:

              countElements = 0
              #print("just some test.. count!"+str(validation_dataset_reader.epochs_completed))
              sumVoteCorrect = 0
              sumLayerCorrect = 0
              sumDistance = 0.0
              sumPixDistance = 0.0
              sumevalCELoss = 0.0
              sumevalCELabelLoss = 0.0
              sumevalIoU = 0.0
              sumrealEvalIoU = 0.0

              while validation_dataset_reader.epochs_completed < 1:
                #print(str(validation_dataset_reader.epochs_completed))
                (valid_images, valid_annotations, valid_coords, valid_keyimages, valid_keyannotations, valid_keycoords) = validation_dataset_reader.next_batch(BATCH_SIZE)
                feed_dict = {image: valid_images, keyimage:valid_keyimages, annotation: valid_annotations, keyannotation:valid_keyannotations, keycoords:valid_keycoords, coords: valid_coords, keep_probability: 1.0, iteration: int(itr), learning_rate_placeholder: FLAGS.learning_rate}
                (evalCELoss, evalDistance, evalIoU, evalKeyCELoss, evalKeyDistance, evalKeyIoU), (normalizedHeatmapsTensor, predictedPointsTensor, tempSelectedError, error_mean),realEvalIoU = sess.run([allLosses, allErrorMeasures, realIouValue], feed_dict=feed_dict)
                #print(str(validation_dataset_reader.epochs_completed))
                for itr2 in range(BATCH_SIZE):
                  countElements +=1
                  #add up directly obtained error measures:
                  sumevalCELoss += evalCELoss
                  sumPixDistance += evalDistance
                  sumevalIoU += evalIoU
                  sumrealEvalIoU += realEvalIoU

                  #calculate average distance:
                  labelCoords = valid_coords[itr2][1:3]
                  labelAvgDepth = valid_coords[itr2][3]
                  classifiedCoords = predictedPointsTensor[itr2]

                  difference = labelCoords-classifiedCoords
                  pixDistance = np.linalg.norm(difference)
                  distance = utils.pixelDistInCm(pixDistance, labelAvgDepth)
                  sumDistance += distance

              validation_dataset_reader.epochs_completed = 0

              #step,CEBatch,CELabelBatch,iouBatch,distanceBatch,   CEEval,CELabelEval,classificationAccuracyVote,classificationAccuracyLayer,iouTrainEval,iouRealEval,distancePix,distanceCm
              step = str(itr)
              CEBatch = str(trainCELoss)
              iouBatch = str(trainIoU)
              distanceBatch= str(trainDistance)
              if countElements == 0:
                countElements = 1
              #print("done some tests... "+str(validation_dataset_reader.epochs_completed))
              CEEval = str(sumevalCELoss / countElements)
              classificationAccuracyVote = str(float(sumVoteCorrect)/countElements)
              classificationAccuracyLayer = str(float(sumLayerCorrect)/countElements)
              iouTrainEval = str(sumevalIoU / countElements)
              iouRealEval = str(sumrealEvalIoU / countElements)

              distancePix = str(sumPixDistance / countElements)
              distanceCm = str(sumDistance / countElements)

              accFile.write(step+','+CEBatch+','+'0'+','+iouBatch+','+distanceBatch+','+CEEval+','+'0'+','+classificationAccuracyVote+','+classificationAccuracyLayer+','+iouTrainEval+','+iouRealEval+','+distancePix+','+distanceCm+'\n')
              accFile.flush()
              print("Complete Validation Set avg distance:"+distanceCm)

    elif FLAGS.mode == "visualize":
        (valid_images, valid_annotations, valid_coords, valid_keyimages, valid_keyannotations, valid_keycoords) = validation_dataset_reader.get_random_batch(BATCH_SIZE)

        pred, predk, classLayerKey, classLayer, ( normalizedHeatmapsTensor, predictedPointsTensor, tempSelectedError, error_mean) = sess.run([norm_networkOutput, norm_networkOutput2, classificationLayerKey, classificationLayer, allErrorMeasures], feed_dict={image: valid_images, keyimage:valid_keyimages, annotation: valid_annotations, keyannotation:valid_keyannotations, keycoords:valid_keycoords, coords: valid_coords, keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)

        for itr in range(BATCH_SIZE):
          utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="input_" + str(itr))
          valid_annots = valid_annotations[itr] * 255.0
          utils.save_image(valid_annots.astype(np.uint8), FLAGS.logs_dir, name="goal_" + str(itr))

          utils.save_image(valid_keyimages[itr].astype(np.uint8), FLAGS.logs_dir, name="keyinput_" + str(itr))

          leslice1 = pred[itr,:, :,0] * 255.0
          leslice2 = pred[itr,:, :,1] * 255.0
          #utils.save_image(leslice1.astype(np.uint8), FLAGS.logs_dir, name="normLayer0_" + str(itr))
          utils.save_image(leslice2.astype(np.uint8), FLAGS.logs_dir, name="normLayer1_" + str(itr))

          leslice1 = predk[itr,:, :,0] * 255.0
          leslice2 = predk[itr,:, :,1] * 255.0
          #utils.save_image(leslice1.astype(np.uint8), FLAGS.logs_dir, name="keynormLayer0_" + str(itr))
          utils.save_image(leslice2.astype(np.uint8), FLAGS.logs_dir, name="keynormLayer1_" + str(itr))


          print(predictedPointsTensor[itr])
          print(tempSelectedError[itr])
          #print('results of innermost "classification" layer: (key image)')
          #print(classLayerKey)
          #print('results of innermost "classification" layer: (multiobj image, selec-thresholded)')
          #print(classLayer)
          print("Saved image: %d" % itr)

    elif FLAGS.mode == "evaluate":
      number = 0
      evaluationFile = open('distances.csv', 'w')
      sumElements = 0
      sumDistance = 0

      while validation_dataset_reader.epochs_completed < 1:
        (valid_images, valid_annotations, valid_coords, valid_keyimages, valid_keyannotations, valid_keycoords) = validation_dataset_reader.next_batch(BATCH_SIZE)

        feed_dict = {image: valid_images, keyimage:valid_keyimages, annotation: valid_annotations, keyannotation:valid_keyannotations, keycoords:valid_keycoords, coords: valid_coords, keep_probability: 1.0}
        pred, (normalizedHeatmapsTensor, predictedPointsTensor, tempSelectedError, error_mean) = sess.run([norm_networkOutput, allErrorMeasures], feed_dict=feed_dict)

        for itr in range(BATCH_SIZE):
          labelCoords = valid_coords[itr][1:3]
          labelAvgDepth = valid_coords[itr][3]
          classifiedCoords = predictedPointsTensor[itr]

          difference = labelCoords-classifiedCoords
          distance = np.linalg.norm(difference)
          distance = utils.pixelDistInCm(distance, labelAvgDepth)
          sumDistance += distance
          sumElements +=1
          line = str(distance)+"\n"
          evaluationFile.write(line)

          leslice = pred[itr,:, :,0]
          lenormheatmap = (normalizedHeatmapsTensor[itr,:, :]) * 255.0
          utils.save_image(leslice.astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(number))
          utils.save_image(lenormheatmap.astype(np.uint8), FLAGS.logs_dir, name="nhm_" + str(number))
          number += 1

      print("from "+str(sumElements)+" elements, average distance was:"+str(float(sumDistance)/sumElements))

if __name__ == "__main__":
    tf.app.run()

