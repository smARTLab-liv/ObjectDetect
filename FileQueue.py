import numpy as np
import scipy.misc as misc
import os.path
import downscaleImages
from PIL import Image
from random import shuffle

NUM_CLASSES_AND_NOTHING = downscaleImages.NUM_CLASSES_AND_NOTHING
image_size = downscaleImages.image_size

num_color_channels = downscaleImages.num_color_channels

def binaryObjExistsVector(listOfObjects):
  b = np.zeros((NUM_CLASSES_AND_NOTHING), dtype=np.int32)
  b[listOfObjects.astype(int)] = 1
  return b


class FileQueue:
    learningRecords = []

    batch_offset = 0
    epochs_completed = 0

    readFunctions = dict()
    imageBuffer = dict()

    def __init__(self, records_list):
        print("Initializing Batch Dataset Reader...")
        self.readFunctions['.png'] = self.read_image
        self.readFunctions['.jpg'] = self.read_image
        self.readFunctions['.bin'] = self.read_binary
        self.readFunctions['.npy'] = self.read_numpy

        self._read_records(records_list)

        #start off with self.batch_offset to be > len(learningRecords) - that way, we shuffle immediately
        #also, we'd increase the epochs number - so start that with -1
        self.batch_offset = len(self.learningRecords) + 1
        self.epochs_completed = -1

    def read_image(self, filename):
      return np.array(misc.imread(filename))

    def bufferImage(self, filename):
      if filename in FileQueue.imageBuffer: # using buffer as static variable, for the test cases where we use valid set == train set... normally unused
        return
      image = self._read_file(filename)
      FileQueue.imageBuffer[filename] = image

    def read_numpy(self, filename):
      return np.load(filename)

    def read_binary(self, filename):
      bytespervalue = 1 # TODO switched from float to byte...
      bytecount = image_size[1]*image_size[0]*num_color_channels*bytespervalue

      with open(filename, "rb") as f:
        chunk = f.read(bytecount)
        arr = np.fromstring(chunk, dtype=np.uint8)
        arr = arr.reshape([image_size[1], image_size[0], num_color_channels])
      return arr

    def _read_records(self, records_list):
        self.learningRecords = [self._read_record(entry) for entry in records_list]

    def _read_record(self, entry):
        objtype = entry['type']

        imageFilename = entry['image']
        keyImageFilename = entry['keyImage']

        annotation = self._transform_annot(entry['annotation'], objtype)
        keyAnnotation = self._transform_annot(entry['keyAnnotation'], objtype)

        objectTypes, coords, allCoords = self._transform_coords(entry['coords'], objtype)
        keyObjectTypes, keyCoords, _ = self._transform_coords(entry['keyCoords'], objtype)

        self.bufferImage(imageFilename)
        self.bufferImage(keyImageFilename)

        # TODO! also buffer annotations, they are not changed anymore based on keyimage

        return (imageFilename, annotation, coords, keyImageFilename, keyAnnotation, keyCoords, objectTypes, keyObjectTypes, allCoords)

    def _transform_annot(self, filename, objType):
      arr = self._read_file(filename)
      #arr = np.where(arr == objType, 1, 0) # all 'objType' fields will become 1, all others 0
      # removed the above, should be done by TF now, during testing. need entire annotation to find "all" objects.
      arr = np.expand_dims(arr, axis=3) # TODO we should be able to get rid of this
      return arr

    def _transform_coords(self, coords, objType):
      #coords[0] = 1 # setting the "type" of the object to 1 for all - will be used for lookup later, and we will only have 2 layers
      # IoU and distance error use this field to select the layer to work on - we want positive activations on layer 0, as this is what CE gives us by default.

      if(len(coords.shape) > 1):
        matchingCoord = np.array([row for row in coords if row[0] == objType]) # TODO this is a workaround... as of now, we only hand in the first coordinate to the distance (euclidean) function
        matchingCoord = np.squeeze(matchingCoord)
        if(len(matchingCoord.shape) > 1):
          #print("TODO error filename here") # TODO.
          matchingCoord = matchingCoord[0]
        objectTypes = binaryObjExistsVector(coords[:,0])
      else:
        matchingCoord = np.array(coords)
        objectTypes = binaryObjExistsVector(coords[0])

      return objectTypes, matchingCoord, coords

    def _inPlaceLoadImages(self, tuples):
      loadedImgs = [[self.imageBuffer[imageFilename], annotation, coords, self.imageBuffer[keyImageFilename], keyAnnotation, keyCoords, objectTypes, keyObjectTypes, allCoords] for (imageFilename, annotation, coords, keyImageFilename, keyAnnotation, keyCoords, objectTypes, keyObjectTypes, allCoords) in tuples]
      transverseZipped = zip(*loadedImgs)
      return transverseZipped

    def _read_file(self, filename):
      extension = os.path.splitext(filename)[1]
      readingFunction = self.readFunctions[extension]
      if(readingFunction is None):
        raise ValueError("Unknown File extension:" + extension)

      return readingFunction(filename)

    def reset_batch_offset(self, offset=0):
      self.batch_offset = offset

    def next_batch(self, batch_size):
      start = self.batch_offset
      self.batch_offset += batch_size
      if self.batch_offset > len(self.learningRecords):
        # Finished epoch
        self.epochs_completed += 1
        if self.epochs_completed > 0:
          print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
        # Shuffle the data
        shuffle(self.learningRecords)
        # Start next epoch
        start = 0
        self.batch_offset = batch_size

      end = self.batch_offset
      return self._inPlaceLoadImages(self.learningRecords[start:end])

    def get_random_batch(self, batch_size):
      if self.epochs_completed < 0:
        shuffle(self.learningRecords)
        self.epochs_completed = 0

      startIndex = np.random.randint(0, len(self.learningRecords)-batch_size)
      return self._inPlaceLoadImages(self.learningRecords[startIndex:startIndex+batch_size])
