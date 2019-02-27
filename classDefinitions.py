from bidict import bidict

classes = bidict({'Axis' : 0, 'Bearing' : 1, 'Bearing_Box' : 2, 'Distance_Tube' : 3, 'F20_20_B' : 4, 'F20_20_G' : 5, 'M20' : 6, 'M20_100' : 7, 'M30' : 8, 'Motor' : 9, 'R20' : 10, 'S40_40_B' : 11, 'S_40_40_G' : 12, 'nothing' : 13, 'unspecified' : 255}) #255 is translated to 13 at some point

BATCH_SIZE = 6
PRINT_EVERYTHING = False
OUTPUT_LAYERS = 2 # classes['nothing'] + 1 # 13 image types + nothing = 14

#HOLDOUT_OBJECT_DURING_TRAINING = None # [0.0, 4.0] # MUST BE list od DOUBLE, or None
