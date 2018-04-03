from coremltools.converters.keras import convert
from Unet import Unet

model = Unet(1, 'adam', input_width=352, input_height=288)
model.load_weights('weights/unet.hdf5')

coreml_model = convert(model, input_names='image', image_input_names='image')
coreml_model.save('unet.mlmodel')
