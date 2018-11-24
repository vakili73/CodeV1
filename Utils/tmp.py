from keras.models import load_model
model = load_model('best-model.h5') #See 'How to export keras models?' to generate this file before loading it.
from keras.utils import plot_model
plot_model(model, to_file='best-model.png')