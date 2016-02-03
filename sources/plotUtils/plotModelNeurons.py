import PIL
import matplotlib.pyplot as plt
from PIL import Image

from Configs import Configs
from model import RNN
from task.AdditionTask import AdditionTask
from task.XorTaskHot import XorTaskHot
import numpy

seed = 132
task = AdditionTask(144, seed)
#out_dir = Configs.output_dir + str(task)
#net = Rnn.load_model('/home/giulio/RNNs/models/completed/100 hidden/add_task, min_length: 144_average/model.npz')
net = RNN.load_model('/home/giulio/RNNs/models/add_task, min_length: 144/current_model.npz')
batch = task.get_batch(1)

y, h = net.net_ouput_numpy(batch.inputs)

h_mean = numpy.mean(h, axis=2)

# tanh
saturation_b1 = numpy.tanh(2)
saturation_b2 = numpy.tanh(-2)

# relu
# saturation_b1 = numpy.tanh(0.)
# saturation_b2 = numpy.tanh(-numpy.inf)

sat_color1 = [0, 0, 255]
sat_color2 = [255, 0, 0]
non_sat_color = [0, 255, 0]

rgbArray = numpy.zeros((h.shape[0], h.shape[1], 3), 'uint8')
rgbArray[:, :, :] = non_sat_color
rgbArray[h_mean >= saturation_b1, :] = sat_color1
rgbArray[h_mean <= saturation_b2, :] = sat_color2

print('sum', sum(h_mean>saturation_b1) + sum(h_mean<saturation_b2))

img = Image.fromarray(rgbArray)
basewidth = 500
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
img.show()

print('h_shape', h_mean.shape)
print('h', h_mean)

plt.matshow(h_mean)
plt.show()


