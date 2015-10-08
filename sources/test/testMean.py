
import numpy
import numpy.random
__author__ = 'giulio'


a = numpy.random.rand(1, 100)*10
b = numpy.random.rand(1, 100)+a

prodMean = numpy.mean(numpy.multiply(a, b))

meanProd = numpy.mean(a) * numpy.mean(b)

print(prodMean)
print(meanProd)