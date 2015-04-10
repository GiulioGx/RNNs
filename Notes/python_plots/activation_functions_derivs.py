from pylab import *
import numpy

x = numpy.arange(-2,2)
y_sigmoid = 1/(1+numpy.exp(-x));
y_relu = numpy.maximum(0,x);

y1_sigmoid = y_sigmoid .* (1-y_sigmoid);
y1_relu = numpy.maximum(0, numpy.sign(x));

#plot(t, s1,'b.',t,s2,'r.')

line_1, = plot(x, y_sigmoid,'r')
line_2, = plot(x,y_relu,'b')
line_11, = plot(x, y1_sigmoid,'y')
line_21, = plot(x,y1_relu,'p')
legend([line_1, line_2, line_11, line_21],['sigmoid','ReLU','d_sigmoid','d_ReLU'],shadow=True, fancybox=True)

xlabel('x')
ylabel('y')
grid(True)
#savefig("test.svg")
show() 
