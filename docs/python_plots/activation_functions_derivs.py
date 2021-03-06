from pylab import *
import numpy


linewidth= 2.0;
x_start=-4;
x_end = 5;
xlim(x_start,x_end-1);
x = numpy.arange(x_start,x_end,0.01)

##tanh
ylim([-1.2,1.2])
y_tanh = (numpy.exp(x)-numpy.exp(-x))/(numpy.exp(x)+numpy.exp(-x));
y1_tanh = 1-y_tanh*y_tanh;



line_3, = plot(x, y_tanh,color='g',linewidth=linewidth);
line_31, = plot(x, y1_tanh,color='y',linewidth=linewidth);

legend([line_3, line_31],['tanh','derivative'],shadow=True, fancybox=True,loc=0)

xlabel('x')
ylabel('y')
grid(True)
yticks(numpy.arange(-1.2,1.2,0.2))
xticks(numpy.arange(x_start,x_end,1))

#sigmoid
figure();
ylim([-0.2,1])
y_sigmoid = 1/(1+numpy.exp(-x));
y1_sigmoid = y_sigmoid * (1-y_sigmoid);



line_1, = plot(x, y_sigmoid,color='g',linewidth=linewidth);
line_11, = plot(x, y1_sigmoid,color='y',linewidth=linewidth);

legend([line_1, line_11],['sigmoid','derivative'],shadow=True, fancybox=True,loc=0)

xlabel('x')
ylabel('y')
grid(True)
yticks(numpy.arange(-0.2,1,0.2))
xticks(numpy.arange(x_start,x_end,1))


#ReLU
figure();
ylim([-0.2,1.2])
xlim(x_start,x_end-1);
xlabel('x')
ylabel('y')
grid(True)
yticks(numpy.arange(-0.2,1.2,0.2))
xticks(numpy.arange(x_start,x_end,1))

y_relu = numpy.maximum(0,x);
y1_relu = numpy.maximum(0, numpy.sign(x));

line_2, = plot(x,y_relu,color='g',linewidth=linewidth);
line_21, = plot(x,y1_relu,color='y',linewidth=linewidth);


legend([line_2, line_21],['ReLU','derivative'],shadow=True, fancybox=True,loc=0)



#savefig("test.svg")
show() 
