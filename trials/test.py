import numpy
import theano
import theano.tensor as T

location = T.tensor3(name="tt")

def set_value_at_position(a_location):
    return a_location

result, updates = theano.scan(fn=set_value_at_position,
                              outputs_info=None,
                              sequences=[location])

assign_values_at_positions = theano.function(inputs=[location], outputs=result, allow_input_downcast=True)

# test
t_list=[]
for i in range(10):
    rarr=numpy.random.uniform(-1,1,size=(5,128,7,3))
    t_list.append(rarr)
test_locations = t_list
res=assign_values_at_positions(test_locations)
print res
print res.shape
