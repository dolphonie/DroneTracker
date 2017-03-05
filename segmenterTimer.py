import lrs  
import timeit

lrs.startStream()
start_time = timeit.default_timer()
for i in range (0,10):
    lrs.getFrame()

print timeit.default_timer() - start_time