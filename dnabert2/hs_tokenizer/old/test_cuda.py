print('DNA-BERT Container')

from torch import cuda as tc
x=tc.is_available()
print('Is cuda is_available:', x)
if x:
    print(tc.get_device_name())
else:
    print('running on cpu...')