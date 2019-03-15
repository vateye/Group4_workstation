import numpy as np

#create random number
negtive_samples=np.random.randint(0,256,(100,4096))
positive_samples=np.random.randint(0,100,(30,4096))
negtive_labels=np.ones((100,1))*-1
positive_labels=np.ones((30,1))

discriminative_s(negtive_samples,negtive_labels,positive_samples,positive_labels)
