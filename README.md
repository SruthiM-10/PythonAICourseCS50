# PythonAICourseCS50
Traffic.py:
-
_Observation Log:_

    | First tried doubling amount of convolution and pooling layers
    **Result:**
        Greatly increased accuracy (96.64%)
        Not always consistent though, once got 90.13%, 93.36%
    
    | Next increased number and size of convolutional layers
    (32, (3,3)) -> (64, (2, 2))
    Result:
        Slower computation and only accurcay of 91.66%

    | Different pool sizes for convolutional layers
    (2, 2) -> (4, 4)
    Result:
        very fast computation but poor accuracy (59.39%)
    Try 2:
    (2, 2) -> (3, 3)
    Result:
        Pretty fast computation but still poor accuracy (61.69%)
    Try 3:
    (2, 2) -> (4, 4) the first time
    (2, 2) -> (1, 1) the second time
    Result:
        Decently fast computation and decent accuracy (88.43%)
    Possible improvement try 4:
    (2, 2) -> (3, 3) the first time
    (2, 2) -> (1, 1) the second time
    Result:
        good accuracy and okay speed (94.86%) but still doesn't match the efficiency of (2, 2)

    | 2 hidden layers (both 128 units)
    Result:
        Good accuracy (95.06%) but very slow speed and still worse than the first try
    Try 2:
    increase both pool sizes to (4, 4) and double number of second filters
    Result:
        Increased speed but only 84.65% accuracy
    Try 3:
    add one more hidden layer while maintaining Try 2's effects
    Result:
        Decent speed but accuracy of 91.26%
    Try 4:
    two hidden layers with 1024, 256 units while maintaining Try 2's effects
    Result:
        Very slow speed and only accuracy of 89.38%
    **Try 5:**
    1st filter = 32, 1st pool = (2, 2), 2nd filters = 128, 2nd pool = (4, 4)
    four hidden layers
    Result:
        very slow speed but 96.74%!
    Try 6:
    try 5 - 1 hidden layer
    Result:
        not that much decrease in speed and decrease in accuracy (94.06%)
    Try 7:
    Try 6 -> hidden layers have units: (64, 128, 256) and 2nd pool -> (5, 5)
        Tries showed that the best is to maximize layers closer to the output 
        which makes sense due to backpropagation
    **Try 8**
    ConvLayers = (32, 64), PoolLayers((2,2), (2,2)), Hidden Layers(128,128)
    Result:
        7ms/step so okay speed, up to 96.96% accuracy
    
    | Toggling Dropout values from try 8
    0 : 95.19%
        Accuracy was up to 98 on training but only 95.19% on testing
    0.1 : 96.65%
    **0.2** : 97.62%
    0.3: 96.35%
    0.4: 97.03%
    0.5: 97.02%
    0.7: 93.05%

Conclusion:
- 
_**FOR MAXIMUM ACCURACY**_

    Two convolutional and pool layers
    - Increasing the latter convolutional layer especially
    - (2, 2) pool layers, (3, 3) kernel matrix
    - Note! When creating pool layers watch out that you are not exceeding the amount of pixels in image
    
    Dropout = 0.2
    
    Two hidden layers

    