Own implementation of recursive model this is just test of idea

example vocab: [1, 2, 3, 4, ..., +, -, =, NA]

example input (one hot encoding)

        1   +   2   =   NA
1       [1] [0] [0] [0] [0]
2       [0] [0] [1] [0] [0]
3       [0] [0] [0] [0] [0]
4       [0] [0] [0] [0] [0]
...
+       [0] [1] [0] [0] [0]
-       [0] [0] [0] [0] [0]
=       [0] [0] [0] [1] [0]
NA      [0] [0] [0] [0] [1]

output will be feed back to model as input on n step we read the output neuron for value and evalue against real evalueted value

        3   NA  NA  NA  NA      + model float out put     
1       [0] [0] [0] [0] [0]     [3.0]
2       [0] [0] [1] [0] [0]
3       [1] [0] [0] [0] [0]
4       [0] [0] [0] [0] [0]
...
+       [0] [1] [0] [0] [0]
-       [0] [0] [0] [0] [0]
=       [0] [0] [0] [1] [0]
NA      [0] [0] [0] [0] [1]
