Own implementation of recursive model this is just test of idea

example vocab: [1, 2, 3, 4, ..., +, -, =, NULL]

example input (one hot encoding)

        1   +   2   =   NULL
1       [1] [0] [0] [0] [0]
2       [0] [0] [1] [0] [0]
3       [0] [0] [0] [0] [0]
4       [0] [0] [0] [0] [0]
...
+       [0] [1] [0] [0] [0]
-       [0] [0] [0] [0] [0]
=       [0] [0] [0] [1] [0]
NULL    [0] [0] [0] [0] [1]
