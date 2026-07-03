from defines import VOCAB, INPUT_SIZE, OPERATORS
import numpy as np

def tokenizeExpression(exp: str):
    if len(exp) > INPUT_SIZE:
        raise ValueError(f"Expression length exceeds maximum input size of {INPUT_SIZE}.")

    embedding = np.zeros((INPUT_SIZE, len(VOCAB)))
    # one hot encode the expression
    for i, char in enumerate(exp):
        if char not in VOCAB:
            raise ValueError(f"Character '{char}' not in vocabulary.")
        embedding[i, VOCAB.index(char)] = 1
    
    for i in range(len(exp), INPUT_SIZE):
        embedding[i, VOCAB.index("EMPTY")] = 1

    return embedding

def generateExpression():
    num_numbers = np.random.randint(2, 3)
    numbers = []
    for _ in range(num_numbers):
        if np.random.rand() < 0.5:
            numbers.append(str(np.random.randint(1, 10)))
        else:
            numbers.append(str(np.random.randint(10, 100)))

    expression = numbers[0]
    for i in range(1, num_numbers):
        op = np.random.choice(["+", "-"])
        expression += op + numbers[i]

    result = eval(expression)
    expression += "="
    return expression, result


if __name__ == "__main__":
    # Example usage
    expression = "1+2+3+5=8"
    embedding = tokenizeExpression(expression)
    print(embedding)
    expression, result = generateExpression()
    print(f"Generated Expression: {expression}, Result: {result}")