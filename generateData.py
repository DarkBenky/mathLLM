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
    numberOfTokens = np.random.randint(3, INPUT_SIZE // 2)
    probabilityOfOperator = 0.3
    expression = ""
    for i in range(numberOfTokens):
        if np.random.rand() < probabilityOfOperator and i > 0 and i < numberOfTokens - 1 and expression[-1] not in OPERATORS:  # Ensure operators are not at the start or end
            expression += np.random.choice(OPERATORS)
        else:
            # Prevent leading zeros: first digit of a number must be 1-9
            if i == 0 or expression[-1] in OPERATORS:
                expression += str(np.random.randint(1, 10))
            else:
                expression += str(np.random.randint(0, 10))
    result = str(int(eval(expression)))
    expression += "="
    return expression, result


if __name__ == "__main__":
    # Example usage
    expression = "1+2+3+5=8"
    embedding = tokenizeExpression(expression)
    print(embedding)
    expression, result = generateExpression()
    print(f"Generated Expression: {expression}, Result: {result}")