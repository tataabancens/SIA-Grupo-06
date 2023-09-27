def partition(x,y,p):
    training_qty = int(len(x) * p)
    if training_qty == len(x) and p < 1.0:
        training_qty -= 1
    new_x = x[:training_qty]
    new_y = y[:training_qty]
    test_x = x[training_qty:]
    test_y =y[training_qty:]
    return new_x, new_y, test_x, test_y