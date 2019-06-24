from models.utils import load_mnist

# np.set_printoptions(threshold=sys.maxsize)


(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

print(y_train.shape)
