from PIL import Image


def get_tensor_from_filename(filename, transform):
    img = Image.open(filename).convert("RGB")
    return transform(img)


def get_read_data_function(transform):
    return lambda x: get_tensor_from_filename(x, transform)
