"""Read numeric arrays for verification without NumPy or pickle."""
import ast
import math
import struct
import zipfile


def read_npy(stream):
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("Not an NPY array")
    major, minor = stream.read(2)
    length = struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]
    header = ast.literal_eval(stream.read(length).decode("latin1"))
    if header["fortran_order"]:
        raise ValueError("Fortran arrays are not supported by this saved verifier")
    formats = {"<f4": "f", "<f8": "d", "<i8": "q", "<i4": "i", "|u1": "B"}
    if header["descr"] not in formats:
        raise ValueError("Object/non-numeric arrays are not loaded")
    payload = stream.read()
    fmt = "<" + formats[header["descr"]]
    values = [v[0] for v in struct.iter_unpack(fmt, payload)]
    if len(values) != math.prod(header["shape"]):
        raise ValueError("NPY shape/payload mismatch")
    return header["shape"], values


def numeric_sample(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in ("x", "edge_index", "y", "bus_ids"):
            with archive.open(name + ".npy") as stream:
                result[name] = read_npy(stream)
    return result


def sample_edges(sample):
    shape, flat = sample["edge_index"]
    if len(shape) != 2 or shape[0] != 2:
        raise ValueError("Invalid stored edge shape")
    return list(zip(flat[:shape[1]], flat[shape[1]:]))
