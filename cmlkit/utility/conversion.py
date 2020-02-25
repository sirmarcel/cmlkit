def convert(data, quantity, per):
    if per == "atom" or per is None:
        return quantity
    if per in ["structure", "cell", "struc", "molecule", "mol"]:
        return quantity * data.aux["n_atoms"]
    if per in ["cat", "sub", "non_O", "cation"]:
        return quantity * data.aux["n_atoms"] / data.aux["n_non_O"]


def unconvert(data, quantity, from_per):
    if from_per == "atom" or from_per is None:
        return quantity
    if from_per in ["structure", "cell", "struc", "molecule", "mol"]:
        return quantity / data.aux["n_atoms"]
    if from_per in ["cat", "sub", "non_O", "cation"]:
        return quantity * data.aux["n_non_O"] / data.aux["n_atoms"]
