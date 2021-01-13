from lifelines.datasets import load_gbsg2
from sklearn.model_selection import train_test_split

from xgbse.converters import convert_to_structured


def get_data():
    data = load_gbsg2()

    feat = ["age", "tsize", "pnodes", "progrec", "estrec"]
    X = data[feat]
    T = data["time"]
    E = data["cens"]

    split_params = {"test_size": 0.2, "random_state": 0}
    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        X, T, E, **split_params
    )
    X_train, X_valid, T_train, T_valid, E_train, E_valid = train_test_split(
        X_train, T_train, E_train, **split_params
    )

    y_train = convert_to_structured(T_train, E_train)
    y_test = convert_to_structured(T_test, E_test)
    y_valid = convert_to_structured(T_valid, E_valid)

    return (
        X_train,
        X_test,
        X_valid,
        T_train,
        T_test,
        T_valid,
        E_train,
        E_test,
        E_valid,
        y_train,
        y_test,
        y_valid,
        feat,
    )
