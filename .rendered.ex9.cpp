

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/LU>

namespace py = pybind11;

// convenient matrix indexing comes for free
double get(Eigen::MatrixXd xs, int i, int j) {
    return xs(i, j);
}

// takes numpy array as input and returns double
double det(Eigen::MatrixXd xs) {
    return xs.determinant();
}

// takes numpy array as input and returns another numpy array
Eigen::MatrixXd inv(Eigen::MatrixXd xs) {
    return xs.inverse() + xs.det();
}

PYBIND11_MODULE(ex9, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("inv", &inv);
    m.def("det", &det);
}