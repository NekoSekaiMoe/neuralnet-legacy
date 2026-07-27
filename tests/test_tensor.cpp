#include <cassert>
#include <iostream>
#include <cmath>
#include "neuralnet/tensor.h"

using namespace nn;

void test_tensor_add() {
    Matrix m1(2, 2); m1.set_value(0, 0, 1); m1.set_value(1, 1, 2);
    Matrix m2(2, 2); m2.set_value(0, 0, 3); m2.set_value(1, 1, 4);

    Tensor t1(m1, true);
    Tensor t2(m2, true);

    Tensor t3 = add(t1, t2);
    t3.backward();

    assert(t3.data().at(0, 0) == 4);
    assert(t3.data().at(1, 1) == 6);
    assert(t1.grad().at(0, 0) == 1);
    assert(t1.grad().at(1, 1) == 1);
    assert(t2.grad().at(0, 0) == 1);
    assert(t2.grad().at(1, 1) == 1);
    std::cout << "test_tensor_add passed\n";
}

void test_tensor_matmul() {
    Matrix m1(2, 3); // 2 rows, 3 cols
    m1.set_value(0, 0, 1); m1.set_value(0, 1, 2); m1.set_value(0, 2, 3);
    m1.set_value(1, 0, 4); m1.set_value(1, 1, 5); m1.set_value(1, 2, 6);

    Matrix m2(3, 2); // 3 rows, 2 cols
    m2.set_value(0, 0, 7); m2.set_value(0, 1, 8);
    m2.set_value(1, 0, 9); m2.set_value(1, 1, 10);
    m2.set_value(2, 0, 11); m2.set_value(2, 1, 12);

    Tensor t1(m1, true);
    Tensor t2(m2, true);

    Tensor t3 = matmul(t1, t2);
    t3.backward();

    // dC/dt1 = dC * t2^T. Here dC is ones(2, 2)
    // t2^T = [7, 9, 11; 8, 10, 12]
    // dC * t2^T = [1, 1; 1, 1] * [7, 9, 11; 8, 10, 12] 
    // = [15, 19, 23; 15, 19, 23]
    
    assert(t1.grad().at(0, 0) == 15);
    assert(t1.grad().at(0, 1) == 19);
    assert(t1.grad().at(0, 2) == 23);
    assert(t1.grad().at(1, 0) == 15);
    std::cout << "test_tensor_matmul passed\n";
}

void test_tensor_relu() {
    Matrix m(2, 2);
    m.set_value(0, 0, -1); m.set_value(0, 1, 2);
    m.set_value(1, 0, 3); m.set_value(1, 1, -4);
    Tensor t(m, true);
    Tensor t2 = relu(t);
    t2.backward();
    
    assert(t2.data().at(0, 0) == 0);
    assert(t2.data().at(0, 1) == 2);
    assert(t2.data().at(1, 0) == 3);
    assert(t2.data().at(1, 1) == 0);
    
    assert(t.grad().at(0, 0) == 0);
    assert(t.grad().at(0, 1) == 1);
    assert(t.grad().at(1, 0) == 1);
    assert(t.grad().at(1, 1) == 0);
    std::cout << "test_tensor_relu passed\n";
}

void test_tensor_linear() {
    Matrix x(2, 2); // (in_features=2, batch=2)
    x.set_value(0, 0, 1); x.set_value(0, 1, 2);
    x.set_value(1, 0, 3); x.set_value(1, 1, 4);
    
    Matrix w(3, 2); // (out_features=3, in_features=2)
    w.set_value(0, 0, 1); w.set_value(0, 1, 2);
    w.set_value(1, 0, 3); w.set_value(1, 1, 4);
    w.set_value(2, 0, 5); w.set_value(2, 1, 6);
    
    Matrix b(3, 1);
    b.set_value(0, 0, 1); b.set_value(1, 0, 2); b.set_value(2, 0, 3);
    
    Tensor tx(x, true);
    Tensor tw(w, true);
    Tensor tb(b, true);
    
    Tensor ty = linear(tx, tw, tb);
    ty.backward();
    
    // y = w * x + b
    // [1 2] * [1 2] + [1] = [7 10] + [1] = [8 11]
    // [3 4]   [3 4]   [2]   [15 22]  [2]   [17 24]
    // [5 6]           [3]   [23 34]  [3]   [26 37]
    assert(ty.data().at(0, 0) == 8);
    assert(ty.data().at(0, 1) == 11);
    assert(ty.data().at(1, 0) == 17);
    assert(ty.data().at(1, 1) == 24);
    assert(ty.data().at(2, 0) == 26);
    assert(ty.data().at(2, 1) == 37);
    
    // b.grad should be [2, 2, 2] since out.grad is ones
    assert(tb.grad().at(0, 0) == 2);
    assert(tb.grad().at(1, 0) == 2);
    assert(tb.grad().at(2, 0) == 2);
    
    std::cout << "test_tensor_linear passed\n";
}

#include "test_runner.h"

TestEntry tests[] = {
    {"tensor_add", test_tensor_add},
    {"tensor_matmul", test_tensor_matmul},
    {"tensor_relu", test_tensor_relu},
    {"tensor_linear", test_tensor_linear}
};

int main(int argc, char** argv) {
    return run_tests(argc, argv, tests, sizeof(tests)/sizeof(tests[0]));
}
