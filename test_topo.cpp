#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <neuralnet/nn/nn.h>

int main() {
    auto t1_node = std::make_shared<nn::TensorNode>(nn::Matrix(1, 1), true, true);
    auto t2_node = std::make_shared<nn::TensorNode>(nn::Matrix(1, 1), true, true);
    auto t3_node = std::make_shared<nn::TensorNode>(nn::Matrix(1, 1), true, false);
    
    t3_node->children.push_back(t1_node);
    t3_node->children.push_back(t2_node);
    
    std::vector<std::string> exec_order;
    t1_node->backward_op = [&]() { exec_order.push_back("t1"); };
    t2_node->backward_op = [&]() { exec_order.push_back("t2"); };
    t3_node->backward_op = [&]() { exec_order.push_back("t3"); };
    
    nn::Tensor t3(t3_node);
    t3.backward();
    
    assert(exec_order.size() == 3);
    assert(exec_order[0] == "t3");
    assert(exec_order[1] == "t2");
    assert(exec_order[2] == "t1");
    
    std::cout << "t1 t2 t3 \n";
    return 0;
}
