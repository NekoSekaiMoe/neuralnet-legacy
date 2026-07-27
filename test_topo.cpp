#include <iostream>
#include <vector>
#include <unordered_set>
#include <memory>

struct TensorNode {
    std::vector<std::shared_ptr<TensorNode>> children;
    std::string name;
};

int main() {
    auto t1 = std::make_shared<TensorNode>(); t1->name = "t1";
    auto t2 = std::make_shared<TensorNode>(); t2->name = "t2";
    auto t3 = std::make_shared<TensorNode>(); t3->name = "t3";
    t3->children.push_back(t1);
    t3->children.push_back(t2);
    auto node_ = t3;

    std::vector<std::shared_ptr<TensorNode>> topo;
    std::unordered_set<TensorNode*> visited;
    
    std::vector<std::shared_ptr<TensorNode>> stack;
    std::unordered_set<TensorNode*> expanded;
    stack.push_back(node_);
    while (!stack.empty()) {
        auto v = stack.back();
        if (!v) {
            stack.pop_back();
            continue;
        }
        if (expanded.find(v.get()) != expanded.end()) {
            stack.pop_back();
            continue;
        }
        if (visited.find(v.get()) == visited.end()) {
            visited.insert(v.get());
            for (auto it = v->children.rbegin(); it != v->children.rend(); ++it) {
                if (*it && visited.find((*it).get()) == visited.end()) {
                    stack.push_back(*it);
                }
            }
        } else {
            stack.pop_back();
            expanded.insert(v.get());
            topo.push_back(v);
        }
    }

    for (auto& n : topo) {
        std::cout << n->name << " ";
    }
    std::cout << "\n";
    return 0;
}
