#include <cstdio>
extern "C" int run_test();
int main() {
    printf("Vector addition %s\n", run_test() ? "PASSED" : "FAILED");
    return 0;
}
