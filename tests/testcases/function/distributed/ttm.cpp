//
// Created by 丁明朔 on 2022/3/11.
//

#include "FunctionDistributedTest.hpp"

TEST_F(FunctionDistributedTest, TTM1) {
    // Initialization
    auto *dis_global = new DistributionGlobal();
    Tensor<double> m1(dis_global, {3, 4});
    for (size_t i = 0; i < m1.size(); i++) {
        m1[i] = (double) i;
    }
    Tensor<double> m2(dis_global, {2, 3});
    for (size_t i = 0; i < m2.size(); i++) {
        m2[i] = (double) i;
    }
    // Calculate
    t = Function::ttm<double>(t, m1, 1);
    t = Function::ttm<double>(t, m2, 2);
    // Gather
    auto ans = Function::gather(t);
    // Ground Truth
    if (mpi_rank() == 0) {
        double ground_truth[] = {3564.0, 3672.0, 3780.0, 4416.0, 4524.0, 4122.0,
                                 4254.0, 4386.0, 5148.0, 5280.0, 4680.0, 4836.0,
                                 4992.0, 5880.0, 6036.0, 5220.0, 5382.0, 5544.0,
                                 6540.0, 6702.0, 6021.0, 6219.0, 6417.0, 7614.0,
                                 7812.0, 6822.0, 7056.0, 7290.0, 8688.0,
                                 8922.0};
        for (size_t i = 0; i < ans.size(); i++) {
            EXPECT_DOUBLE_EQ(ans[i], ground_truth[i]);
        }
    }
}