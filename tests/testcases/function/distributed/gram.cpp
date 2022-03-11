//
// Created by 丁明朔 on 2022/3/11.
//

#include "FunctionDistributedTest.hpp"

TEST_F(FunctionDistributedTest, Gram1) {
// Calculate
    auto ans = Function::gram<double>(t, 0);
// Ground Truth
    if (mpi_rank() == 0) {
        double ground_truth[] = {7665.0, 7908.0, 8151.0, 9720.0, 9963.0, 7908.0,
                                 8163.0, 8418.0, 10062.0, 10317.0, 8151.0,
                                 8418.0, 8685.0, 10404.0, 10671.0, 9720.0,
                                 10062.0, 10404.0, 12620.0, 12962.0, 9963.0,
                                 10317.0, 10671.0, 12962.0, 13316.0};
        for (size_t i = 0; i < ans.size(); i++) {
            EXPECT_DOUBLE_EQ(ans[i], ground_truth[i]);
        }
    }
}